"""Learn compact waveform timbre coordinates from public perceptual features.

One paired, full-recording teacher vector for each training clip; sentence crops
never inherit an identity classification target. Hyperparameters use development only.
"""
import argparse
import math
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
from runtime import ROOT, write_json
from dataset import load_data, unit
from wave_student import WaveMap,batch
from wave_folds import paired_labels
from perceptual_rank import metrics


class AttentiveMap(WaveMap):
    def __init__(self):
        super().__init__()
        self.attn=nn.Sequential(nn.Conv1d(96,32,1),nn.Tanh(),nn.Conv1d(32,96,1))
        nn.init.zeros_(self.attn[-1].weight);nn.init.zeros_(self.attn[-1].bias)

    def forward(self,wave,length):
        m=self.mel(wave).clamp_min(1e-8).log()
        mask=(torch.arange(m.shape[-1],device=m.device)[None,:]<=length[:,None]//160)[:,None,:]
        m=(m-m.masked_fill(~mask,-100.).amax((1,2),keepdim=True)).clamp(-18,0)/6
        h=self.inp(m.transpose(1,2)).transpose(1,2)*mask
        for layer in self.blocks:h=layer(h,mask)
        weights=self.attn(h).masked_fill(~mask,-1e4).softmax(-1)
        mean=(h*weights).sum(-1)
        std=((h.square()*weights).sum(-1)-mean.square()).clamp_min(1e-6).sqrt()
        return F.normalize(self.out(torch.cat((mean,std),dim=-1)),dim=-1)


def read(data):
    names=sorted(s for s in data['speakers'] if data['manifest']['split'][s]!='test')
    waves=[];reference=[]
    for s in names:
        with np.load(ROOT/'features/voxsim_large'/f'{s}.npz') as f:
            clips=sorted((p,v) for p,v in zip(f['names'],f['vectors']) if '/parallel100/' in str(p))
        assert len(clips)==12
        reference.append([v for p,v in clips])
        for p,v in clips:
            x,sr=sf.read(ROOT/'data_jvs'/p,dtype='float32');g=math.gcd(16000,sr)
            waves.append({'wave':resample_poly(x,16000//g,sr//g).astype('float32')})
    return names,waves,np.array(reference)


def projection(reference,train):
    flat=reference[train,:4].reshape(-1,reference.shape[-1])
    mean=flat.mean(0);_,_,axes=np.linalg.svd(flat-mean,full_matrices=False)
    return unit((reference-mean)@axes[:16].T).astype('float32')


@torch.inference_mode()
def measure(model,waves,people,start,labels):
    model.eval()
    report={}
    for seconds in (0,1,2,4):
        ids=[p*12+i for p in people for i in range(start,start+4)]
        chunks=[model(*batch(waves,ids[k:k+16],seconds)).cpu().numpy() for k in range(0,len(ids),16)]
        report[str(seconds)]=metrics(np.concatenate(chunks).reshape(len(people),4,-1),labels)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--pool',choices=('mean','attention'),required=True)
    parser.add_argument('--recipe',choices=('pca','gram'),required=True)
    a=parser.parse_args();data=load_data();names,waves,reference=read(data)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(9000+a.repeat)
    by_sex=[rng.permutation([k for k,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([p[i::5] for p in by_sex]) for i in range(5)]
    folder=ROOT/'runs'/f'refstudent_{a.repeat}_{a.pool}_{a.recipe}_{a.steps}';folder.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[te,va])
        vl=paired_labels(data,names,va);tl=paired_labels(data,names,te)
        low=projection(reference,tr);ref=torch.from_numpy(reference).cuda();proj=torch.from_numpy(low).cuda()
        fit_mean=ref[tr,:4].reshape(-1,ref.shape[-1]).mean(0)
        seed=771+10*fold+100*a.repeat;torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=(WaveMap() if a.pool=='mean' else AttentiveMap()).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00005)
        best=-np.inf;history=[];tic=time.monotonic()
        for step in range(1,a.steps+1):
            model.train()
            who=rng.choice(tr,32,replace=False);j=rng.integers(4,size=len(who))
            x,n=batch(waves,who*12+j,float(rng.choice([1,2,3,4,6])),rng)
            valid=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
            power=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
            db=torch.empty(len(x),1,device='cuda').uniform_(12,35)
            x=(x+torch.randn_like(x)*power*10**(-db/20)*(torch.rand(len(x),1,device='cuda')<.4))*valid
            z=model(x,n);loss=(z-proj[who,j]).square().sum(-1).mean()
            if a.recipe=='gram':
                src=F.normalize(ref[who,j]-fit_mean,dim=-1)
                mask=~torch.eye(len(x),dtype=torch.bool,device='cuda')
                loss=.2*loss+4*F.mse_loss((z@z.T)[mask],(src@src.T)[mask])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();lr.step()
            if step%250==0 or step==a.steps:
                scores=measure(model,waves,va,4,vl)
                score=np.mean([v['single_srcc'] for v in scores.values()])
                history.append([step,float(score)])
                if score>best:
                    best=score;chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state)
        test=measure(model,waves,te,8,tl)
        write_json(dest,dict(fold=fold,selected_step=chosen,dev_history=history,outer=test,
            reference_full=metrics(reference[te,8:12],tl),pca16_full=metrics(low[te,8:12],tl),
            n_train=len(tr),n_test=len(te),seconds=time.monotonic()-tic))
        print(a.repeat,a.pool,a.recipe,fold,test['0'],flush=True)


if __name__=='__main__':main()
