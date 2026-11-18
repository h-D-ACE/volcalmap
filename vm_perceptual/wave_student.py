"""Small waveform-to-map student with paired full-utterance targets and optional rating supervision."""
import argparse,json,math,time
from pathlib import Path
from runtime import ROOT, write_json
from dataset import load_data
from train_map import correlations
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio


class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.depth=nn.Conv1d(channels,channels,5,padding=2*dilation,dilation=dilation,groups=channels)
        self.mix=nn.Linear(channels,channels)
        self.norm=nn.LayerNorm(channels)

    def forward(self,x,mask):
        h=self.depth(x).transpose(1,2)
        h=self.mix(F.silu(self.norm(h))).transpose(1,2)
        return (x+h)*mask


class WaveMap(nn.Module):
    def __init__(self,channels=96):
        super().__init__()
        self.mel=torchaudio.transforms.MelSpectrogram(16000,n_fft=512,win_length=400,hop_length=160,n_mels=64,power=2)
        self.inp=nn.Linear(64,channels)
        self.blocks=nn.ModuleList(TemporalBlock(channels,d) for d in (1,2,4,8))
        self.out=nn.Sequential(nn.LayerNorm(2*channels),nn.Linear(2*channels,96),nn.SiLU(),nn.Linear(96,16))
        self.alpha=nn.Parameter(torch.tensor(.8));self.log_beta=nn.Parameter(torch.tensor(0.))

    def forward(self,wave,length):
        m=self.mel(wave).clamp_min(1e-8).log()
        mask=(torch.arange(m.shape[-1],device=m.device)[None,:] <= length[:,None]//160)[:,None,:]
        m=(m-m.masked_fill(~mask,-100.).amax((1,2),keepdim=True)).clamp(-18,0)/6
        h=self.inp(m.transpose(1,2)).transpose(1,2)*mask
        for layer in self.blocks:h=layer(h,mask)
        n=mask.sum(-1).clamp_min(1)
        mean=h.sum(-1)/n
        std=((h.square().sum(-1)/n-mean.square()).clamp_min(1e-6)).sqrt()
        return F.normalize(self.out(torch.cat([mean,std],dim=-1)),dim=-1)


def load_records():
    data=load_data();records=[];by={}
    for i,s in enumerate(data['speakers']):
        group=data['manifest']['split'][s]
        cache=np.load(ROOT/'features/crop_records'/f'{s}.npz')
        for j in np.flatnonzero(cache['condition'][:,0]==0):
            name=str(cache['names'][j]);path=ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit'/name
            wave,sr=sf.read(path,dtype='float32')
            g=math.gcd(sr,16000);wave=resample_poly(wave,16000//g,sr//g).astype('float32')
            by.setdefault(i,[]).append(len(records))
            records.append(dict(wave=wave,target=cache['targets'][j],speaker=i,split=group,name=name))
    return data,records,by


def batch(records,indices,seconds,rng=None):
    nmax=max(len(records[i]['wave']) for i in indices) if seconds==0 else int(seconds*16000)
    x=np.zeros((len(indices),nmax),np.float32);length=[]
    for j,i in enumerate(indices):
        w=records[i]['wave'];n=min(len(w),nmax);limit=len(w)-n
        start=limit//2 if rng is None else rng.integers(limit+1)
        x[j,:n]=w[start:start+n];length.append(n)
    return torch.from_numpy(x).cuda(),torch.tensor(length,device='cuda')


@torch.no_grad()
def assess(model,data,records,by,split):
    model.eval()
    ids=[i for i,r in enumerate(records) if r['split']==split]
    stage={}
    for sec in (0,1,2,4):
        output=[]
        for k in range(0,len(ids),24):
            x,n=batch(records,ids[k:k+24],sec)
            output.append(model(x,n).cpu().numpy())
        z=np.concatenate(output)
        truth=np.array([records[i]['target'] for i in ids])
        diff=np.linalg.norm(z-truth,axis=1);loc={i:j for j,i in enumerate(ids)}
        pair=[p for p in data['pairs'] if p['split']==split]
        a,b=([p[key] for p in pair] for key in ('a','b'))
        ranks=[]
        for i in range(4):
            for j in range(4):
                if i==j:continue
                za=np.array([z[loc[by[s][i]]] for s in a]);zb=np.array([z[loc[by[s][j]]] for s in b])
                ranks.append(correlations(pair,-np.linalg.norm(za-zb,axis=1))['mean_spearman'])
        stage[str(sec)]=dict(mean_error=float(diff.mean()),median_error=float(np.median(diff)),single_srcc=float(np.mean(ranks)),n=len(diff))
    return stage


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=76)
    p.add_argument('--rating-weight',type=float,default=0);p.add_argument('--augment',action='store_true')
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--tag',required=True);p.add_argument('--confirm-test',action='store_true')
    a=p.parse_args();run=ROOT/'runs'/a.tag;run.mkdir(exist_ok=True,parents=True)
    torch.manual_seed(a.seed);np.random.seed(a.seed);rng=np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,records,by=load_records();model=WaveMap().cuda()
    if a.confirm_test:
        ck=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(ck['state']);write_json(run/'test.json',assess(model,data,records,by,'test'));return
    if (run/'best.pt').exists():raise FileExistsError('Use a new experiment tag')
    train=[i for i,s in enumerate(data['speakers']) if data['manifest']['split'][s]=='train']
    labels=np.full((100,100),np.nan,np.float32)
    for r in data['pairs']:
        if r['split']=='train':labels[r['a'],r['b']]=labels[r['b'],r['a']]=r['score']
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.steps,eta_min=.00005)
    best=-1e9;history=[]
    for step in range(1,a.steps+1):
        model.train();speakers=rng.choice(train,32,replace=False)
        ids=[rng.choice(by[s]) for s in speakers]
        sec=float(rng.choice([1.,2.,3.,4.,6.]))
        x,n=batch(records,ids,sec,rng)
        if a.augment:
            scale=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
            db=torch.empty((len(x),1),device='cuda').uniform_(12,35)
            noisy=torch.rand((len(x),1),device='cuda')<.4
            mask=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
            x=(x+torch.randn_like(x)*scale*10**(-db/20)*noisy)*mask
        target=torch.tensor(np.asarray([records[i]['target'] for i in ids]),device='cuda')
        z=model(x,n);distill=(z-target).square().sum(-1).mean()
        matrix=torch.from_numpy(labels[np.ix_(speakers,speakers)]).cuda();mask=torch.isfinite(matrix)
        d2=(z[:,None]-z[None,:]).square().sum(-1)
        pred=3*torch.tanh(model.alpha-model.log_beta.clamp(-5,5).exp()*d2[mask])
        ratings=F.huber_loss(pred,matrix[mask]);loss=distill+a.rating_weight*ratings
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();schedule.step()
        if step%250==0 or step==a.steps:
            scores=assess(model,data,records,by,'dev')
            error=np.mean([r['mean_error'] for r in scores.values()])
            val=scores['0']['single_srcc']-.5*error
            history.append(dict(step=step,scores=scores,utility=float(val)))
            if val>best:
                best=val;torch.save(dict(state=model.state_dict(),step=step,dev=scores,utility=float(val),config=vars(a),params=sum(x.numel() for x in model.parameters())),run/'best.pt')
            write_json(run/'dev.json',history);print(step,val,flush=True)
    print('done',a.tag,best,flush=True)


if __name__=='__main__':
    main()
