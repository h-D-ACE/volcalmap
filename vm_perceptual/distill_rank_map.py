"""Distill our full three-frontend perceptual map into a single-waveform network."""
import argparse
import math
import numpy as np
import torch
from torch.nn import functional as F
from runtime import ROOT,write_json,file_hash
from wave_student import WaveMap,batch,load_records
from reference_student import AttentiveMap
from perceptual_rank import metrics
from wave_folds import paired_labels


@torch.inference_mode()
def evaluate(model,records,ids_by,data,targets,split):
    model.eval()
    names=[s for s in data['speakers'] if data['manifest']['split'][s]==split]
    idx=[i for s in names for i in ids_by[data['speakers'].index(s)]]
    if len(idx)!=4*len(names):raise ValueError('Expected four evaluation sentences per voice')
    labels=paired_labels(data,names,np.arange(len(names)));report={}
    for sec in (0,1,2,4):
        chunks=[model(*batch(records,idx[k:k+20],sec)).cpu().numpy() for k in range(0,len(idx),20)]
        z=np.concatenate(chunks)
        mae=np.linalg.norm(z-targets[idx],axis=1)
        scores=metrics(z.reshape(len(names),4,-1),labels)
        report[str(sec)]=dict(**scores,mean_teacher_error=float(mae.mean()),median_teacher_error=float(np.median(mae)))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--recipe',choices=('mean','attention','attention_relation'),required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=3600);ap.add_argument('--confirm-test',action='store_true')
    a=ap.parse_args();data,records,by=load_records()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    model=(WaveMap() if a.recipe=='mean' else AttentiveMap()).cuda()
    folder=ROOT/'runs'/f'rankstudent_{a.recipe}_{a.seed}_{a.steps}';folder.mkdir(exist_ok=True)
    version=file_hash(ROOT/'runs/rank_map_own.npz');lookup={}
    targets=[]
    for rec in records:
        person=data['speakers'][rec['speaker']]
        if person not in lookup:
            files=sorted((ROOT/'data_jvs'/person/'parallel100/wav24kHz16bit').glob('*.wav'))
            z=np.load(ROOT/'features/rank_map_own'/f'{person}.npy')
            lookup[person]=dict(zip([p.name for p in files],z,strict=True))
        targets.append(lookup[person][rec['name']])
    targets=np.array(targets)
    if a.confirm_test:
        ck=torch.load(folder/'best.pt',weights_only=True)
        if ck['teacher_sha256']!=version:raise ValueError('Teacher changed')
        model.load_state_dict(ck['state'])
        write_json(folder/'test.json',evaluate(model,records,by,data,targets,'test'));return
    if (folder/'best.pt').exists():raise FileExistsError(folder)
    train=[i for i,s in enumerate(data['speakers']) if data['manifest']['split'][s]=='train']
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00005)
    target=torch.from_numpy(targets).cuda()
    best=-np.inf;history=[]
    for step in range(1,a.steps+1):
        people=rng.choice(train,32,replace=False);ids=[rng.choice(by[p]) for p in people]
        seconds=float(rng.choice([1,2,3,4,6]));model.train()
        x,n=batch(records,ids,seconds,rng)
        mask=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
        power=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
        db=torch.empty(len(x),1,device='cuda').uniform_(12,35)
        x=(x+torch.randn_like(x)*power*10**(-db/20)*(torch.rand(len(x),1,device='cuda')<.4))*mask
        z=model(x,n);ref=target[ids]
        loss=(z-ref).square().sum(-1).mean()
        if a.recipe=='attention_relation':
            loss=loss+2*F.mse_loss(z@z.T,ref@ref.T)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();sched.step()
        if step%300==0 or step==a.steps:
            scores=evaluate(model,records,by,data,targets,'dev')
            utility=np.mean([r['single_srcc']-.5*r['mean_teacher_error'] for r in scores.values()])
            history.append(dict(step=step,utility=float(utility),scores=scores))
            if utility>best:
                best=utility;torch.save(dict(state=model.state_dict(),teacher_sha256=version,step=step,dev=scores,
                    recipe=a.recipe,seed=a.seed,parameters=sum(p.numel() for p in model.parameters())),folder/'best.pt')
            write_json(folder/'dev.json',history)
    print(a.recipe,a.seed,best,flush=True)


if __name__=='__main__':main()
