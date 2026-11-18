"""Rebuild the full ordinal teacher and single-backbone student inside every outer people split.

Training text coverage uses IDs 1/2 versus 1/2/3/4/7/8, development IDs 5/6,
and outer IDs 9/10/11/12. No outer voice or text is used for model selection.
"""
import argparse
import itertools
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import rankdata
from runtime import ROOT,write_json
from dataset import load_rating_graph,unit
from perceptual_rank import bank,transform,train_triples
from wave_folds import paired_labels
from wave_student import batch
from ssl_rank_student import SSLMap
from reference_student import read


def rate(z,labels):
    assignments=list(itertools.permutations(range(z.shape[1]),2));scores=[]
    for a,b,y in labels:
        v=np.stack([(z[a,i]*z[b,j]).sum(-1) for i,j in assignments])
        r=rankdata(v,axis=1);r-=r.mean(1,keepdims=True);g=rankdata(y);g-=g.mean()
        scores.append(float(np.mean(r@g/np.maximum(np.linalg.norm(r,axis=1)*np.linalg.norm(g),1e-12))))
    return float(np.mean(scores))


def make_teacher(x,tr,va,labels,triples,seed):
    flat=x[tr,:4].reshape(-1,x.shape[-1]);mu=flat.mean(0);_,_,vt=np.linalg.svd(flat-mu,full_matrices=False)
    t=torch.tensor((x-mu)@vt[:64].T,dtype=torch.float32,device='cuda')
    torch.manual_seed(seed)
    weight=torch.nn.Parameter(torch.eye(16,64,device='cuda'));prior=weight.detach().clone()
    opt=torch.optim.AdamW([weight],lr=.002)
    best=-np.inf
    for step in range(1001):
        if step%100==0:
            with torch.no_grad():
                val=rate(F.normalize(F.linear(t[va,4:6],weight),dim=-1).cpu().numpy(),labels)
                if val>best:best=val;state=weight.detach().clone();chosen=step
        if step==1000:break
        z=F.normalize(F.linear(t[tr,:4],weight),dim=-1)
        a,b,c=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
        j=torch.randint(4,(3,len(a)),device='cuda')
        diff=(z[a,j[0]]*(z[b,j[1]]-z[c,j[2]])).sum(-1)
        loss=F.softplus(-6*diff).mean()+.02*(weight-prior).square().sum()/16
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():return F.normalize(F.linear(t,state),dim=-1).cpu().numpy(),chosen


@torch.inference_mode()
def measure(net,waves,people,slots,target,labels,durations):
    net.eval();ids=[i*12+s for i in people for s in slots];result={}
    original=target[np.ix_(people,slots)].reshape(-1,16)
    for sec in durations:
        chunks=[]
        for start in range(0,len(ids),4):
            x,n=batch(waves,ids[start:start+4],sec)
            with torch.autocast('cuda',dtype=torch.bfloat16):chunks.append(net(x,n).cpu().numpy())
        z=np.concatenate(chunks)
        result[str(sec)]=dict(srcc=rate(z.reshape(len(people),len(slots),16),labels),
            teacher_l2=float(np.linalg.norm(z-original,axis=1).mean()))
    return result


def student(waves,tr,va,te,target,labels,seed,slots,steps):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);net=SSLMap('frozen').cuda()
    learn=[p for p in net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(learn,lr=.0005,weight_decay=.01)
    lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00003)
    y=torch.from_numpy(target).cuda();best=-np.inf;history=[];start=time.monotonic()
    for step in range(1,steps+1):
        net.train()
        people=rng.choice(tr,16,replace=False);which=rng.choice(slots,len(people))
        x,n=batch(waves,people*12+which,float(rng.choice([1,2,3,4,5])),rng)
        valid=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
        amp=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
        db=torch.rand((16,1),device='cuda')*23+12
        x=(x+torch.randn_like(x)*amp*10**(-db/20)*(torch.rand((16,1),device='cuda')<.4))*valid
        with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x,n)
        ref=y[people,which];loss=(z-ref).square().sum(-1).mean()+F.mse_loss(z@z.T,ref@ref.T)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learn,3);opt.step();lr.step()
        if step%300==0 or step==steps:
            report=measure(net,waves,va,[4,5],target,labels[0],(0,2))
            utility=float(np.mean([v['srcc']-.5*v['teacher_l2'] for v in report.values()]))
            history.append([step,utility])
            if utility>best:
                best=utility;chosen=step;state=[p.detach().clone() for p in learn]
    with torch.no_grad():
        for p,v in zip(learn,state):p.copy_(v)
    result=measure(net,waves,te,[8,9,10,11],target,labels[1],(0,1,2,4))
    return dict(steps=chosen,history=history,outer=result,elapsed=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,required=True);p.add_argument('--steps',type=int,default=1200)
    args=p.parse_args();data=load_rating_graph(('train','dev'));names,blocks=bank(data,include_reference=False);wave_names,waves,_=read(data,include_reference=False)
    if names!=wave_names:raise ValueError('Audio and frame people differ')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(9000+args.repeat)
    groups=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    folds=[np.concatenate([g[j::5] for g in groups]) for j in range(5)]
    folder=ROOT/'runs'/f'nested_ssl_{args.repeat}_{args.steps}';folder.mkdir(exist_ok=True)
    for fold in range(5):
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        labels=(paired_labels(data,names,va),paired_labels(data,names,te));triples=train_triples(tr,names,data)
        X=np.concatenate(transform(blocks[:3],tr),axis=-1)/np.sqrt(3)
        target,teacher_step=make_teacher(X,tr,va,labels[0],triples,881+fold)
        teacher=rate(target[te,8:12],labels[1])
        for seed in (76,431):
            for slots in ([0,1],[0,1,2,3,6,7]):
                dest=folder/f'{fold}_seed{seed}_texts{len(slots)}.json'
                if dest.exists():continue
                result=student(waves,tr,va,te,target,labels,seed,slots,args.steps)
                result.update(teacher_outer=teacher,teacher_step=teacher_step,fold=fold,repeat=args.repeat,seed=seed,train_texts=[s+1 for s in slots])
                stage=dest.with_suffix('.partial');write_json(stage,result);stage.replace(dest)
                print(args.repeat,fold,seed,len(slots),round(result['outer']['0']['srcc'],3),round(result['elapsed']),flush=True)


if __name__=='__main__':main()
