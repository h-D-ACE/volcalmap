"""A label-allocation learning curve with exactly matched numbers of listening votes.

All configurations see the same unlabeled training audio. They differ in the number of
voices covered by the human labels and the number of ratings available per voice pair.
"""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from runtime import ROOT,write_json
from dataset import load_data
from perceptual_rank import bank,transform,metrics
from wave_folds import paired_labels
from listwise_retrieval import build_votes


def draw_edges(votes,people,gender,budget,rng):
    edges=[(i,j) for k,i in enumerate(people) for j in people[k+1:] if gender[i]==gender[j]]
    if budget is None:counts=np.full(len(edges),10)
    else:
        if budget>len(edges)*10 or budget<len(edges):raise ValueError('Invalid matched allocation')
        counts=np.full(len(edges),budget//len(edges))
        counts[rng.permutation(len(edges))[:budget%len(edges)]]+=1
    mat=np.full((len(gender),len(gender)),np.nan)
    for (i,j),n in zip(edges,counts):
        raw=votes[i,j]
        y=rng.choice(raw,n,replace=False).mean()
        mat[i,j]=mat[j,i]=y
    aa=[];bb=[];cc=[]
    for a in people:
        b,c=np.where((mat[a,:,None]-mat[a,None,:])>=1.)
        aa.extend([a]*len(b));bb.extend(b);cc.extend(c)
    return torch.tensor(np.stack((aa,bb,cc)),device='cuda'),int(sum(counts)),len(edges)


def trial(t,triples,va,te,labels,seed):
    torch.manual_seed(seed);w=torch.nn.Parameter(torch.eye(16,64,device='cuda'));init=w.detach().clone()
    opt=torch.optim.AdamW([w],lr=.002);best=-np.inf
    for step in range(601):
        if step%100==0:
            with torch.no_grad():
                z=F.normalize(F.linear(t[va,4:8],w),dim=-1).cpu().numpy()
                score=metrics(z,labels[0])['single_srcc']
                if score>best:best=score;selected=step;state=w.detach().clone()
        if step==600:break
        aa,bb,cc=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
        j=torch.randint(4,(3,len(aa)),device='cuda')
        z=F.normalize(F.linear(t[:,:4],w),dim=-1)
        delta=(z[aa,j[0]]*(z[bb,j[1]]-z[cc,j[2]])).sum(-1)
        loss=F.softplus(-6*delta).mean()+.02*(w-init).square().sum()/16
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():z=F.normalize(F.linear(t[te,8:12],state),dim=-1).cpu().numpy()
    return dict(selected_step=selected,outer=metrics(z,labels[1]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,required=True);a=ap.parse_args()
    data=load_data();names,blocks=bank(data);votes=build_votes(data,names)
    gender=np.array([data['manifest']['gender'][s] for s in names])
    rng=np.random.default_rng(9000+a.repeat)
    buckets=[rng.permutation(np.flatnonzero(gender==g)) for g in ('male','female')]
    folds=[np.concatenate([b[f::5] for b in buckets]) for f in range(5)]
    folder=ROOT/'runs'/f'annotation_budget_{a.repeat}';folder.mkdir(exist_ok=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        vectors=transform(blocks[:3],tr)
        x=np.concatenate(vectors,axis=-1)/np.sqrt(3)
        flat=x[tr,:4].reshape(-1,x.shape[-1]);mu=flat.mean(0)
        _,_,axes=np.linalg.svd((flat-mu).astype('float64'),full_matrices=False)
        t=torch.tensor((x-mu)@axes[:64].T,device='cuda',dtype=torch.float32)
        labels=(paired_labels(data,names,va),paired_labels(data,names,te))
        with torch.no_grad():plain=F.normalize(t[te,8:12,:16],dim=-1).cpu().numpy()
        report={'unlabeled':metrics(plain,labels[1]),'trials':[]}
        for sub in range(4):
            seed=70000+a.repeat*500+fold*20+sub
            rng=np.random.default_rng(seed)
            gen=[rng.permutation(tr[gender[tr]==g]) for g in ('male','female')]
            for count,budget in ((11,1000),(17,1000),(22,1000),(22,None)):
                people=np.concatenate([g[:count] for g in gen])
                if len(people)!=2*count:raise ValueError('Insufficient voices')
                triples,used,edges=draw_edges(votes,people,gender,budget,np.random.default_rng(seed+count))
                res=trial(t,triples,va,te,labels,seed)
                report['trials'].append(dict(subset=sub,people=2*count,ratings=used,pairs=edges,**res))
        write_json(dest,report);print(a.repeat,fold,len(report['trials']),flush=True)


if __name__=='__main__':main()
