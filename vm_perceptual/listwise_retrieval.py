"""Learn 16-D coordinates against a distribution over alternative reference voices."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from dataset import load_data
from runtime import ROOT,write_json
from perceptual_rank import bank,transform,metrics
from gallery_benchmark import quality
from wave_folds import paired_labels


def build_votes(data,names):
    lookup={n:i for i,n in enumerate(names)}
    votes=np.full((len(names),len(names),10),np.nan,np.float32)
    for r in data['pairs']:
        if r['speaker_a'] in lookup and r['speaker_b'] in lookup:
            i,j=lookup[r['speaker_a']],lookup[r['speaker_b']]
            votes[i,j]=votes[j,i]=r['ratings']
    return votes


def gallery_scores(z,people,query_ix,galleries,names,data,votes):
    stats=[]
    for p in people:
        gal=galleries[data['manifest']['gender'][names[p]]]
        ref=z[gal,4:8].mean(1);score=z[p,query_ix]@ref.T
        y=votes[p,gal].mean(1)
        stats.extend([quality(row,y) for row in score])
    return {k:float(np.mean([v[k] for v in stats])) for k in stats[0]}


def fit(data,names,X,votes,tr,va,te,galleries,method,seed):
    flat=X[tr,:4].reshape(-1,X.shape[-1]);mu=flat.mean(0)
    _,_,vt=np.linalg.svd((flat-mu).astype('float64'),full_matrices=False)
    t=torch.tensor((X-mu)@vt[:64].T,dtype=torch.float32,device='cuda')
    w=torch.nn.Parameter(torch.eye(16,64,device='cuda'));initial=w.detach().clone()
    opt=torch.optim.AdamW([w],lr=.002,weight_decay=.01)
    rng=np.random.default_rng(seed)
    person={g:np.array([i for i in tr if data['manifest']['gender'][names[i]]==g]) for g in ('female','male')}
    best=-np.inf;history=[]
    for step in range(801):
        if step%100==0:
            with torch.no_grad():
                z=F.normalize(F.linear(t,w),dim=-1).cpu().numpy()
                # Gallery uses 5..8. Validation queries below use 9..12 instead for disjoint texts.
                rank=gallery_scores(z,va,slice(8,12),galleries,names,data,votes)
                utility=rank['ndcg3']
                if utility>best:
                    best=utility;chosen=step;state=w.detach().clone()
                history.append([step,utility])
        if step==800:break
        g=('female','male')[step%2]
        sample=rng.permutation(person[g]);gal=sample[:10];anchors=sample[10:]
        raw=votes[anchors[:,None],gal]
        if not np.isfinite(raw).all():raise ValueError('Missing within-sex training ratings')
        if method=='mean_softmax':
            target=raw.mean(-1)/.5;target=np.exp(target-target.max(-1,keepdims=True));target/=target.sum(-1,keepdims=True)
        else:
            # Random half of rating votes per pair simulates ranking uncertainty without extra labels.
            ids=rng.random(raw.shape).argsort(-1)[:,:,:5]
            draws=np.take_along_axis(raw,ids,axis=-1).mean(-1)
            top=draws>=np.sort(draws,axis=-1)[:,-3,None]
            target=top/top.sum(-1,keepdims=True)
        z=F.normalize(F.linear(t,w),dim=-1)
        q=z[anchors,rng.integers(4,size=len(anchors))];ref=z[gal,:4].mean(1)
        log_prob=(q@ref.T/.13).log_softmax(1)
        y=torch.as_tensor(target,dtype=torch.float32,device='cuda')
        loss=-(y*log_prob).sum(1).mean()+.02*(w-initial).square().sum()/16
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():z=F.normalize(F.linear(t,state),dim=-1).cpu().numpy()
    return dict(selected=chosen,history=history,outer_gallery=gallery_scores(z,te,slice(8,12),galleries,names,data,votes),
        outer_disjoint=metrics(z[te,8:12],paired_labels(data,names,te)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,default=0)
    a=ap.parse_args();data=load_data();names,blocks=bank(data);votes=build_votes(data,names)
    rng=np.random.default_rng(9000+a.repeat)
    people=[rng.permutation([i for i,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([p[i::5] for p in people]) for i in range(5)]
    folder=ROOT/'runs'/f'listwise_{a.repeat}';folder.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        va,te=folds[(fold+1)%5],folds[fold]
        tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        galleries={g:np.random.default_rng(8402+fold).choice([i for i in tr if data['manifest']['gender'][names[i]]==g],10,replace=False) for g in ('female','male')}
        parts=transform(blocks,tr);result={}
        for name,n in [('own',3),('combined',4)]:
            x=np.concatenate(parts[:n],axis=-1)/np.sqrt(n)
            result[name+'_fixed']=gallery_scores(x,te,slice(8,12),galleries,names,data,votes)
            for method in ('mean_softmax','resampled_top3'):
                result[name+'_'+method]=fit(data,names,x,votes,tr,va,te,galleries,method,8402+fold)
        write_json(dest,result);print(a.repeat,fold,result['own_mean_softmax']['outer_gallery'],flush=True)


if __name__=='__main__':main()
