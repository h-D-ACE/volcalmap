"""Fit a Euclidean timbre map to discrete listening scores using ordered logistic probabilities."""
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dataset import load_data
from perceptual_rank import bank,transform,metrics
from runtime import ROOT,write_json
from wave_folds import paired_labels


class RatingMap(nn.Module):
    def __init__(self,learn):
        super().__init__()
        self.weight=nn.Parameter(torch.eye(16,64),requires_grad=learn)
        self.first=nn.Parameter(torch.tensor(-2.))
        self.gaps=nn.Parameter(torch.ones(5)*.1)
        self.log_slope=nn.Parameter(torch.tensor(1.))

    def forward(self,x):
        return F.normalize(F.linear(x,self.weight),dim=-1)

    def distribution(self,sim):
        cuts=torch.cat((self.first[None],self.first+torch.cumsum(F.softplus(self.gaps)+.001,dim=0)))
        cdf=torch.sigmoid(cuts[None,:]-F.softplus(self.log_slope)*sim[:,None])
        return torch.cat((cdf[:,:1],cdf[:,1:]-cdf[:,:-1],1-cdf[:,-1:]),dim=1).clamp_min(1e-8)


def observations(names,ids,data):
    idx={names[k]:i for i,k in enumerate(ids)}
    rows=[r for r in data['pairs'] if r['speaker_a'] in idx and r['speaker_b'] in idx]
    a=torch.tensor([idx[r['speaker_a']] for r in rows],device='cuda')
    b=torch.tensor([idx[r['speaker_b']] for r in rows],device='cuda')
    ratings=np.stack([np.bincount(np.array(r['ratings'],int)+3,minlength=7) for r in rows])
    counts=torch.tensor(ratings,device='cuda',dtype=torch.float32)
    return a,b,counts


def distribution_metrics(p,hist):
    h=hist/hist.sum(1,keepdim=True)
    return dict(nll=float(-(h*p.log()).sum(1).mean()),brier=float((p.square()-2*p*h+h).sum(1).mean()),
        count=int(hist.sum()),pairs=len(hist))


def fit(x,data,names,tr,va,te,learn):
    flat=x[tr,:4].reshape(-1,x.shape[-1]);mean=flat.mean(0)
    _,_,vt=np.linalg.svd((flat-mean).astype('float64'),full_matrices=False)
    t=torch.tensor((x-mean)@vt[:64].T,dtype=torch.float32,device='cuda')
    model=RatingMap(learn).cuda();initial=model.weight.detach().clone()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.009,weight_decay=0.)
    aa,bb,hist=observations(names,tr,data)
    a,b,hv=observations(names,va,data)
    at,bt,ht=observations(names,te,data)
    prior=(hist.sum(0)+.5)/(hist.sum()+3.5)
    def predict(ids,start,first,second):
        z=model(t[ids,start:start+4])
        mean=z.mean(1);sim=(mean[first]*mean[second]).sum(-1)
        return z,model.distribution(sim)
    best=np.inf
    for step in range(401):
        _,p=predict(tr,0,aa,bb)
        loss=-(hist*p.log()).sum()/hist.sum()+.004*(model.weight-initial).square().sum()/16
        if step%20==0:
            with torch.no_grad():
                _,vp=predict(va,4,a,b);score=distribution_metrics(vp,hv)['nll']
                if score<best:
                    best=score;chosen=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if step==400:break
        opt.zero_grad();loss.backward();opt.step()
    model.load_state_dict(state)
    with torch.no_grad():
        z,tp=predict(te,8,at,bt);quality=metrics(z.cpu().numpy(),paired_labels(data,names,te))
        return dict(selected_step=chosen,development_nll=best,outer=distribution_metrics(tp,ht),single=quality,
            train_prior=distribution_metrics(prior[None].expand(len(ht),-1),ht))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,default=0)
    a=ap.parse_args();data=load_data();names,blocks=bank(data)
    rng=np.random.default_rng(9000+a.repeat)
    buckets=[rng.permutation([i for i,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([g[i::5] for g in buckets]) for i in range(5)]
    folder=ROOT/'runs'/f'ordinal_geometry_{a.repeat}';folder.mkdir(exist_ok=True,parents=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        v=transform(blocks,tr);report={}
        for name,indices in [('own',(0,1,2)),('combined',(0,1,2,3))]:
            x=np.concatenate([v[i] for i in indices],axis=-1)/np.sqrt(len(indices))
            for learn in (False,True):
                report[name+('_learned' if learn else '_fixed')]=fit(x,data,names,tr,va,te,learn)
        write_json(dest,report);print(a.repeat,fold,report['combined_learned']['outer'],flush=True)


if __name__=='__main__':main()
