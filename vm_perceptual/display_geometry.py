"""Out-of-sample display projection. Retrieval always stays in the fixed high-dimensional map."""
import argparse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from dataset import load_data
from runtime import ROOT,write_json


class View(nn.Module):
    def __init__(self,dim,nonlinear):
        super().__init__()
        self.linear=nn.Linear(16,dim,bias=False)
        self.extra=(nn.Sequential(nn.Linear(16,48),nn.Tanh(),nn.Linear(48,dim,bias=False)) if nonlinear else None)
        if self.extra is not None:nn.init.zeros_(self.extra[-1].weight)

    def forward(self,x):
        y=self.linear(x)
        return y if self.extra is None else y+self.extra(x)


def scores(query,gal,display,reference):
    full=cdist(query,gal);low=cdist(display,reference)
    correct=np.argsort(full,axis=1)[:,:5];near=np.argsort(low,axis=1)[:,:5]
    rank=np.argsort(np.argsort(full,axis=1),axis=1)+1
    return dict(recall5=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(correct,near)])),
        top1=float(np.mean(correct[:,0]==near[:,0])),
        median_true_rank_of_display_first=float(np.median(rank[np.arange(len(near)),near[:,0]])),
        global_srcc=float(spearmanr(full.ravel(),low.ravel()).statistic))


def fit_one(tr,va,te,dim,variant,seed=182):
    mu=tr.mean(0);_,_,axes=np.linalg.svd(tr-mu,full_matrices=False)
    scale=np.maximum(((tr-mu)@axes[:dim].T).std(),1e-4)
    q=lambda arr:torch.tensor(arr-mu,device='cuda',dtype=torch.float32)
    train=q(tr);dev=q(va);test=q(te)
    torch.manual_seed(seed)
    net=View(dim,variant=='mlp').cuda()
    with torch.no_grad():net.linear.weight.copy_(torch.tensor(axes[:dim]/scale))
    if variant!='pca':
        opt=torch.optim.AdamW(net.parameters(),lr=.004,weight_decay=.002)
        dist=torch.cdist(train,train).square().clamp_min(0)
        diag=torch.eye(len(tr),dtype=torch.bool,device='cuda')
        bandwidth=dist.masked_fill(diag,float('inf')).kthvalue(10,dim=1).values.clamp_min(1e-5)
        soft=(-dist/bandwidth[:,None]*2.5).masked_fill(diag,-1e6).softmax(1)
        for step in range(500):
            y=net(train);yd=torch.cdist(y,y).square()
            logp=(-torch.log1p(yd)).masked_fill(diag,-1e6).log_softmax(1)
            loss=-(soft*logp).sum(1).mean()
            opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():values=[net(x).cpu().numpy() for x in (train,dev,test)]
    return dict(dev=scores(va,tr,values[1],values[0]),test=scores(te,tr,values[2],values[0])),dict(
        mean=mu.tolist(),model={k:v.detach().cpu().tolist() for k,v in net.state_dict().items()},dim=dim,kind=variant)


def main():
    data=load_data();split=data['manifest']['split']
    keys=('current','own','combined')
    result={};out=ROOT/'runs/display_geometry';out.mkdir(exist_ok=True)
    for key in keys:
        arrays={}
        for group in ('train','dev','test'):
            z=[]
            for s in data['speakers']:
                if split[s]!=group:continue
                if key=='current':
                    with np.load(ROOT/'features/gallery_vectors'/f'{s}.npz') as f:vec=f['current']
                else:vec=np.load(ROOT/'features'/f'rank_map_{key}'/f'{s}.npy')
                # Fixed gallery size: four recordings per training voice; disjoint texts on development.
                z.extend(vec[4:8] if group=='dev' else vec[:4])
            arrays[group]=np.array(z)
        per={}
        for dim in (1,2,3,8):
            variants={};artifacts={}
            for method in ('pca','linear','mlp'):
                variants[method],artifacts[method]=fit_one(arrays['train'],arrays['dev'],arrays['test'],dim,method)
            # Selection uses only development recall; reject a large loss of global geometric fidelity.
            lower=variants['pca']['dev']['global_srcc']-.05
            eligible=[m for m in variants if variants[m]['dev']['global_srcc']>=lower]
            selected=max(eligible,key=lambda m:variants[m]['dev']['recall5'])
            per[str(dim)]=dict(choice=selected,candidates=variants)
            write_json(out/f'{key}_{dim}d.json',artifacts[selected])
        result[key]=per
    write_json(ROOT/'results/display_geometry.json',result)
    print({k:[(d,v['choice'],round(v['candidates'][v['choice']]['test']['recall5'],3)) for d,v in x.items()] for k,x in result.items()})


if __name__=='__main__':main()
