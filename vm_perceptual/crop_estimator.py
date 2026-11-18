"""Estimate the full-clip position from a short observation; keep an unchanged candidate."""
import json
from runtime import ROOT,write_json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cdist


def main():
    torch.set_num_threads(2);device='cuda'
    data=[]
    for p in sorted((ROOT/'features/crop_records').glob('*.npz')):
        with np.load(p) as f:
            row={k:f[k] for k in ('features','targets','emb','prior','names','condition','split','failed')}
        row['speaker']=np.repeat(p.stem,len(row['names']))
        row['split']=np.repeat(str(row['split']),len(row['names']))
        data.append(row)
    if len(data)!=100:raise ValueError('Await all 100 speaker partitions')
    d={k:np.concatenate([r[k] for r in data]) for k in ('features','targets','emb','prior','names','condition','speaker','split')}
    tr=d['split']=='train';va=d['split']=='dev';te=d['split']=='test'
    dur=d['condition'][:,0];full=dur==0
    mu=d['features'][tr].mean(0);sd=np.maximum(d['features'][tr].std(0),.03)
    base=np.clip((d['features']-mu)/sd,-8,8)
    embmu=d['emb'][tr & full].mean(0)
    u=torch.from_numpy(d['emb'][tr & full]-embmu).to(device)
    _,_,vt=torch.linalg.svd(u,full_matrices=False)
    axes=vt[:48].cpu().numpy()
    xp=(d['emb']-embmu)@axes.T
    xscale=np.maximum(xp[tr].std(0),.01)
    x=np.concatenate([base,np.clip(xp/xscale,-8,8)],axis=1)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    xt=tensor(x);prior=tensor(d['prior']);target=tensor(d['targets'])
    def score(pred,group,include_retrieval=False):
        out={};mask=d['split']==group
        gallery=mask&full;gt=d['targets'][gallery]
        for sec in (0,1,2,4):
            m=mask&(dur==sec)
            row=dict(n=int(m.sum()),mean_l2=float(np.linalg.norm(pred[m]-d['targets'][m],axis=1).mean()),median_l2=float(np.median(np.linalg.norm(pred[m]-d['targets'][m],axis=1))))
            if include_retrieval:
                found=cdist(pred[m],gt).argmin(1)
                row['full_recording_top1']=float(np.mean((d['speaker'][gallery][found]==d['speaker'][m])&(d['names'][gallery][found]==d['names'][m])))
            out[str(sec)]=row
        return out
    loss_score=lambda st:np.mean([st[str(q)]['mean_l2'] for q in (1,2,4)])
    uncorrected=score(d['prior'],'dev');best=loss_score(uncorrected);state=None;choice='unchanged';dev_trials={}
    for width_dim in (45,93):
        for clean_weight in (3.,12.):
            torch.manual_seed(76)
            head=nn.Sequential(nn.Linear(width_dim,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,16)).to(device)
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
            optim=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
            weights=tensor(np.where(full[tr],clean_weight,1.))
            for step in range(600):
                correction=.75*torch.tanh(head(xt[tr,:width_dim]))
                z=F.normalize(prior[tr]+correction,dim=-1)
                loss=((z-target[tr]).square().sum(1)*weights).mean()
                optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),3);optim.step()
                if step+1 in (50,150,300,600):
                    with torch.no_grad():pred=F.normalize(prior[va]+.75*torch.tanh(head(xt[va,:width_dim])),dim=-1).cpu().numpy()
                    tmp=d['prior'].copy();tmp[va]=pred;st=score(tmp,'dev')
                    value=loss_score(st);name=f'd{width_dim}_w{clean_weight:g}_s{step+1}'
                    dev_trials[name]=dict(mean=float(value),full_shift=st['0']['mean_l2'])
                    if value<best and st['0']['mean_l2']<.05:
                        best=value;choice=name;state=(width_dim,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
    pred=d['prior'].copy()
    if state is not None:
        width_dim,parameters=state
        head=nn.Sequential(nn.Linear(width_dim,96),nn.LayerNorm(96),nn.SiLU(),nn.Linear(96,16)).to(device)
        head.load_state_dict(parameters)
        with torch.no_grad():pred=F.normalize(prior+.75*torch.tanh(head(xt[:,:width_dim])),dim=-1).cpu().numpy()
        torch.save(dict(parameters=parameters,width_dim=width_dim,mu=mu.tolist(),sd=sd.tolist(),embmu=embmu.tolist(),axes=axes.tolist(),xscale=xscale.tolist(),choice=choice),ROOT/'runs/crop_estimator.pt')
    result=dict(choice=choice,dev_trials=dev_trials,dev_raw=uncorrected,test_raw=score(d['prior'],'test',True),test_selected=score(pred,'test',True),front_failures=int(sum(r['failed'] for r in data)),counts={k:int(np.sum(d['split']==k)) for k in ('train','dev','test')})
    write_json(ROOT/'results/crop_estimator.json',result);print(choice,result['test_raw'],result['test_selected'],flush=True)


if __name__=='__main__':
    main()
