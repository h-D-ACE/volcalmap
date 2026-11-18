"""Learn a residual correction to frozen map coordinates from paired signal corruptions."""
import json
from runtime import ROOT,write_json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from infer import encode


class Correction(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(44,width),nn.SiLU(),nn.Linear(width,16))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,z):
        return F.normalize(z+self.net(x),dim=-1)


def main():
    torch.set_num_threads(2)
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    if ck['kind']!='trained' or ck['output_dim']!=16:raise ValueError('Requires frozen 16-D acoustic map')
    records=[]
    for p in sorted((ROOT/'features/perturbations').glob('*.npz')):
        d=np.load(p)
        for name,views in zip(d['names'],d['features']):
            if not np.isfinite(views[0]).all():continue
            for mode,feat in zip(d['modes'],views):
                if np.isfinite(feat).all():records.append((str(d['speaker']),str(d['split']),str(name),str(mode),feat,views[0]))
    if len(set(r[0] for r in records))!=100:raise ValueError('Expected completed feature collection')
    def base(array):
        # Reuse the trained feature preprocessing and head for a single batch.
        conf=ck['preprocessing'];block=conf['blocks'][0]
        z=(array[:,:40]-block['mean'])/block['scale']/block['divisor']
        z=(z-conf['pca_mean'])@np.array(conf['pca_axes']).T
        with torch.no_grad():
            w=ck['model']['projection.weight']
            return F.normalize(torch.tensor(z,dtype=torch.float32)@w.T,dim=-1).numpy()
    sp,split,names,mode=[np.array([r[i] for r in records]) for i in range(4)]
    features=np.array([r[4] for r in records])
    z=base(features);target=base(np.array([r[5] for r in records]))
    known=['noise_10','noise_30','band_8k','band_12k','tilt']
    train=(split=='train') & np.isin(mode,['clean']+known)
    mu=features[train].mean(0);scale=np.maximum(features[train].std(0),.001)
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    x=tensor(np.clip((features-mu)/scale,-10,10));zin=tensor(z);y=tensor(target)
    def summarize(pred,group):
        stats={}
        # Equal speaker weighting via medians before averaging.
        err=np.linalg.norm(pred-target,axis=-1);base_err=np.linalg.norm(z-target,axis=-1)
        for condition in sorted(set(mode)):
            mask=(split==group)&(mode==condition)
            people=sorted(set(sp[mask]))
            current=np.array([np.median(err[mask&(sp==s)]) for s in people])
            original=np.array([np.median(base_err[mask&(sp==s)]) for s in people])
            stats[condition]=dict(n=int(mask.sum()),voices=len(people),error=float(current.mean()),before=float(original.mean()),by_voice_delta=(current-original).tolist())
        return stats
    baseline=summarize(z,'dev')
    def utility(stats):
        return np.mean([stats[k]['error'] for k in known]) + 3*stats['clean']['error']
    best=utility(baseline);best_state=None;choice='unchanged';candidates={choice:float(best)}
    for clean_weight in (4.,16.,64.):
        torch.manual_seed(319)
        model=Correction(64).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        clean=tensor(np.where(mode[train]=='clean',clean_weight,1.))
        for step in range(300):
            optimizer.zero_grad();p=model(x[train],zin[train])
            loss=((p-y[train]).square().sum(1)*clean).mean();loss.backward();optimizer.step()
            if step+1 in (50,150,300):
                with torch.no_grad():pred=model(x,zin).cpu().numpy()
                stats=summarize(pred,'dev');v=utility(stats);name=f'w{clean_weight:g}_s{step+1}'
                candidates[name]=dict(utility=float(v),clean=stats['clean']['error'])
                if stats['clean']['error']<=.075 and v<best:
                    best=v;choice=name
                    best_state={k:w.detach().cpu().clone() for k,w in model.state_dict().items()}
    if best_state is None:pred=z
    else:
        model.load_state_dict(best_state)
        with torch.no_grad():pred=model(x,zin).cpu().numpy()
        torch.save(dict(model=best_state,mean=mu.tolist(),scale=scale.tolist(),width=64,base_hash=ck['data_hash'],choice=choice),ROOT/'runs/noise_correction_dev.pt')
    result=dict(choice=choice,development=candidates,test=summarize(pred,'test'),valid_views=len(records),unseen_conditions=['noise_20','echo'],clean_gate=.075,scope='Same waveform under synthetic corruption; distances to original map position, not human ratings')
    write_json(ROOT/'results/paired_correction.json',result)
    print(choice,{k:round(r['error']-r['before'],4) for k,r in result['test'].items()})


if __name__=='__main__':
    main()
