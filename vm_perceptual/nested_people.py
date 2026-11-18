"""Repeated nested speaker splits for profile versus cross-text model selection."""
import argparse,json,time
from pathlib import Path
from runtime import ROOT,write_json
from dataset import load_data,unit
from train_map import MetricHead
from scipy.stats import rankdata
import numpy as np
import torch
from torch.nn import functional as F


def rank_mean(target,values,groups):
    v=np.atleast_2d(values)
    by=[]
    for g in groups:
        a=rankdata(target[g]);a-=a.mean()
        b=rankdata(v[:,g],axis=1);b-=b.mean(1,keepdims=True)
        denom=np.linalg.norm(a)*np.linalg.norm(b,axis=1)
        by.append(float(np.mean(np.divide(b@a,denom,out=np.zeros(len(v)),where=denom>0))))
    return float(np.mean(by))


def pair_rows(allrows,lookup,people):
    names={s for s,i in lookup.items() if i in people}
    rows=[r for r in allrows if r['speaker_a'] in names and r['speaker_b'] in names]
    return dict(a=np.array([lookup[r['speaker_a']] for r in rows]),b=np.array([lookup[r['speaker_b']] for r in rows]),
                y=np.array([r['score'] for r in rows]),w=np.array([1/(1+r['variance']/r['n']) for r in rows]),
                groups=[np.array([r['gender']==g for r in rows]) for g in ('male','female')])


def summary(z,clips,pairs):
    a,b=pairs['a'],pairs['b']
    prof=-np.square(z[a]-z[b]).sum(-1)
    # Average both text orientations. All models face identical individual clip pairs.
    sim=np.stack([-np.square(clips[a,i]-clips[b,j]).sum(-1) for i in range(4) for j in range(4) if i!=j])
    return dict(profile=rank_mean(pairs['y'],prof,pairs['groups']),single=rank_mean(pairs['y'],sim,pairs['groups']))


def build_rep(bank,tr,keys):
    blocks=[]
    for k in keys:
        x=bank[k] if k=='acoustic' else unit(bank[k])
        train=x[tr,:4].reshape(-1,x.shape[-1]);mu=train.mean(0)
        if k=='acoustic':scale=np.maximum(train.std(0),.05)*np.sqrt(x.shape[-1])
        else:scale=max(np.sqrt(np.mean(np.square(train-mu).sum(-1))),1e-10)
        blocks.append((x-mu)/scale)
    clips=np.concatenate(blocks,axis=-1)/np.sqrt(len(keys))
    prof=clips[:,:4].mean(1)
    center=prof[tr].mean(0)
    _,_,vt=np.linalg.svd(prof[tr]-center,full_matrices=False)
    return prof,clips,(prof-center)@vt[:32].T,(clips-center)@vt[:32].T


def one_run(cache,data,seed,fold):
    bank=dict(acoustic=cache['acoustic'],ecapa=cache['ecapa'],wavlm=cache['layers'][:,:,3])
    names=cache['speakers'].tolist();lookup={s:i for i,s in enumerate(names)}
    rng=np.random.default_rng(seed)
    halves=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    parts=[np.concatenate([h[f::5] for h in halves]) for f in range(5)]
    te=parts[fold];va=parts[(fold+1)%5];tr=np.concatenate([parts[(fold+j)%5] for j in (2,3,4)])
    rp,rv,rt=[pair_rows(data['pairs'],lookup,x) for x in (tr,va,te)]
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    yt=tensor(rp['y']);wt=tensor(rp['w'])
    candidates={}
    for rep,keys in [('acoustic',['acoustic']),('ecapa',['ecapa']),('wavlm',['wavlm']),('fusion',['acoustic','ecapa'])]:
        profiles,clips,p,c=build_rep(bank,tr,keys)
        def evaluate(z,cs):
            return dict(dev=summary(z,cs[:,4:8],rv),outer=summary(z,cs[:,8:12],rt))
        candidates[rep+'_full']=evaluate(unit(profiles),unit(clips))
        candidates[rep+'_pca16']=evaluate(unit(p[:,:16]),unit(c[:,:,:16]))
        p,c=tensor(p),tensor(c)
        model=MetricHead(p.shape[-1],16).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        for step in range(300):
            opt.zero_grad();z=model(p);zc=model(c[tr,:4])
            score=model.scores(z,rp['a'],rp['b'])
            loss=(F.huber_loss(score,yt,reduction='none')*wt).mean()+.01*(zc-z[tr,None]).square().sum(-1).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step+1 in (25,75,150,300):
                with torch.no_grad():zz=model(p).cpu().numpy();cc=model(c).cpu().numpy()
                candidates[f'{rep}_trained{step+1}']=evaluate(zz,cc)
    policies={}
    for endpoint in ['profile','single']:
        for family in ['static','trained','all']:
            options=[k for k in candidates if family=='all' or (('trained' in k)==(family=='trained'))]
            best=max(options,key=lambda k:candidates[k]['dev'][endpoint])
            policies[family+'_'+endpoint]=dict(name=best,dev=candidates[best]['dev'][endpoint],outer=candidates[best]['outer'])
    return dict(seed=seed,fold=fold,n=[len(tr),len(va),len(te)],pair_counts=[len(x['y']) for x in (rp,rv,rt)],policies=policies,candidates=candidates)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    cache=dict(np.load(ROOT/'features/development_layers.npz'));data=load_data()
    out=ROOT/'results/nested_people';out.mkdir(exist_ok=True,parents=True)
    report=[]
    for j in range(a.repeats):
        seed=17401+541*j
        for f in range(5):
            file=out/f'{seed}_{f}.json'
            if file.exists():r=json.loads(file.read_text())
            else:
                r=one_run(cache,data,seed,f);write_json(file,r)
            report.append(r)
            print(seed,f,r['policies']['all_single'],flush=True)
    policies={k:dict(single_mean=float(np.mean([r['policies'][k]['outer']['single'] for r in report])),
                     profile_mean=float(np.mean([r['policies'][k]['outer']['profile'] for r in report])),
                     single_range=np.quantile([r['policies'][k]['outer']['single'] for r in report],[0,.5,1]).tolist()) for k in report[0]['policies']}
    write_json(ROOT/'results/nested_people_summary.json',dict(repeated_splits=len(report),people=80,policies=policies,scope='Repeated speakers across folds; dispersion is descriptive; original test people excluded.'))
    print(policies)


if __name__=='__main__':
    main()
