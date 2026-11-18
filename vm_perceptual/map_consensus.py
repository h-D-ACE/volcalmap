"""Measure optimizer-induced map drift and fit an aligned ensemble, with training-only anchors."""
import numpy as np
import torch
from torch.nn import functional as F
from runtime import ROOT,write_json,file_hash
from dataset import load_data,unit
from rank_map_candidate import preprocess,Frontends
from perceptual_rank import bank,metrics,train_triples
from wave_folds import paired_labels
from reference_benchmark import benchmark
from gallery_benchmark import quality
import json


def orthogonal(source,target):
    u,_,vt=np.linalg.svd(source.T@target,full_matrices=False)
    return u@vt


def head(t,tr,va,labels,triples,seed):
    torch.manual_seed(seed);w=torch.nn.Parameter(torch.eye(16,64,device='cuda'));prior=w.detach().clone()
    opt=torch.optim.AdamW([w],lr=.002);best=-np.inf
    for step in range(1001):
        if step%100==0:
            with torch.no_grad():
                z=F.normalize(F.linear(t[va,4:8],w),dim=-1).cpu().numpy()
                val=metrics(z,labels)['single_srcc']
                if val>best:best=val;state=w.detach().cpu().numpy().copy();chosen=step
        if step==1000:break
        z=F.normalize(F.linear(t[tr,:4],w),dim=-1)
        a,b,c=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
        j=torch.randint(4,(3,len(a)),device='cuda')
        delta=(z[a,j[0]]*(z[b,j[1]]-z[c,j[2]])).sum(-1)
        loss=F.softplus(-6*delta).mean()+.02*(w-prior).square().sum()/16
        opt.zero_grad();loss.backward();opt.step()
    return state,chosen


def embed(x,weights,rotations):
    return unit(np.mean([unit(x@w.T)@r for w,r in zip(weights,rotations)],axis=0))


def main():
    data=load_data();names,blocks=bank(data)
    artifact=ROOT/'runs/rank_map_own.npz'
    with np.load(artifact) as f:params={k:f[k] for k in f.files}
    all_x=preprocess(blocks,params);t=torch.from_numpy(all_x).cuda()
    tr=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='train'])
    va=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='dev'])
    labels=paired_labels(data,names,va);triples=train_triples(tr,names,data)
    rng=np.random.default_rng(24576);anchors=rng.choice(tr,32,replace=False)
    ax=all_x[anchors,:4].reshape(-1,64);base=unit(ax@params['weight'].T)
    weights=[];rots=[];steps=[]
    for seed in (101,201,301,401,501,601,701,801,901,1001,1101,1201):
        w,step=head(t,tr,va,labels,triples,seed);steps.append(step)
        r=orthogonal(unit(ax@w.T),base);weights.append(w);rots.append(r)
    weights=np.stack(weights);rots=np.stack(rots)
    all_features={s:all_x[i] for i,s in enumerate(names)}
    front=Frontends(False)
    for s in data['speakers']:
        if data['manifest']['split'][s]!='test':continue
        files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))
        raw=[front.extract(p) for p in files]
        all_features[s]=preprocess([np.stack(v) for v in zip(*raw)],params)
    canonical={s:unit(x@params['weight'].T) for s,x in all_features.items()}
    ensemble={s:embed(x,weights,rots) for s,x in all_features.items()}
    acoustic={}
    for s in data['speakers']:
        with np.load(ROOT/'features/gallery_vectors'/f'{s}.npz') as f:acoustic[s]=f['current']
    report={};gallery=json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    ratings={}
    for p in data['pairs']:ratings[p['speaker_a'],p['speaker_b']]=ratings[p['speaker_b'],p['speaker_a']]=p['score']
    for key,group,start,end in [('dev','dev',4,8),('test','test',0,4),('fresh','test',4,12)]:
        ids=[s for s in data['speakers'] if data['manifest']['split'][s]==group and len(all_features[s])>=end]
        views={'current':acoustic,'canonical':canonical,'ensemble':ensemble}
        result=benchmark(ids,data,{n:{s:z[s][start:end] for s in ids} for n,z in views.items()},2285)
        for n,z in views.items():
            evaluated=[]
            for s in ids:
                gal=gallery[data['manifest']['gender'][s]]
                y=np.array([ratings[s,j] for j in gal]);g=np.stack([z[j][4:8].mean(0) for j in gal])
                q=z[s][8:12] if key=='fresh' else z[s][:4]
                evaluated.extend(quality(row,y) for row in q@g.T)
            result['models'][n]['gallery_hit3']=float(np.mean([v['hit3'] for v in evaluated]))
        # Compare seed-specific neighborhoods against exactly the same gallery of train recordings.
        gx=all_x[tr,:4].reshape(-1,64);qx=np.concatenate([all_features[s][start:end] for s in ids])
        cq=unit(qx@params['weight'].T);cg=unit(gx@params['weight'].T);ranking=np.argsort(-cq@cg.T,axis=1)
        shifts=[]
        for w,r in zip(weights,rots):
            z=unit(qx@w.T);g=unit(gx@w.T);rr=np.argsort(-z@g.T,axis=1)
            shifts.append(dict(raw_median_l2=float(np.median(np.linalg.norm(z-cq,axis=1))),
                aligned_median_l2=float(np.median(np.linalg.norm(z@r-cq,axis=1))),
                top1=float(np.mean(rr[:,0]==ranking[:,0])),
                top5_overlap=float(np.mean([len(set(i[:5])&set(j[:5]))/5 for i,j in zip(rr,ranking)]))))
        result['mean_across_seeds']={n:float(np.mean([v[n] for v in shifts])) for n in shifts[0]}
        report[key]=result
    train=embed(all_x[tr,:4],weights,rots).reshape(-1,16);mu=train.mean(0);_,_,vt=np.linalg.svd(train-mu,full_matrices=False)
    np.savez_compressed(ROOT/'runs/rank_map_consensus.npz',weights=weights,rotations=rots,base_sha256=file_hash(artifact),view_mean=mu,view_axes=vt)
    write_json(ROOT/'results/map_consensus.json',dict(steps=steps,anchors=len(anchors)*4,results=report))


if __name__=='__main__':main()
