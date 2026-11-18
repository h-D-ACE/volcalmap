"""Fixed reference gallery, disjoint query voices and graded listening judgments."""
import argparse
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import rankdata
from dataset import load_data,unit
from runtime import ROOT,write_json
from extract_features import Frontend
from infer import encode
from wave_student import WaveMap


@torch.inference_mode()
def cache_vectors(data):
    out=ROOT/'features/gallery_vectors';out.mkdir(exist_ok=True)
    front=Frontend(device='cuda',use_ecapa=True,use_wavlm=False)
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    net=WaveMap().cuda().eval();net.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    archive={}
    for name in data['speakers']:
        dest=out/f'{name}.npz'
        if not dest.exists():
            arrays={k:[] for k in ('current','student','identity','voxsim')}
            files=sorted((ROOT/'data_jvs'/name/'parallel100/wav24kHz16bit').glob('*.wav'))
            with np.load(ROOT/'features/voxsim_large'/f'{name}.npz') as f:
                reference=dict(zip(f['names'].tolist(),f['vectors']))
            for file in files:
                path=file.relative_to(ROOT/'data_jvs').as_posix()
                feat=front.extract(file)
                arrays['current'].append(encode(feat,ck));arrays['identity'].append(unit(np.asarray(feat['ecapa'])))
                arrays['voxsim'].append(reference[path])
                x,sr=sf.read(file,dtype='float32');g=math.gcd(sr,16000)
                x=torch.from_numpy(resample_poly(x,16000//g,sr//g).astype('float32')).cuda()[None]
                arrays['student'].append(net(x,torch.tensor([x.shape[1]],device='cuda')).cpu().numpy()[0])
            stage=dest.with_suffix('.partial.npz')
            np.savez_compressed(stage,files=[p.name for p in files],**arrays);stage.replace(dest)
        with np.load(dest) as f:archive[name]={k:f[k] for k in f.files}
    return archive


def quality(score,y):
    order=np.argsort(-score,kind='stable');idx=order[0];top=y[idx]
    ideal=np.sort(y)[::-1];weights=1/np.log2(np.arange(2,5))
    best3=y>=ideal[2]
    return dict(hit3=float(best3[idx]),ndcg3=float((y[order[:3]]+3)@weights/((ideal[:3]+3)@weights)),
        regret=float(ideal[0]-top),rank=float(1+(y>top).sum()),chance=float(best3.mean()))


def main():
    data=load_data();cache=cache_vectors(data)
    split=data['manifest']['split'];gender=data['manifest']['gender']
    names=data['speakers'];rng=np.random.default_rng(30490)
    gal={g:sorted(rng.choice([s for s in names if split[s]=='train' and gender[s]==g],10,replace=False).tolist()) for g in ('female','male')}
    keys=('current','student','identity','voxsim')
    label={}
    for r in data['pairs']:
        label[(r['speaker_a'],r['speaker_b'])]=label[(r['speaker_b'],r['speaker_a'])]=r['score']
    people={};design=[];targets=[];subjects=[];groups=[]
    for s in names:
        gallery=gal[gender[s]]
        if s in gallery:continue
        # All groups use query sentences 1-4; gallery sentences 5-8 differ from them.
        scores={k:cache[s][k][:4]@np.stack([cache[n][k][4:8].mean(0) for n in gallery]).T for k in keys}
        y=np.array([label[(s,n)] for n in gallery])
        person={}
        for k,M in scores.items():
            stats=[quality(row,y) for row in M]
            person[k]={m:float(np.mean([d[m] for d in stats])) for m in stats[0]}
        for j in range(4):
            score=scores['current'][j];order=np.argsort(-score);first=order[0]
            # Inputs depend only on this query's audio, frozen gallery and model scores.
            x=[score[first],score[first]-score[order[1]],score.std()]
            for k in ('student','identity','voxsim'):
                row=scores[k][j]
                x.extend([float(np.mean(row<=row[first])),float(np.corrcoef(rankdata(score),rankdata(row))[0,1])])
            design.append(x);targets.append(quality(score,y)['hit3']);subjects.append(s);groups.append(split[s])
        people[s]=person
    x=np.asarray(design);y=np.asarray(targets);groups=np.asarray(groups);tr=groups=='train'
    mu=x[tr].mean(0);sd=np.maximum(x[tr].std(0),.05)
    x=np.c_[np.ones(len(x)),(x-mu)/sd]
    prior=(y[tr].sum()+1)/(tr.sum()+2)
    def objective(w):
        z=x[tr]@w
        return np.sum(np.logaddexp(0,z)-y[tr]*z)+5*w[1:]@w[1:]
    sol=minimize(objective,np.r_[np.log(prior/(1-prior)),np.zeros(x.shape[1]-1)],method='BFGS',tol=1e-5)
    prob=expit(x@sol.x);report={}
    for group in ('train','dev','test'):
        m=groups==group;sp=[s for s in people if split[s]==group]
        mean={}
        for k in keys:
            mean[k]={key:float(np.mean([people[s][k][key] for s in sp])) for key in people[sp[0]][k]}
        ci={}
        bootstrap=np.random.default_rng(30491)
        strata=[np.flatnonzero([gender[s]==g for s in sp]) for g in ('female','male')]
        draws=np.concatenate([bootstrap.choice(ix,(2000,len(ix))) for ix in strata],axis=1)
        for k in keys:
            delta=np.array([people[s][k]['hit3']-people[s]['current']['hit3'] for s in sp])
            ci[k]=np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist()
        report[group]=dict(people=len(sp),queries=int(m.sum()),models=mean,hit3_delta_95=ci,
            calibrated_probability_brier=float(np.mean((prob[m]-y[m])**2)),constant_brier=float(np.mean((prior-y[m])**2)),
            constant_skill_rate=float(prior),predicted_hit_rate=float(prob[m].mean()),observed_hit_rate=float(y[m].mean()))
    write_json(ROOT/'results/gallery_benchmark.json',dict(gallery=gal,groups=report,model='acoustic_d16 for calibration',
        note='Conditional on 10 same-sex gallery voices. Listener mean ties count equally. No arbitrary-voice confidence.'))
    print({g:{k:round(v['hit3'],3) for k,v in r['models'].items()} for g,r in report.items()},flush=True)


if __name__=='__main__':main()
