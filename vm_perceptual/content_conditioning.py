"""Train-only spectral clustering to compare clips at similar acoustic content."""
import json
from pathlib import Path

from runtime import ROOT, write_json
from extract_features import Frontend, resample_poly
from train_map import correlations
from dataset import load_data, unit
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import soundfile as sf
import torch
from math import gcd


def frames(frontend, path):
    samples, rate = sf.read(path, dtype='float32')
    factor = gcd(rate, 16000)
    if rate != 16000:
        samples = resample_poly(samples,16000//factor,rate//factor)
    with torch.inference_mode():
        mfcc=frontend.mfcc(torch.tensor(samples,device='cuda')).T.cpu().numpy()
    energy=mfcc[:,0]
    keep=energy > np.percentile(energy,20)
    return mfcc[keep,1:20:1]


def descriptors(clips, train_ids, k):
    # Fit pattern vocabulary with equal upper bound on frames per training clip.
    training=np.concatenate([x[::max(1,len(x)//40)][:40] for who in train_ids for x in clips[who]])
    mu,sd=training.mean(0),np.maximum(training.std(0),.01)
    vocabulary=MiniBatchKMeans(k,random_state=304,n_init=3,batch_size=2048)
    vocabulary.fit((training-mu)/sd)
    vectors=[];global_vectors=[]
    for speaker in clips:
        embedded=[];global_embedded=[]
        for x in speaker:
            x=(x-mu)/sd
            global_embedded.append(np.r_[x.mean(0),x.std(0)])
            index=vocabulary.predict(x)
            means=[]
            for j,center in enumerate(vocabulary.cluster_centers_):
                samples=x[index==j]
                means.append((samples-center).sum(0)/(len(samples)+10))
            embedded.append(np.concatenate(means))
        vectors.append(embedded);global_vectors.append(global_embedded)
    def standardize(items):
        x=np.asarray(items);tr=x[train_ids].reshape(-1,x.shape[-1])
        return unit((x-tr.mean(0))/np.maximum(tr.std(0),.05))
    return standardize(global_vectors),standardize(vectors)


def evaluate(group, rows, arrays):
    idx={s:i for i,s in enumerate(group)}
    a,b=([idx[r[key]] for r in rows] for key in ('speaker_a','speaker_b'))
    # Two nonoverlapping four-utterance blocks, all cross-block single clip pairs.
    pairs=[(i,j) for i in range(4,8) for j in range(8,12)]
    by_candidate={}
    for name,z in arrays.items():
        scores=[]
        for i,j in pairs:
            dist=np.linalg.norm(z[a,i]-z[b,j],axis=1)
            scores.append(correlations(rows,-dist)['mean_spearman'])
        by_candidate[name]=float(np.mean(scores))
    return by_candidate


def main():
    data=load_data()
    speakers=[s for s in data['speakers'] if data['manifest']['split'][s] in ('train','dev')]
    front=Frontend(device='cuda',use_ecapa=False,use_wavlm=False)
    clips=[]
    for s in speakers:
        base=ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit'
        paths=sorted(base.glob('*.wav'))[:12]
        if len(paths)!=12:raise ValueError(f'{s}: expected twelve texts')
        clips.append([frames(front,p) for p in paths])
    train=[i for i,s in enumerate(speakers) if data['manifest']['split'][s]=='train']
    dev=[i for i,s in enumerate(speakers) if data['manifest']['split'][s]=='dev']
    rng=np.random.default_rng(47)
    gender=data['manifest']['gender']
    halves=[rng.permutation([i for i in train if gender[speakers[i]]==g]) for g in ('male','female')]
    def measure(fit,valid):
        valid_names={speakers[i] for i in valid}
        rows=[r for r in data['pairs'] if r['speaker_a'] in valid_names and r['speaker_b'] in valid_names]
        outputs={}
        for k in (4,12):
            glob,cond=descriptors(clips,fit,k)
            candidates={'global':glob,f'conditioned{k}':cond,f'combined{k}':np.concatenate([glob,cond],axis=-1)/np.sqrt(2)}
            outputs.update(evaluate(speakers,rows,candidates))
        return outputs
    folds=[]
    for j in range(3):
        val=np.concatenate([g[j::3] for g in halves])
        folds.append(measure(np.setdiff1d(train,val),val))
    choice=max(folds[0],key=lambda k:np.mean([f[k] for f in folds]))
    dev_scores=measure(train,dev)
    report=dict(train_folds=folds,train_selected=choice,selected_dev=dev_scores[choice],dev=dev_scores,
                scope='JVS voice ratings; acoustic clusters unsupervised; different-text single recordings, train-only fitting')
    write_json(ROOT/'results/content_conditioning.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
