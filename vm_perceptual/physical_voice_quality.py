"""Test whether explicit phonation measurements add cross-text listening information."""
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import parselmouth as pm
from parselmouth.praat import call
from scipy.stats import rankdata
import torch
from dataset import load_rating_graph
from nested_ssl_trial import make_teacher
from perceptual_rank import bank, transform, train_triples
from runtime import ROOT, write_json
from wave_folds import paired_labels

FIELDS=('cpps_db','median_hnr_db','median_log_f0','median_voiced_f3_hz')
WEIGHTS=(0.,.05,.15,.3,.5)


def measure(path):
    sound=pm.Sound(str(path))
    cep=call(sound,'To PowerCepstrogram',60,.01,5000,50)
    cpp=call(cep,'Get CPPS',False,.02,.0005,60,500,.05,'Parabolic',.001,.05,'Straight','Robust')
    pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=60,pitch_ceiling=800)
    f=pitch.selected_array['frequency'];voiced=f>0
    h=sound.to_harmonicity_cc(time_step=.01,minimum_pitch=60).values.ravel()
    h=h[np.isfinite(h)&(h>-100)]
    formants=sound.to_formant_burg(time_step=.01,maximum_formant=5500)
    f3=np.array([formants.get_value_at_time(3,t) for t in pitch.xs()[voiced]])
    def median(a):
        a=np.asarray(a);a=a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else np.nan
    return [cpp,median(h),median(np.log(f[voiced])),median(f3)]


def distances(z,people,slots,labels):
    z=z[people]
    return [np.stack([np.mean((z[a,i]-z[b,j])**2,axis=-1)
                      for i,j in itertools.permutations(slots,2)]) for a,b,_ in labels]


def score(a,b,w,labels):
    scores=[]
    for da,db,(_,_,y) in zip(a,b,labels):
        pred=rankdata(-((1-w)*da+w*db),axis=1);pred-=pred.mean(1,keepdims=True)
        target=rankdata(y);target-=target.mean()
        scores.append(float(np.mean(pred@target/np.maximum(np.linalg.norm(pred,axis=1)*np.linalg.norm(target),1e-12))))
    return float(np.mean(scores))


def main():
    data=load_rating_graph(('train','dev'));names,blocks=bank(data,False)
    paths=[p for s in names for p in sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))]
    if len(paths)!=len(names)*12:raise ValueError('Expected 12 aligned normal texts per development speaker')
    cache=ROOT/'features/physical_quality_dev.npz'
    ids=[str(p.relative_to(ROOT/'data_jvs')) for p in paths]
    if cache.exists():
        with np.load(cache) as f:
            if f['files'].tolist()!=ids:raise ValueError('Feature cache has different recordings')
            raw=f['features']
    else:
        with ProcessPoolExecutor(max_workers=4) as pool:raw=np.array(list(pool.map(measure,paths,chunksize=4)))
        raw=raw.reshape(len(names),12,-1)
        temp=cache.with_suffix('.partial.npz');np.savez_compressed(temp,files=ids,features=raw);temp.replace(cache)
    rows=[]
    for repeat in range(5):
        rng=np.random.default_rng(9000+repeat)
        groups=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
        folds=[np.concatenate([g[j::5] for g in groups]) for j in range(5)]
        for k in range(5):
            te,va=folds[k],folds[(k+1)%5]
            tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
            lab=[paired_labels(data,names,x) for x in (va,te)]
            train=raw[tr,:4].reshape(-1,len(FIELDS))
            med=np.nanmedian(train,axis=0);scale=np.maximum(np.nanquantile(train,.75,axis=0)-np.nanquantile(train,.25,axis=0),1e-5)
            q=np.clip((np.where(np.isfinite(raw),raw,med)-med)/scale,-5,5)
            baseline=np.concatenate(transform(blocks,tr),axis=-1)/np.sqrt(3)
            teacher,_=make_teacher(baseline,tr,va,lab[0],train_triples(tr,names,data),881+k)
            # Scale squared teacher unit-vector distances by dimension to use sum distance.
            teacher=teacher*np.sqrt(teacher.shape[-1])
            vd=distances(teacher,va,[4,5],lab[0]);td=distances(teacher,te,[8,9,10,11],lab[1])
            out={}
            for tag,block in [('cpps',q[:,:,:1]),('four_measures',q)]:
                v=distances(block,va,[4,5],lab[0]);t=distances(block,te,[8,9,10,11],lab[1])
                best=max(WEIGHTS,key=lambda w:score(vd,v,w,lab[0]))
                out[tag]=dict(weight=best,outer=score(td,t,best,lab[1]),physical_only=score(td,t,1.,lab[1]))
            rows.append(dict(repeat=repeat,fold=k,teacher=score(td,td,0,lab[1]),variants=out))
    summary={}
    for name in ('cpps','four_measures'):
        summary[name]=dict(teacher=float(np.mean([r['teacher'] for r in rows])),
            blend=float(np.mean([r['variants'][name]['outer'] for r in rows])),
            physical_only=float(np.mean([r['variants'][name]['physical_only'] for r in rows])),
            positive_weight=sum(r['variants'][name]['weight']>0 for r in rows))
    write_json(ROOT/'results/physical_voice_quality.json',dict(summary=summary,folds=rows,fields=FIELDS,clips=len(paths),
        missing=np.sum(~np.isfinite(raw),axis=(0,1)).tolist(),scope='80 development speakers only. Shared people across 25 folds. Fixed 5500 Hz F3 ceiling.'))
    print(summary,flush=True)


if __name__=='__main__':main()
