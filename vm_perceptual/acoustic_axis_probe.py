"""Probe the learned map using held-out voices and texts in physical measurement units."""
import json
import numpy as np
from scipy.stats import spearmanr
from dataset import load_rating_graph
from nested_ssl_trial import make_teacher
from perceptual_rank import bank, transform, train_triples
from physical_voice_quality import FIELDS
from runtime import ROOT, write_json
from wave_folds import paired_labels


def probes(z, raw, tr, va, te, genders):
    results = {}
    for j, key in enumerate(FIELDS):
        target = raw[:,:,j].copy()
        if key == 'median_log_f0':
            target *= 12 / np.log(2)
        training = z[tr,:4].reshape(-1,z.shape[-1])
        y = target[tr,:4].ravel()
        mask=np.isfinite(y)
        mu=training[mask].mean(0)
        response=y[mask].mean()
        x=training[mask]-mu
        yc=y[mask]-response
        valid=target[va,4:6]
        def fit(lam):
            return np.linalg.solve(x.T@x + lam*np.eye(x.shape[1]),x.T@yc)
        def predict(people,sl,coef):
            return (z[people,sl]-mu)@coef + response
        lambdas=(.01,.1,1.,10.)
        chosen=min(lambdas,key=lambda v:np.nanmean(abs(predict(va,slice(4,6),fit(v))-valid)))
        pred=predict(te,slice(8,12),fit(chosen))
        truth=target[te,8:12]
        scores=[];base=[]
        for group in ('male','female'):
            rows=genders[te]==group
            estimate=pred[rows].mean(1)
            observed=np.nanmean(truth[rows],axis=1)
            scores.append(float(spearmanr(estimate,observed,nan_policy='omit').statistic))
            prior=np.nanmedian(target[tr[genders[tr]==group],:4])
            base.append(float(np.nanmean(abs(truth[rows]-prior))))
        results[key]=dict(selected_lambda=chosen,within_group_profile_srcc=float(np.mean(scores)),
                          single_clip_mae=float(np.nanmean(abs(pred-truth))),group_median_mae=float(np.mean(base)))
    return results


def main():
    data=load_rating_graph(('train','dev'))
    names,blocks=bank(data,False)
    with np.load(ROOT/'features/physical_quality_dev.npz') as f:
        ids=[f'{s}/parallel100/wav24kHz16bit/VOICEACTRESS100_{i:03d}.wav' for s in names for i in range(1,13)]
        if f['files'].tolist()!=ids:raise ValueError('Different physical feature order')
        raw=f['features']
    genders=np.array([data['manifest']['gender'][s] for s in names])
    reports=[]
    for repeat in range(5):
        rng=np.random.default_rng(9000+repeat)
        grouped=[rng.permutation(np.flatnonzero(genders==g)) for g in ('male','female')]
        folds=[np.concatenate([g[j::5] for g in grouped]) for j in range(5)]
        for k in range(5):
            te,va=folds[k],folds[(k+1)%5]
            tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
            x=np.concatenate(transform(blocks,tr),axis=-1)/np.sqrt(3)
            z,_=make_teacher(x,tr,va,paired_labels(data,names,va),train_triples(tr,names,data),881+k)
            reports.append(dict(repeat=repeat,fold=k,measurements=probes(z,raw,tr,va,te,genders)))
    summary={key:{metric:float(np.mean([r['measurements'][key][metric] for r in reports]))
                  for metric in ('within_group_profile_srcc','single_clip_mae','group_median_mae')} for key in FIELDS}
    write_json(ROOT/'results/acoustic_axis_probe.json',dict(summary=summary,folds=reports,units=['dB','dB','semitones','Hz'],
        scope='Linear readout of existing coordinates; F0 and HNR were also inputs. No orthogonality, causality or perceptual significance follows from decodability.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
