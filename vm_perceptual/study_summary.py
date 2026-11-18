"""Aggregate pre-specified repeated speaker folds without treating them as independent people."""
import json
import numpy as np
from runtime import ROOT,write_json


def describe(rows):
    table=np.asarray(rows,float)
    return dict(mean=float(table.mean()),fold_q25=float(np.quantile(table,.25)),fold_q75=float(np.quantile(table,.75)),
        repeat_means=table.mean(1).tolist())


def load(folder,filename):
    path=ROOT/'runs'/folder/filename
    return json.loads(path.read_text())


def collect():
    studies={}
    for recipe in ('acoustic','acoustic_ecapa'):
        rows=[[load(f'wave_folds_{r}_2000',f'{f}_{recipe}.json') for f in range(5)] for r in range(5)]
        studies['wave_'+recipe]={k:describe([[x['outer_scores'][k] for x in rep] for rep in rows]) for k in ('0','1','2','4')}
    rows=[[load(f'perceptual_rank_{r}',f'fold_{f}.json') for f in range(5)] for r in range(5)]
    for kind in ('fixed','trained'):
        for key in rows[0][0][kind]:
            def score(x,m):
                v=x[kind][key]
                return (v if kind=='fixed' else v['outer'])[m]
            studies['rank_'+key]={m:describe([[score(x,m) for x in rep] for rep in rows]) for m in ('single_srcc','anchor_agreement')}
    rows=[[load(f'ordinal_geometry_{r}',f'{f}.json') for f in range(5)] for r in range(5)]
    for key in rows[0][0]:
        studies['ordinal_'+key]={m:describe([[x[key][block][field] for x in rep] for rep in rows]) for m,block,field in (
            ('nll','outer','nll'),('prior_nll','train_prior','nll'),('single_srcc','single','single_srcc'))}
    rows=[[load(f'soft_content_{r}',f'{f}.json') for f in range(5)] for r in range(5)]
    for key in ('unconditional','dev_selected'):
        def method(x):return x['candidates'][key if key=='unconditional' else x['selected']]
        studies['content_'+key]={m:describe([[method(x)['outer'][m] for x in rep] for rep in rows]) for m in ('single_srcc','within_over_between','sentence_id_top1')}
    return studies


if __name__=='__main__':
    out=collect()
    write_json(ROOT/'results/repeated_fold_studies.json',dict(studies=out,people=80,repeats=5,folds_per_repeat=5,
        scope='Overlapping speaker resamples and shared corpus; fold quantiles describe variability.'))
    print('summarized',len(out),'conditions')
