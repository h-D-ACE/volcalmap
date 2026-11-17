"""Frozen models on newly available texts with paired speaker bootstrap."""
import itertools,json
from pathlib import Path
from runtime import ROOT, write_json
from layer_blend_map import Encoder
from dataset import load_data, unit
from infer import encode
from scipy.stats import rankdata
import numpy as np
import torch


def rank_scores(y,p):
    a=rankdata(y);a-=a.mean()
    b=rankdata(p,axis=1);b-=b.mean(1,keepdims=True)
    den=np.linalg.norm(a)*np.linalg.norm(b,axis=1)
    return np.divide(b@a,den,out=np.full(len(b),np.nan),where=den>0)


def main():
    data=load_data()
    names=[s for s in data['speakers'] if data['manifest']['split'][s]=='test']
    planned=json.loads((ROOT/'data_sources/test_extra_text_selection.json').read_text())
    paths={s:sorted(ROOT/'data_jvs'/Path(n).relative_to('jvs_ver1') for n in planned if Path(n).parts[1]==s) for s in names}
    counts={s:sum(p.is_file() for p in files) for s,files in paths.items()}
    names=[s for s in names if counts[s]==8]
    encoder=Encoder(ROOT/'runs/layer_blend_dev.npz',ROOT.parent/'claude_baselines/models','cuda')
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    vec={k:[] for k in ('blend','selected','acoustic')}
    for s in names:
        per={k:[] for k in vec}
        for path in paths[s]:
            z,f=encoder(path);per['blend'].append(z);per['selected'].append(encode(f,ck))
            v=(np.asarray(f['acoustic'])-encoder.params['acoustic_mean'])/encoder.params['acoustic_scale']
            per['acoustic'].append(unit(v))
        for k in vec:vec[k].append(per[k])
    rows=[r for r in data['pairs'] if r['speaker_a'] in names and r['speaker_b'] in names]
    a,b=([names.index(r[key]) for r in rows] for key in ('speaker_a','speaker_b'))
    y=np.array([r['score'] for r in rows])
    pairs=list(itertools.product(range(4),range(4,8)))
    pairs+= [(j,i) for i,j in pairs]
    scores={};pred={}
    for key,array in vec.items():
        z=np.asarray(array);pred[key]=np.stack([-np.linalg.norm(z[a,i]-z[b,j],axis=1) for i,j in pairs])
        scores[key]={}
        for g in ('male','female'):
            m=np.array([r['gender']==g for r in rows]);v=rank_scores(y[m],pred[key][:,m])
            scores[key][g]=dict(forward=float(v[:16].mean()),reverse=float(v[16:].mean()),pairs=int(sum(m)))
    rng=np.random.default_rng(20261118);deltas=[]
    groups=[np.array([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    for _ in range(2000):
        ds=[]
        for group in groups:
            freq=np.bincount(rng.choice(group,len(group),replace=True),minlength=len(names))
            ix=np.repeat(np.arange(len(rows)),freq[a]*freq[b])
            if len(np.unique(ix))<3:break
            ds.append(np.mean(rank_scores(y[ix],pred['blend'][:,ix])-rank_scores(y[ix],pred['selected'][:,ix])))
        if len(ds)==2 and np.all(np.isfinite(ds)):deltas.append(np.mean(ds))
    result=dict(available=counts,complete_people=len(names),scores=scores,bootstrap_valid=len(deltas),paired_delta_interval=np.quantile(deltas,[.025,.975]).tolist(),scope='Partial fixed selection, prior JVS test labels; different texts across recordings')
    write_json(ROOT/'results/frozen_new_texts.json',result)
    print(result)


if __name__=='__main__':
    main()
