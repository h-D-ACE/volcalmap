"""Audit how much of the frozen map ordering is explained by mean pitch alone."""
from runtime import ROOT,write_json
from dataset import load_data,transform
from train_map import MetricHead
from scipy.stats import rankdata,spearmanr
import numpy as np
import torch


def residual_rank(y,x):
    a=rankdata(y);d=np.c_[np.ones(len(x)),rankdata(x)]
    return a-d@np.linalg.lstsq(d,a,rcond=None)[0]


def main():
    data=load_data();ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    head=MetricHead(ck['input_dim'],ck['output_dim']);head.load_state_dict(ck['model'])
    profiles=[];raw=[]
    for s in data['speakers']:
        f=[data['features'][p] for p in data['clips'][s]]
        profiles.append(np.mean([transform(row,ck['preprocessing'],pca=True) for row in f],axis=0))
        raw.append(np.mean([row['acoustic'] for row in f],axis=0))
    with torch.no_grad():z=head(torch.tensor(np.asarray(profiles),dtype=torch.float32)).numpy()
    raw=np.array(raw);pitch=raw[:,-2]*12/np.log(2);hnr=raw[:,-1]
    results={}
    for split in ('train','dev','test'):
        for g in ('male','female'):
            rows=[r for r in data['pairs'] if r['split']==split and r['gender']==g]
            a,b=([r[k] for r in rows] for k in ('a','b'))
            y=np.array([r['score'] for r in rows]);p=-abs(pitch[a]-pitch[b]);h=-abs(hnr[a]-hnr[b])
            m=-np.linalg.norm(z[a]-z[b],axis=1)
            rp=residual_rank(y,p);rm=residual_rank(m,p)
            near=(-p<=2)
            results[f'{split}_{g}']=dict(pairs=len(y),map_srcc=float(spearmanr(y,m).statistic),pitch_srcc=float(spearmanr(y,p).statistic),hnr_srcc=float(spearmanr(y,h).statistic),map_pitch_srcc=float(spearmanr(m,p).statistic),partial_map_after_pitch=float(np.corrcoef(rp,rm)[0,1]),within_two_semitones=int(sum(near)),within_two_semitone_map_srcc=float(spearmanr(y[near],m[near]).statistic) if sum(near)>=5 else None)
    print(results);write_json(ROOT/'results/pitch_audit.json',dict(results=results,scope='Frozen four-clip profiles; within-gender associations, no causal interpretation; statistics share speakers'))


if __name__=='__main__':
    main()
