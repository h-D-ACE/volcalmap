"""Compare reruns by recording ID and audio hash, using full-space retrieval."""
from runtime import ROOT, write_json
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def read(root, name):
    return json.loads((root/name).read_text(encoding='utf8'))


def compare(left, right):
    caches=[read(p,'features/jvs_features.json') for p in (left,right)]
    ids=sorted(caches[0])
    assert set(ids)==set(caches[1]) and all(caches[0][k]['sha256']==caches[1][k]['sha256'] for k in ids)
    result=dict(recordings=len(ids), features={})
    for key in ('acoustic','ecapa','wavlm3'):
        a,b=(np.array([cache[i][key] for i in ids]) for cache in caches)
        result['features'][key]=dict(max_absolute=float(abs(a-b).max()),rms=float(np.sqrt(np.mean((a-b)**2))))
    maps=[{r['record_id']:r for r in read(p,'results/map_index.json')['records']} for p in (left,right)]
    assert all(set(m)==set(ids) for m in maps)
    x,y=(np.asarray([m[i]['vector'] for i in ids]) for m in maps)
    result['coordinate_l2_median']=float(np.median(np.linalg.norm(x-y,axis=1))) if x.shape==y.shape else None
    result['coordinate_l2_max']=float(np.linalg.norm(x-y,axis=1).max()) if x.shape==y.shape else None
    distances=[np.linalg.norm(z[:,None]-z[None,:],axis=2) for z in (x,y)]
    upper=np.triu_indices(len(ids),1)
    result['distance_srcc']=float(spearmanr(*(d[upper] for d in distances)).statistic)
    mask=np.equal.outer([caches[0][i]['sha256'] for i in ids],[caches[0][i]['sha256'] for i in ids])
    ranks=[np.argsort(np.where(mask,np.inf,d),axis=1,kind='stable') for d in distances]
    result['matching_top1']=int(np.sum(ranks[0][:,0]==ranks[1][:,0]))
    result['mean_top5_overlap']=float(np.mean([len(set(a[:5]) & set(b[:5]))/5 for a,b in zip(*ranks)]))
    result['evaluations']=[]
    for p in (left,right):
        e=read(p,'results/evaluation.json'); winner=e['winner']['name']
        result['evaluations'].append(dict(winner=winner,test=e['models'][winner]['test']['mean_spearman']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--reference',type=Path,default=ROOT)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=compare(args.reference,args.candidate)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))
