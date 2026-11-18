"""Compare two recording maps from the same model and align by exact recording IDs and file hashes."""
import argparse
import json
from pathlib import Path
from runtime import write_json
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr


def compare(a,b):
    if a['model_version']!=b['model_version'] or a['dimensions']!=b['dimensions']:raise ValueError('Different encoders')
    def index(rows):
        d={r['record_id']:r for r in rows}
        if len(d)!=len(rows):raise ValueError('Duplicate record identifier')
        return d
    left=index(a['records']);right=index(b['records'])
    if left.keys()!=right.keys():raise ValueError('Different recording sets')
    ids=sorted(left)
    if any(left[k]['sample_id']!=right[k]['sample_id'] for k in ids):raise ValueError('Audio hash mismatch')
    x,y=[np.stack([side[k]['vector'] for k in ids]) for side in (left,right)]
    u,v=pdist(x),pdist(y);delta=np.linalg.norm(x-y,axis=1)
    dup=np.equal.outer([left[k]['sample_id'] for k in ids],[left[k]['sample_id'] for k in ids])
    orders=[np.argsort(np.where(dup,np.inf,squareform(d)),axis=1,kind='stable') for d in (u,v)]
    matches=np.sum(orders[0][:,0]==orders[1][:,0])
    overlaps=[len(set(i[:5])&set(j[:5])) for i,j in zip(*orders)]
    return dict(recordings=len(ids),model_version=a['model_version'],median_l2=float(np.median(delta)),max_l2=float(delta.max()),
        same_top1=int(matches),same_all_top5=int(np.sum(np.asarray(overlaps)==5)),mean_top5_overlap=float(np.mean(overlaps)/5),
        pair_geometry_srcc=float(spearmanr(u,v).statistic))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    out=compare(*[json.loads(path.read_text(encoding='utf-8-sig')) for path in (args.a,args.b)])
    write_json(args.output,out);print(out)
