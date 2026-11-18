"""Query-centered view with exact high-dimensional distances from the focus recording."""
import argparse
import json
from pathlib import Path
import numpy as np
from map_index import search
from runtime import write_json


def focus(index,query,k=12,dimensions=2):
    if dimensions not in (1,2,3):raise ValueError('Display dimensions must be 1, 2 or 3')
    result=search(index,query,k=k)
    matches=result['matches']
    if not matches:return dict(dimensions=dimensions,query=[0.]*dimensions,neighbors=[],model_version=index['model_version'])
    records={r['record_id']:r for r in index['records']}
    offsets=np.array([records[m['record_id']]['vector'] for m in matches])-query['timbre']['vector']
    # Directions use the local principal axes of differences from the query.
    # Radius always carries the full-space Euclidean distance. Angles are approximate.
    _,_,axes=np.linalg.svd(offsets,full_matrices=False)
    axes=axes[:dimensions].copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))]<0:axis*=-1
    projection=offsets@axes.T
    unit=np.zeros((len(matches),dimensions));unit[:,:len(axes)]=projection
    norms=np.linalg.norm(unit,axis=1)
    degenerate=norms<1e-12
    unit[~degenerate]/=norms[~degenerate,None];unit[degenerate,0]=1.
    radii=np.linalg.norm(offsets,axis=1)
    local=unit*radii[:,None]
    items=[dict(record_id=m['record_id'],distance=float(r),coordinates=p.tolist()) for m,r,p in zip(matches,radii,local)]
    return dict(dimensions=dimensions,model_version=index['model_version'],query=[0.]*dimensions,neighbors=items,
        scope='Distances from the query are exact; angles and neighbor-to-neighbor distances are local display approximations.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path);p.add_argument('query',type=Path);p.add_argument('--dimensions',type=int,choices=(1,2,3),default=2)
    p.add_argument('--k',type=int,default=12);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    write_json(a.output,focus(json.loads(a.index.read_text(encoding='utf-8')),json.loads(a.query.read_text(encoding='utf-8')),a.k,a.dimensions))
