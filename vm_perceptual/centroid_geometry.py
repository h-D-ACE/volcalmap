"""Estimate occupied dimensions of voice centroids and record the effect of grouping."""
import json
from collections import defaultdict
import numpy as np
from dataset import load_rating_graph, unit
from runtime import ROOT,write_json


def spectrum(points):
    x=np.asarray(points,dtype=np.float64);x-=x.mean(0)
    eigen=np.maximum(np.linalg.eigvalsh(x@x.T)[::-1],0)
    power=eigen/eigen.sum()
    return dict(voices=len(x),coordinates=x.shape[-1],effective_dimension=float(1/(power@power)),
                dimensions_for_90_percent=int(np.searchsorted(np.cumsum(power),.9)+1),
                leading_two_fraction=float(power[:2].sum()))


def main():
    data=load_rating_graph()
    methods={}
    for k,file in [('current','map_index'),('student','wave_index'),('rank_own','rank_own_index'),
                   ('robust_student','robust_student_index')]:
        index=json.loads((ROOT/'results'/f'{file}.json').read_text())
        groups=defaultdict(list)
        for r in index['records']:
            s=r['record_id'].split('/')[0]
            if '/parallel100/' in r['record_id']:groups[s].append(unit(np.asarray(r['vector'])))
        if len(groups)!=100 or any(len(v)!=4 for v in groups.values()):raise ValueError('Expected 100 four-text voice profiles')
        centers={s:unit(np.mean(v,axis=0)) for s,v in groups.items()}
        result={}
        for split in ('all','train','test'):
            for gender in ('all','male','female'):
                keep=[s for s in centers if (split=='all' or data['manifest']['split'][s]==split)
                      and (gender=='all' or data['manifest']['gender'][s]==gender)]
                result[f'{split}_{gender}']=spectrum([centers[s] for s in keep])
        methods[k]=result
    report=dict(models=methods,scope='Four normal utterances per person. Centroid spectra depend on sample, between-group structure and recording protocol. No estimate of a universally optimal embedding width.')
    write_json(ROOT/'results/centroid_geometry.json',report)
    print({k:v['all_all'] for k,v in methods.items()},flush=True)


if __name__=='__main__':main()
