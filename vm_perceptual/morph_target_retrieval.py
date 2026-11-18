"""Retrieve the intended synthetic target among unrelated voices using other sentences."""
from collections import defaultdict
import numpy as np
from dataset import load_rating_graph
from runtime import ROOT, write_json


def main():
    data=load_rating_graph(('dev',))
    names=set(data['speakers'])
    sexes=data['manifest']['gender']
    paths=sorted((ROOT/'features/morph_paths').glob('*.npz'))
    if len(paths)!=80:raise ValueError('Run morph_paths.py first')
    output={}
    for method in ('current','student','own','identity'):
        natural={};series={}
        for path in paths:
            a,b,slot=path.stem.split('_');slot=int(slot)
            if a not in names or b not in names or sexes[a]!=sexes[b]:raise ValueError(path)
            with np.load(path) as f:z=f[method]
            for s,v in ((a,z[0]),(b,z[1])):
                if (s,slot) in natural and not np.allclose(natural[s,slot],v,atol=1e-5):
                    raise ValueError('Inconsistent natural reference')
                natural[s,slot]=v
            series[a,b,slot]=z[2:]
        rows=defaultdict(list)
        for (a,b,slot),trajectory in series.items():
            gallery=sorted(s for s in names if s!=a and sexes[s]==sexes[a])
            i=gallery.index(b)
            anchors=np.stack([np.mean([natural[s,j] for j in range(4) if j!=slot],axis=0) for s in gallery])
            distances=np.linalg.norm(trajectory[:,None]-anchors[None,:],axis=-1)
            gains=distances[0]-distances
            negatives=np.arange(len(gallery))!=i
            for level in range(5):
                d=distances[level];g=gains[level]
                rank=1+int((d<d[i]).sum())
                # At zero all directions tie; count ties as equal shares.
                gain_rank1=float(g[i]==max(g))/np.sum(g==max(g))
                rows[level].append((a,rank==1,rank<=3,gain_rank1,
                                    float(g[i]-np.median(g[negatives])),rank))
        result=[]
        for level,values in rows.items():
            v=np.array([r[1:] for r in values],float)
            result.append(dict(fraction=level/4,target_first=float(v[:,0].mean()),target_top3=float(v[:,1].mean()),
                largest_improvement_is_target=float(v[:,2].mean()),target_gain_above_median_distractor=float(np.median(v[:,3])),
                mean_target_rank=float(v[:,4].mean())))
        output[method]=result
    write_json(ROOT/'results/morph_target_retrieval.json',dict(paths=len(paths),voices=len(names),methods=output,
        setup='Same-sex nine-candidate gallery excludes source voice. Candidate profiles average the other three sentences.',
        scope='WORLD synthesizer manipulates pitch, spectrum and aperiodicity toward a known reference. No natural imitation labels.'))
    print({k:[round(p['target_first'],3) for p in v] for k,v in output.items()},flush=True)


if __name__=='__main__':main()
