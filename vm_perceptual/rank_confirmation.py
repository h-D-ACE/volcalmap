"""Matched uncertainty and fresh-text checks for experimental compact ranking maps."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from dataset import load_data
from gallery_benchmark import quality
from reference_benchmark import benchmark
from rank_map_candidate import Frontends,encode
from runtime import ROOT,write_json


def main():
    data=load_data();names=data['speakers'];groups=data['manifest']['split']
    keys=('current','student','voxsim','own','combined')
    bank={k:{} for k in keys}
    for s in names:
        with np.load(ROOT/'features/gallery_vectors'/f'{s}.npz') as f:
            for k in keys[:3]:bank[k][s]=f[k]
        for k in keys[3:]:bank[k][s]=np.load(ROOT/'features'/f'rank_map_{k}'/f'{s}.npy')
    report={}
    for group,start,end in [('dev',4,8),('test',0,4),('fresh',4,12)]:
        split='test' if group=='fresh' else group
        subset=[s for s in names if groups[s]==split and len(bank['current'][s])>=end]
        maps={k:{s:bank[k][s][start:end] for s in subset} for k in keys}
        report[group]=benchmark(subset,data,maps,2762)
    # Preview fidelity only. Fit display axes on train, evaluate all clips of the
    # original dev split. Same-person candidates are excluded here.
    preview={}
    for k in keys[3:]:
        with np.load(ROOT/'runs'/f'rank_map_{k}.npz') as par:
            who=[s for s in names if groups[s]=='dev']
            z=np.concatenate([bank[k][s] for s in who])
            speaker=np.repeat(who,[len(bank[k][s]) for s in who])
            mask=speaker[:,None]==speaker[None,:]
            def ranks(vec):
                d=np.linalg.norm(vec[:,None]-vec[None,:],axis=-1)
                return np.argsort(np.where(mask,np.inf,d),axis=1)[:,:5],d[~mask]
            truth,dt=ranks(z);scores={}
            for dim in (1,2,3,5,8,16):
                rr,dd=ranks((z-par['view_mean'])@par['view_axes'][:dim].T)
                scores[str(dim)]=dict(recall5=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(rr,truth)])),
                    pair_srcc=float(spearmanr(dt,dd).statistic))
            preview[k]=scores
    report['preview_dev']=preview
    # Re-encode selected paths from original waveforms and compare to frozen cache.
    front=Frontends(True);consistency={}
    for k in keys[3:]:
        values=[]
        with np.load(ROOT/'runs'/f'rank_map_{k}.npz') as p:
            for group in ('train','dev','test'):
                s=next(s for s in names if groups[s]==group)
                for i in (0,3):
                    file=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))[i]
                    new=encode(front.extract(file),p)
                    values.append(float(np.linalg.norm(new-bank[k][s][i])))
        consistency[k]=dict(n=len(values),max_l2=max(values))
    report['cache_vs_one_audio']=consistency
    write_json(ROOT/'results/rank_confirmation.json',report)


if __name__=='__main__':main()
