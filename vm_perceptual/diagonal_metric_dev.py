"""Speaker-disjoint development of a regularized perceptual metric on training voices."""
import itertools
from runtime import ROOT, write_json
from dataset import load_data, unit
from train_map import correlations
import numpy as np
from scipy.optimize import lsq_linear


def main():
    data=load_data()
    values=np.asarray([[data['features'][name]['acoustic'] for name in data['clips'][s]] for s in data['speakers']])
    tr=np.flatnonzero(data['split']=='train')
    rng=np.random.default_rng(20261117)
    gender=data['manifest']['gender']
    groups=[rng.permutation([i for i in tr if gender[data['speakers'][i]]==g]) for g in sorted(set(gender.values()))]
    folds=[np.concatenate([g[j::3] for g in groups]) for j in range(3)]
    pairs=[(i,j) for i,j in itertools.permutations(range(4),2)]
    def run(train,valid):
        flat=values[train].reshape(-1,values.shape[-1])
        x=(values-flat.mean(0))/np.maximum(flat.std(0),.05)
        z=unit(x)
        trainset=set(train); valset=set(valid)
        trs=[r for r in data['pairs'] if r['a'] in trainset and r['b'] in trainset]
        vas=[r for r in data['pairs'] if r['a'] in valset and r['b'] in valset]
        def differences(rows):
            a,b=([r[k] for r in rows] for k in ('a','b'))
            return np.stack([(z[a,i]-z[b,j])**2 for i,j in pairs])
        t,v=differences(trs),differences(vas)
        design=t.mean(0);y=-np.array([r['score'] for r in trs])
        dx=design-design.mean(0);dy=y-y.mean()
        res={}
        for penalty in (None,.01,.1,1.,10.):
            if penalty is None:weights=np.ones(x.shape[-1])
            else:
                prior=np.full(x.shape[-1],max(0.,np.linalg.lstsq(dx.sum(1)[:,None],dy,rcond=None)[0][0]))
                weight=np.sqrt(penalty)
                weights=lsq_linear(np.vstack([dx,weight*np.eye(x.shape[-1])]),np.r_[dy,weight*prior],bounds=(0,np.inf)).x
            res[str(penalty)]=float(np.mean([correlations(vas,-d@weights)['mean_spearman'] for d in v]))
        return res
    inner=[run(np.setdiff1d(tr,fold),fold) for fold in folds]
    means={k:float(np.mean([row[k] for row in inner])) for k in inner[0]}
    choice=max(means,key=means.get)
    dev=run(tr,np.flatnonzero(data['split']=='dev'))
    out=dict(train_only_fold_scores=inner,train_cv_mean=means,choice=choice,dev_selected=dev[choice],dev_unweighted=dev['None'])
    print(out);write_json(ROOT/'results/diagonal_metric_dev.json',out)


if __name__=='__main__':
    main()
