"""Calibrate engineering radii for crop-to-full-utterance displacement in a fixed map."""
import math
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor
from dataset import load_data
from runtime import ROOT,write_json
from neighbor_margin import boundary_radius


def interval(hits,n):
    return [0. if hits==0 else float(beta.ppf(.025,hits,n-hits+1)),1. if hits==n else float(beta.ppf(.975,hits+1,n-hits))]


def main():
    data=load_data();collection={k:[] for k in ('train','dev','test')};gallery={}
    rng=np.random.default_rng(49912)
    for name in data['speakers']:
        with np.load(ROOT/'features/crop_records'/f'{name}.npz') as f:
            for sec in (1,2,4):
                sel=np.flatnonzero(f['condition'][:,0]==sec)
                if not len(sel):continue
                if data['manifest']['split'][name]!='train':
                    # Exactly one predetermined random available crop per duration per speaker; no error enters selection.
                    sel=np.array([rng.choice(sel)])
                for j in sel:
                    error=float(np.linalg.norm(f['prior'][j]-f['targets'][j]))
                    feature=np.r_[sec,f['features'][j]];collection[data['manifest']['split'][name]].append(
                        (name,sec,feature,error,f['prior'][j],f['targets'][j]))
            if data['manifest']['split'][name]=='train':
                ix=np.flatnonzero(f['condition'][:,0]==0)
                gallery[name]=f['targets'][ix[:4]].mean(0)
    tr=collection['train'];x=np.stack([r[2] for r in tr]);y=np.array([r[3] for r in tr])
    medians={sec:np.median([r[3] for r in tr if r[1]==sec]) for sec in (1,2,4)}
    model=HistGradientBoostingRegressor(loss='quantile',quantile=.75,max_iter=80,max_leaf_nodes=7,min_samples_leaf=60,random_state=451)
    model.fit(x,y);report={}
    for method in ('duration','feature_scaled'):
        per={}
        for sec in (1,2,4):
            va=[r for r in collection['dev'] if r[1]==sec];te=[r for r in collection['test'] if r[1]==sec]
            def scale(rows):
                if method=='duration':return np.full(len(rows),medians[sec])
                return np.maximum(.05,model.predict(np.stack([r[2] for r in rows])))
            errs=lambda rows:np.array([r[3] for r in rows])
            conformity=np.sort(errs(va)/scale(va))
            # Finite-sample split-conformal quantile (one calibration observation per person).
            rank=math.ceil((len(va)+1)*.90)
            if rank>len(va):raise ValueError('Too few speakers for 90 percent calibration')
            scale_factor=conformity[rank-1];radius=scale(te)*scale_factor
            hits=int(np.sum(errs(te)<=radius));cert=actual=violations=0;margins=[]
            for r,bound in zip(te,radius):
                gender=data['manifest']['gender'][r[0]]
                gal=np.stack([z for s,z in gallery.items() if data['manifest']['gender'][s]==gender])
                first,margin=boundary_radius(r[4],gal);margins.append(margin)
                nearest=int(np.argmin(np.linalg.norm(gal-r[5],axis=1)))
                safe=bound<margin;cert+=int(safe);actual+=int(first==nearest)
                # Exact geometric check using the observed displacement, independent of calibrated radius.
                if r[3]<margin-1e-6 and first!=nearest:violations+=1
            per[str(sec)]=dict(calibration_people=len(va),test_people=len(te),radius_median=float(np.median(radius)),
                hits=hits,coverage=hits/len(te),coverage_exact_95=interval(hits,len(te)),certified_by_calibrated_radius=cert,
                actually_same_neighbor=actual,geometric_violations=violations,median_boundary_radius=float(np.median(margins)))
        report[method]=per
    write_json(ROOT/'results/crop_uncertainty.json',dict(nominal=.90,models=report,
        scope='Distance to the complete version of the same recording. Coverage is empirical across the held-out people and shifted sentence sets.'))
    print(report,flush=True)


if __name__=='__main__':main()
