"""Represent a recording by frame means and a shrinkage log-covariance vector.

All axes and scales are learned from training people and training sentences.
The sqrt(2) factor on off-diagonal matrix entries preserves Frobenius distance.
"""
import argparse
import numpy as np
from runtime import ROOT,write_json
from dataset import load_data,unit
from perceptual_rank import metrics
from wave_folds import paired_labels


def moments(names):
    mu=[];cov=[]
    for s in names:
        with np.load(ROOT/'features/content_frames'/f'{s}.npz') as f:
            frames=np.split(f['shallow'].astype('float32'),np.cumsum(f['lengths'])[:-1])
        per_mean=[];per_cov=[]
        for x in frames:
            m=x.mean(0);res=x-m
            per_mean.append(m);per_cov.append(res.T@res/max(1,len(res)-1))
        mu.append(per_mean);cov.append(per_cov)
    return np.asarray(mu),np.asarray(cov)


def scale(x,tr):
    train=x[tr,:4].reshape(-1,x.shape[-1])
    centered=x-train.mean(0)
    rms=np.sqrt(np.mean(np.sum((train-train.mean(0))**2,axis=-1)))
    return centered/max(rms,1e-8)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,required=True);a=ap.parse_args()
    data=load_data();names=[s for s in data['speakers'] if data['manifest']['split'][s]!='test']
    mean,cov=moments(names)
    rng=np.random.default_rng(9000+a.repeat)
    people=[rng.permutation([i for i,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([g[i::5] for g in people]) for i in range(5)]
    folder=ROOT/'runs'/f'covariance_geometry_{a.repeat}';folder.mkdir(exist_ok=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        labels=(paired_labels(data,names,va),paired_labels(data,names,te));rows={}
        def assess(name,z):
            z=unit(z)
            rows[name]=dict(dev=metrics(z[va,4:8],labels[0]),outer=metrics(z[te,8:12],labels[1]))
        raw=scale(mean,tr);assess('mean',raw)
        science=cov[tr,:4].mean((0,1));val,axes=np.linalg.eigh(science)
        axes=(axes[:,::-1]/np.sqrt(np.maximum(val[::-1],1e-6)))
        for dim in (8,16,32):
            w=axes[:,:dim]
            transformed=w.T@cov@w
            ix=np.triu_indices(dim);factors=np.where(ix[0]==ix[1],1.,np.sqrt(2.))
            for shrink in (.05,.35):
                shrunk=(1-shrink)*transformed+shrink*np.eye(dim)
                ev,v=np.linalg.eigh(shrunk)
                logs=(v*np.log(np.maximum(ev,1e-7))[...,None,:])@np.swapaxes(v,-1,-2)
                dst=scale(logs[:,:,ix[0],ix[1]]*factors,tr)
                diag=scale(np.log(np.diagonal(shrunk,axis1=-2,axis2=-1).clip(1e-7)),tr)
                for shape,stats in [('diag',diag),('full',dst)]:
                    for weight in (.25,.5,.75):
                        z=np.concatenate((np.sqrt(1-weight)*raw,np.sqrt(weight)*stats),axis=-1)
                        tag=f'd{dim}_{shape}_s{shrink}_w{weight}'
                        assess(tag,z)
        choice=max(rows,key=lambda k:rows[k]['dev']['single_srcc'])
        diagonal=max((k for k in rows if '_diag_' in k or k=='mean'),key=lambda k:rows[k]['dev']['single_srcc'])
        full=max((k for k in rows if '_full_' in k or k=='mean'),key=lambda k:rows[k]['dev']['single_srcc'])
        write_json(dest,dict(choice=choice,diagonal_choice=diagonal,full_choice=full,candidates=rows))
        print(a.repeat,fold,choice,rows[choice]['outer'],flush=True)


if __name__=='__main__':main()
