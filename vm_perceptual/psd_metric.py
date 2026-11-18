"""Fit a shared quadratic perceptual distance by ridge regression, with or without a PSD constraint.

Signed quadratic scores are a diagnostic. Only the PSD solution yields Euclidean coordinates.
"""
import argparse
import itertools
import numpy as np
from scipy.stats import rankdata
from dataset import load_data
from perceptual_rank import bank,transform
from wave_folds import paired_labels
from runtime import ROOT,write_json


def svec(A):
    d=A.shape[-1];a,b=np.triu_indices(d)
    return A[...,a,b]*np.where(a==b,1.,np.sqrt(2.))


def smat(v,d):
    a,b=np.triu_indices(d);M=np.zeros((d,d));M[a,b]=M[b,a]=v/np.where(a==b,1.,np.sqrt(2.))
    return M


def score(z,M,labels):
    ranks=[]
    for a,b,y in labels:
        delta=np.stack([z[a,i]-z[b,j] for i,j in itertools.permutations(range(4),2)])
        pred=-np.einsum('spd,de,spe->sp',delta,M,delta,optimize=True)
        v=rankdata(pred,axis=1);v-=v.mean(1,keepdims=True)
        g=rankdata(y);g-=g.mean()
        den=np.linalg.norm(v,axis=1)*np.linalg.norm(g)
        r=np.divide(v@g,den,out=np.zeros(len(den)),where=den>1e-12)
        ranks.append(np.mean(r))
    return float(np.mean(ranks))


def train_design(z,names,tr,data):
    local={names[k]:i for i,k in enumerate(tr)}
    rows=[r for r in data['pairs'] if r['speaker_a'] in local and r['speaker_b'] in local]
    a=np.array([local[r['speaker_a']] for r in rows]);b=np.array([local[r['speaker_b']] for r in rows])
    target=-np.array([r['score'] for r in rows])
    X=0
    for i,j in itertools.permutations(range(4),2):
        e=z[tr[a],i]-z[tr[b],j]
        X=X+svec(e[:,:,None]*e[:,None,:])/12
    # Unpenalized rating intercept per gender. The learned metric remains shared.
    for g in ('female','male'):
        mask=np.array([r['gender']==g for r in rows]);target[mask]-=target[mask].mean()
        X[mask]-=X[mask].mean(0)
    return X,target


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,required=True);a=ap.parse_args()
    data=load_data();names,blocks=bank(data)
    rng=np.random.default_rng(9000+a.repeat)
    buckets=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    folds=[np.concatenate([g[j::5] for g in buckets]) for j in range(5)]
    folder=ROOT/'runs'/f'psd_metric_{a.repeat}';folder.mkdir(exist_ok=True)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        vl=paired_labels(data,names,va);tl=paired_labels(data,names,te)
        parts=transform(blocks[:3],tr)
        x=np.concatenate(parts,axis=-1)/np.sqrt(3)
        flat=x[tr,:4].reshape(-1,x.shape[-1]);mu=flat.mean(0)
        _,_,vt=np.linalg.svd(flat-mu,full_matrices=False)
        rows={}
        for d in (4,8,16,32):
            z=(x-mu)@vt[:d].T
            z=z/np.sqrt(np.mean((z[tr,:4]**2).sum(-1)))
            X,y=train_design(z,names,tr,data);p=X.shape[1]
            # Unit-metric target has a training-only scale from one simple regression.
            idv=svec(np.eye(d));u=X@idv
            prior=max(0,float(u@y)/max(float(u@u),1e-12))*idv
            gram=X.T@X;vals,U=np.linalg.eigh(gram)
            grad=U.T@X.T@(y-X@prior)
            for lam in (.001,.01,.1,1.,10.):
                reg=lam*max(float(np.trace(gram)/p),1e-8)
                beta=prior+U@(grad/(np.maximum(vals,0)+reg))
                m=smat(beta,d);eig,ev=np.linalg.eigh(m)
                mass=float(np.abs(eig[eig<0]).sum()/max(np.abs(eig).sum(),1e-12))
                positive=(ev*np.maximum(eig,0))@ev.T
                for typ,M in [('signed',m),('psd',positive)]:
                    key=f'{typ}_{d}_{lam}'
                    rows[key]=dict(development=score(z[va,4:8],M,vl),outer=score(z[te,8:12],M,tl),dimension=d,
                        effective_rank=int((eig>1e-8).sum()),negative_mass=mass)
            rows[f'pca_{d}']=dict(development=score(z[va,4:8],np.eye(d),vl),outer=score(z[te,8:12],np.eye(d),tl),dimension=d)
        chosen={k:max((n for n in rows if n.startswith(k+'_')),key=lambda n:rows[n]['development']) for k in ('pca','psd','signed')}
        write_json(dest,dict(choices=chosen,candidates=rows))
        print(a.repeat,fold,{k:round(rows[n]['outer'],3) for k,n in chosen.items()},flush=True)


if __name__=='__main__':main()
