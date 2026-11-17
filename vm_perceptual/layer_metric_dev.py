"""Train a nonnegative mixture of frozen timbre distances, with speaker-disjoint selection."""
import itertools,json
from runtime import ROOT, write_json
from dataset import load_data, unit
from train_map import correlations
from scipy.optimize import nnls
from scipy.stats import rankdata
import numpy as np


def normalized(x, train, local_scale=False):
    if not local_scale:
        x=unit(x)
    f=x[train].reshape(-1,x.shape[-1])
    return unit((x-f.mean(0))/(np.maximum(f.std(0),.05) if local_scale else 1))


def mean_rank(x,y):
    if len(x)<3:return np.nan
    a,b=rankdata(x),rankdata(y)
    return float(np.corrcoef(a,b)[0,1]) if np.std(b)>0 and np.std(a)>0 else np.nan


def run():
    data=load_data();features=np.load(ROOT/'features/development_layers.npz')
    names=features['speakers'].tolist();genders=data['manifest']['gender']
    bank={'acoustic':features['acoustic'],'ecapa':features['ecapa']}
    layers=features['layers']
    for k in range(layers.shape[2]):
        bank[f'wavlm_{k}']=layers[:,:,k,:]
    idx={s:i for i,s in enumerate(names)}
    train=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='train'])
    dev=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='dev'])
    def evaluate(tr,va):
        trs={names[i] for i in tr};vas={names[i] for i in va}
        rows=lambda ids:[r for r in data['pairs'] if r['speaker_a'] in ids and r['speaker_b'] in ids]
        rt,rv=rows(trs),rows(vas)
        ab=lambda rows:([idx[r['speaker_a']] for r in rows],[idx[r['speaker_b']] for r in rows])
        ta,tb=ab(rt);a,b=ab(rv)
        # Training target averages all cross-text distances; validation compares individual clips.
        train_design=[];val_design=[]
        for k,raw in bank.items():
            z=normalized(raw,tr,k=='acoustic')
            first=z[:,:4]
            dots=(first[ta].sum(1)*first[tb].sum(1)).sum(-1)-(first[ta]*first[tb]).sum((-1,-2))
            train_design.append(2-2*dots/12)
            val_design.append(np.stack([((z[a,i]-z[b,j])**2).sum(-1) for i in range(4,8) for j in range(8,12)]))
        t=np.stack(train_design,axis=-1);v=np.stack(val_design,axis=-1)
        y=-np.asarray([r['score'] for r in rt]);out={};preds={};fits={}
        t=t-t.mean(0);y-=y.mean()
        for k,key in enumerate(bank):
            preds[key]=-v[:,:,k]
        for lam in (.1,1,10,100):
            design=np.vstack([t,np.sqrt(lam)*np.eye(t.shape[1])])
            weights=nnls(design,np.r_[y,np.zeros(t.shape[1])])[0]
            key=f'blend_{lam:g}';preds[key]=-v@weights
            fits[key]=dict(zip(bank,weights.tolist()))
        gold=np.asarray([r['score'] for r in rv])
        groups=[np.array([r['gender']==g for r in rv]) for g in ('female','male')]
        for key,p in preds.items():
            out[key]=float(np.mean([mean_rank(gold[m],row[m]) for row in p for m in groups]))
        return out,fits
    rng=np.random.default_rng(98)
    halves=[rng.permutation([i for i in train if genders[names[i]]==g]) for g in ('male','female')]
    cvs=[]
    for f in range(3):
        hold=np.concatenate([group[f::3] for group in halves])
        cvs.append(evaluate(np.setdiff1d(train,hold),hold)[0])
    scores={k:float(np.mean([c[k] for c in cvs])) for k in cvs[0]}
    choice=max(scores,key=scores.get)
    d,w=evaluate(train,dev)
    report=dict(train_cv_by_fold=cvs,cv_mean=scores,choice=choice,dev_selected=d[choice],dev_acoustic=d['acoustic'],dev_exploratory=d,selected_weights=w.get(choice),train_texts=[1,2,3,4],dev_cross_text_pairs='5..8 versus 9..12',dev_n=20)
    write_json(ROOT/'results/layer_metric_dev.json',report)
    print(choice,d[choice],d['acoustic'])


if __name__=='__main__':
    run()
