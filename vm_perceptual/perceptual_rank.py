"""Speaker-disjoint anchored ordinal training for a compact perceptual timbre map.

All four reference sentences contribute to a rated voice profile. No identity
classification targets. New sentences and people are held out at each fold.
"""
import argparse
import itertools
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import rankdata
from dataset import load_data,unit
from runtime import ROOT,write_json
from wave_folds import paired_labels


def bank(data, include_reference=True):
    with np.load(ROOT/'features/development_layers.npz') as f:
        names=f['speakers'].tolist()
        blocks=[f['acoustic'].astype('float32'),f['ecapa'],f['layers'][:,:,1]]
    if not include_reference:
        return names, blocks
    reference=[]
    for s in names:
        with np.load(ROOT/'features/voxsim_large'/f'{s}.npz') as f:
            normal=sorted((p,v) for p,v in zip(f['names'],f['vectors']) if '/parallel100/' in str(p))
            assert len(normal)==12
            reference.append([v for p,v in normal])
    return names,[*blocks,np.asarray(reference)]


def transform(blocks,tr):
    vectors=[]
    for i,block in enumerate(blocks):
        x=block if i==0 else unit(block)
        train=x[tr,:4].reshape(-1,x.shape[-1])
        mu=train.mean(0)
        scale=np.maximum(train.std(0),.05) if i==0 else 1.
        x=unit((x-mu)/scale)
        vectors.append(x.astype('float32'))
    return vectors


def metrics(z,labels):
    ranks=[]; anchored=[]
    ids=list(itertools.permutations(range(4),2))
    for a,b,y in labels:
        a=np.asarray(a);b=np.asarray(b);y=np.asarray(y)
        y_rank=rankdata(y);y_rank-=y_rank.mean()
        scores=np.stack([(z[a,i]*z[b,j]).sum(-1) for i,j in ids])
        sr=rankdata(scores,axis=1);sr-=sr.mean(1,keepdims=True)
        ranks.append(float(np.mean(sr@y_rank/(np.linalg.norm(sr,axis=1)*np.linalg.norm(y_rank)))))
        # Compare two rated pairs sharing the anchor; average orientations to respect symmetry.
        mean_scores=(scores+scores[[ids.index((j,i)) for i,j in ids]])/2
        pa,pb=np.triu_indices(len(y),1)
        keep=((a[pa]==a[pb])|(a[pa]==b[pb])|(b[pa]==a[pb])|(b[pa]==b[pb])) & (abs(y[pa]-y[pb])>=1.)
        pa,pb=pa[keep],pb[keep]
        d=(mean_scores[:,pa]-mean_scores[:,pb])*np.sign(y[pa]-y[pb])
        anchored.append(float(np.mean((d>0)+.5*(d==0))))
    return dict(single_srcc=float(np.mean(ranks)),anchor_agreement=float(np.mean(anchored)))


def train_triples(tr,names,data):
    inverse={names[k]:i for i,k in enumerate(tr)}
    m=np.full((len(tr),len(tr)),np.nan)
    for r in data['pairs']:
        if r['speaker_a'] in inverse and r['speaker_b'] in inverse:
            i=inverse[r['speaker_a']];j=inverse[r['speaker_b']];m[i,j]=m[j,i]=r['score']
    aa=[];bb=[];cc=[]
    for a,row in enumerate(m):
        b,c=np.where(row[:,None]-row[None,:]>=1.)
        aa.extend([a]*len(b));bb.extend(b);cc.extend(c)
    return torch.tensor(np.stack((aa,bb,cc)),device='cuda')


def fit(x,tr,va,te,labels,triples,seed,recipe):
    torch.manual_seed(seed)
    # Fixed equal-block Euclidean coordinates; train PCA only on the training voices' four profiles.
    flat=x[tr,:4].reshape(-1,x.shape[-1])
    mu=flat.mean(0)
    _,s,vt=np.linalg.svd((flat-mu).astype('float64'),full_matrices=False)
    x=((x-mu)@vt[:64].T).astype('float32')
    t=torch.from_numpy(x).cuda()
    head=nn.Linear(x.shape[-1],16,bias=False,device='cuda')
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:16,:16]=torch.eye(16,device='cuda')
    initial=head.weight.detach().clone()
    optim=torch.optim.AdamW(head.parameters(),lr=.002)
    dev_labels,test_labels=labels
    best=-np.inf
    path=[]
    for step in range(0,1001):
        if step%100==0:
            with torch.no_grad():
                z=F.normalize(head(t[va,4:8]),dim=-1).cpu().numpy()
                score=metrics(z,dev_labels)['single_srcc']
                if score>best:
                    best=score;chosen=step;state=head.weight.detach().clone()
                path.append([step,score])
        if step==1000:break
        head.train()
        z=F.normalize(head(t[tr,:4]),dim=-1)
        # For the profile loss each side is an average of four recorded sentences.
        if recipe=='profile':
            p=z.mean(1);a,b,c=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
            delta=(p[a]*(p[b]-p[c])).sum(-1)
        else:
            a,b,c=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
            j=torch.randint(4,(3,len(a)),device='cuda')
            delta=(z[a,j[0]]*(z[b,j[1]]-z[c,j[2]])).sum(-1)
        loss=F.softplus(-delta*6).mean()+.02*(head.weight-initial).square().sum()/16
        optim.zero_grad();loss.backward();optim.step()
    with torch.no_grad():
        head.weight.copy_(state)
        z=F.normalize(head(t[te,8:12]),dim=-1).cpu().numpy()
    return dict(best_step=chosen,dev_history=path,outer=metrics(z,test_labels))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args();data=load_data();names,blocks=bank(data)
    rng=np.random.default_rng(9000+args.repeat)
    partitions=[rng.permutation([i for i,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([p[i::5] for p in partitions]) for i in range(5)]
    folder=ROOT/'runs'/f'perceptual_rank_{args.repeat}';folder.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        dest=folder/f'fold_{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        labels=(paired_labels(data,names,va),paired_labels(data,names,te))
        triples=train_triples(tr,names,data);v=transform(blocks,tr)
        row={'fixed':{},'trained':{},'tr_people':len(tr),'te_people':len(te)}
        for key,z in zip(('acoustic','identity','wavlm_layer1','voxsim'),v):
            row['fixed'][key]=metrics(z[te,8:12],labels[1])
        row['fixed']['voxsim_author_dot']=metrics(blocks[-1][te,8:12],labels[1])
        for name,indices in [('own',(0,1,2)),('own_and_reference',(0,1,2,3))]:
            x=np.concatenate([v[i] for i in indices],axis=-1)/np.sqrt(len(indices))
            row['fixed'][name]=metrics(x[te,8:12],labels[1])
            for recipe in ('profile','single'):
                row['trained'][name+'_'+recipe]=fit(x,tr,va,te,labels,triples,881+fold,recipe)
        write_json(dest,row);print(args.repeat,fold,row['fixed']['voxsim_author_dot'],flush=True)


if __name__=='__main__':
    main()
