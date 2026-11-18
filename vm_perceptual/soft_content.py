"""Train-fold-only content conditioning of audio frames with complete sentence coverage."""
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from dataset import load_data, unit
from perceptual_rank import metrics
from wave_folds import paired_labels
from runtime import ROOT,write_json


def collect(names):
    x=[];h=[];clip=[];length=[]
    for i,s in enumerate(names):
        with np.load(ROOT/'features/content_frames'/f'{s}.npz') as f:
            if len(f['lengths'])!=12:raise ValueError('Expected 12 sentences')
            lo=np.split(f['shallow'],np.cumsum(f['lengths'])[:-1])
            hi=np.split(f['deep'],np.cumsum(f['lengths'])[:-1])
            for j,(l,r) in enumerate(zip(lo,hi)):
                r=r.astype('float32')
                # Utterance-specific centering for the unit assignment only.
                h.append(unit(r-r.mean(0)));x.append(l.astype('float32'))
                clip.append(np.full(len(l),12*i+j));length.append(len(l))
    return tuple(torch.as_tensor(np.concatenate(arr),device='cuda') for arr in (x,h,clip)),torch.tensor(length,device='cuda')


def clusters(high,k,seed):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    ix=torch.randperm(len(high),generator=gen,device='cuda')[:min(32768,len(high))]
    sample=high[ix]
    centers=sample[torch.randperm(len(sample),generator=gen,device='cuda')[:k]].clone()
    for _ in range(18):
        labels=(sample@centers.T).argmax(1)
        new=torch.zeros_like(centers).index_add_(0,labels,sample)
        valid=new.norm(dim=1)>1e-8
        centers=torch.where(valid[:,None],F.normalize(new,dim=1),centers)
    return centers


def structure(z,labels):
    out=metrics(z,labels)
    n,t,d=z.shape
    pure=z.mean(1)
    dist=np.linalg.norm(z-pure[:,None],axis=-1)
    others=np.linalg.norm(pure[:,None]-pure[None,:],axis=-1)[np.triu_indices(n,1)]
    out['within_over_between']=float(np.median(dist)/np.median(others))
    top=[]
    for i in range(n):
        # Diagnose residual sentence cues by predicting the shared sentence index across other people.
        templates=(z.sum(0)-z[i])/(n-1)
        top.extend((z[i]@unit(templates).T).argmax(1)==np.arange(t))
    out['sentence_id_top1']=float(np.mean(top))
    return out


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=int,default=0)
    a=ap.parse_args();data=load_data();names=[s for s in data['speakers'] if data['manifest']['split'][s]!='test']
    (raw,high,clip),length=collect(names);num=len(length)
    rng=np.random.default_rng(9000+a.repeat)
    parts=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    folds=[np.concatenate([group[j::5] for group in parts]) for j in range(5)]
    folder=ROOT/'runs'/f'soft_content_{a.repeat}';folder.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    def aggregate(values):
        return values.new_zeros(num,values.shape[-1]).index_add_(0,clip,values)
    for fold in range(5):
        dest=folder/f'{fold}.json'
        if dest.exists():continue
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        trainclips=(tr[:,None]*12+np.arange(4)).ravel()
        trainmask=torch.isin(clip,torch.tensor(trainclips,device='cuda'))
        mu=raw[trainmask].mean(0);sd=raw[trainmask].std(0).clamp_min(.001)
        x=(raw-mu)/sd;report={}
        labels=[paired_labels(data,names,people) for people in (va,te)]
        def eval_candidate(key,residual,weights):
            den=aggregate(weights[:,None])
            mean=aggregate(weights[:,None]*residual)/den
            std=(aggregate(weights[:,None]*residual.square())/den-mean.square()).clamp_min(1e-8).sqrt()
            feat=torch.cat((mean,std),dim=1)
            train=feat[trainclips];feat=F.normalize((feat-train.mean(0))/train.std(0).clamp_min(.05),dim=1)
            z=feat.cpu().numpy().reshape(len(names),12,-1)
            effective=den[:,0].square()/aggregate(weights[:,None].square())[:,0]/length
            report[key]=dict(dev=structure(z[va,4:8],labels[0]),outer=structure(z[te,8:12],labels[1]),
                outer_effective_frame_fraction=float(effective.reshape(len(names),12)[te,8:12].median()))
        ones=torch.ones(len(clip),device='cuda')
        eval_candidate('unconditional',x,ones)
        for k in (8,32,64):
            c=clusters(high[trainmask],k,308+fold)
            p=(high@c.T/.085).softmax(1)
            design=p[trainmask]
            beta=torch.linalg.solve(design.T@design+torch.eye(k,device='cuda')*2,design.T@x[trainmask])
            expected=p@beta
            occ=aggregate(p)/length[:,None];prior=design.mean(0)
            adjustment=(p*(prior/(.25*prior+.75*occ[clip]).clamp_min(1e-6))).sum(1)
            for fraction in (.5,1.):
                residual=x-fraction*expected
                eval_candidate(f'k{k}_subtract{fraction}',residual,ones)
                eval_candidate(f'k{k}_subtract{fraction}_balance',residual,adjustment)
        choice=max(report,key=lambda k:report[k]['dev']['single_srcc'])
        write_json(dest,dict(selected=choice,candidates=report))
        print(a.repeat,fold,choice,report[choice]['outer'],flush=True)


if __name__=='__main__':main()
