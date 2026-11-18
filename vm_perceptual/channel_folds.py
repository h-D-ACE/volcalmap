"""Paired device augmentation under held-out voices and texts with single-file inference."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F
from channel_models import apply_channel
from dataset import load_rating_graph
from nested_ssl_trial import make_teacher, rate
from perceptual_rank import bank, transform, train_triples
from recording_transfer import assess
from reference_student import read
from runtime import ROOT, file_hash, write_json
from ssl_rank_student import SSLMap
from wave_folds import paired_labels
from wave_student import batch


def microphones(root):
    pairs={}
    for record in json.loads((root/'manifest.json').read_text())['files']:
        path=root/record['file']
        if path.suffix!='.flac':continue
        if file_hash(path)!=record['sha256']:raise ValueError(path)
        name,mic=path.stem.rsplit('_',1)
        wave,sr=sf.read(path,dtype='float32')
        f=math.gcd(sr,16000)
        pairs[name,mic]=resample_poly(wave,16000//f,sr//f).astype('float32')
    names=sorted({k[0] for k in pairs})
    return names,[[pairs[s,m] for s in names] for m in ('mic1','mic2')]


@torch.inference_mode()
def single(model,wave):
    x=torch.from_numpy(wave).cuda()[None]
    return model(x,torch.tensor([x.shape[-1]],device='cuda'))[0].cpu().numpy()


@torch.inference_mode()
def rating(model,waves,people,slots,labels):
    z=np.stack([[single(model,waves[p*12+s]['wave']) for s in slots] for p in people])
    return rate(z,labels)


def train(waves,tr,va,te,teacher,labels,seed,method,steps,mics):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=SSLMap('frozen').cuda()
    learned=[w for w in model.parameters() if w.requires_grad]
    opt=torch.optim.AdamW(learned,lr=.0005,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps,eta_min=.00003)
    target=torch.from_numpy(teacher).cuda()
    best=-np.inf;history=[]
    for step in range(1,steps+1):
        who=rng.choice(tr,16,replace=False)
        slots=rng.choice([0,1,2,3,6,7],16)
        x,n=batch(waves,who*12+slots,float(rng.choice([1,2,3,4,5])),rng)
        kind=str(rng.choice(['clean','equalizer','room','both']))
        y=target[who,slots]
        noisy=apply_channel(x,n,kind,step+seed*100000) if method=='consistency' else x
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(noisy,n)
        loss=(z-y).square().sum(-1).mean()+F.mse_loss(z@z.T,y@y.T)
        if method=='consistency' and kind!='clean':
            with torch.autocast('cuda',dtype=torch.bfloat16):anchor=model(x,n)
            loss+=.5*(z-anchor.detach()).square().sum(-1).mean()+.5*(anchor-y).square().sum(-1).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learned,3);opt.step();sched.step()
        if step%300==0 or step==steps:
            score=rating(model,waves,va,[4,5],labels[0])
            history.append([step,score])
            if score>best:
                best=score;chosen=step;state=[w.detach().clone() for w in learned]
    with torch.no_grad():
        for w,v in zip(learned,state):w.copy_(v)
    names,groups=mics
    mic_vectors=[np.stack([single(model,w) for w in group]) for group in groups]
    return dict(step=chosen,validation=history,outer_srcc=rating(model,waves,te,[8,9,10,11],labels[1]),
                english_microphones=assess(names,*mic_vectors))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repeat',type=int,required=True);ap.add_argument('--steps',type=int,default=900)
    ap.add_argument('--mic-data',type=Path,default=Path('/root/autodl-tmp/vm_acoustic/data_vctk_confirmation'))
    a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=load_rating_graph(('train','dev'));names,blocks=bank(data,include_reference=False)
    people,waves,_=read(data,include_reference=False)
    if people!=names:raise ValueError('Audio and features must have the same speaker order')
    mics=microphones(a.mic_data)
    rng=np.random.default_rng(9000+a.repeat)
    grouped=[rng.permutation([i for i,s in enumerate(names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    folds=[np.concatenate([g[j::5] for g in grouped]) for j in range(5)]
    folder=ROOT/'runs'/f'channel_folds_{a.repeat}_{a.steps}';folder.mkdir(exist_ok=True)
    for fold in range(5):
        te,va=folds[fold],folds[(fold+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[te,va])
        labels=(paired_labels(data,names,va),paired_labels(data,names,te))
        x=np.concatenate(transform(blocks,tr),axis=-1)/np.sqrt(3)
        y,teacher_step=make_teacher(x,tr,va,labels[0],train_triples(tr,names,data),881+fold)
        for seed in (76,431):
            for method in ('unaugmented','consistency'):
                path=folder/f'{fold}_{seed}_{method}.json'
                if path.exists():continue
                t=time.monotonic()
                row=train(waves,tr,va,te,y,labels,seed,method,a.steps,mics)
                row.update(repeat=a.repeat,fold=fold,seed=seed,recipe=method,teacher_step=teacher_step,
                           elapsed=time.monotonic()-t)
                temp=path.with_suffix('.partial');write_json(temp,row);temp.replace(path)
                print(a.repeat,fold,seed,method,round(row['outer_srcc'],3),round(row['elapsed']),flush=True)


if __name__=='__main__':main()
