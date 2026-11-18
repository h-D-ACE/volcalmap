"""Controlled WORLD source/filter perturbations. A vocoder intervention is a proxy only."""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from dataset import load_data,unit
from runtime import ROOT,write_json
from infer import encode
from extract_features import Frontend
from wave_student import WaveMap


def warp(envelope,ratio):
    # Compress/expand the frequency grid of the spectral envelope, preserving the excitation.
    freq=np.arange(envelope.shape[1])
    return np.stack([np.interp(freq/ratio,freq,row) for row in envelope])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',default='/root/autodl-tmp/voxsim_reference')
    p.add_argument('--split',choices=('train','dev','test'),default='dev')
    args=p.parse_args()
    sys.path.insert(0,str(Path(args.reference)/'dependencies'))
    import pyworld as world
    data=load_data();front=Frontend(device='cuda',use_ecapa=True,use_wavlm=False)
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    model=WaveMap().cuda().eval();model.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    people=[s for s in data['speakers'] if data['manifest']['split'][s]==args.split]
    out=ROOT/'features'/f'source_filter_{args.split}';out.mkdir(exist_ok=True)
    treatments={'original':(0,1.,1.),'resynthesis':(0,1.,1.),'pitch_up':(2,1.,1.),
        'pitch_down':(-2,1.,1.),'tract_up':(0,1.08,1.),'tract_down':(0,1/1.08,1.),
        'flatter_contour':(0,1.,.6),'wider_contour':(0,1.,1.4)}
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'measurement.wav'
        for s in people:
            dest=out/f'{s}.npz'
            if dest.exists():continue
            arrays={k:[] for k in ('map','student','ecapa','logf0')}
            for relative in data['clips'][s]:
                w,sr=sf.read(ROOT/'data_jvs'/relative,dtype='float64')
                rms=np.sqrt(np.mean(w*w))
                initial,t=world.dio(w,sr,f0_floor=60,f0_ceil=750)
                f=world.stonemask(w,initial,t,sr);voiced=f>0
                sp=world.cheaptrick(w,f,t,sr);ap=world.d4c(w,f,t,sr)
                median=np.median(np.log(f[voiced]))
                for key,(semitones,scale,stretch) in treatments.items():
                    if key=='original':y=w
                    else:
                        f_new=f.copy()
                        f_new[voiced]=np.exp(median+stretch*(np.log(f[voiced])-median))*2**(semitones/12)
                        y=world.synthesize(f_new,sp if scale==1 else warp(sp,scale),ap,sr)[:len(w)]
                        y=y*rms/max(np.sqrt(np.mean(y*y)),1e-12)
                    sf.write(path,y,sr,subtype='FLOAT')
                    feat=front.extract(path)
                    arrays['map'].append(encode(feat,ck))
                    arrays['ecapa'].append(unit(np.asarray(feat['ecapa'])))
                    g=math.gcd(sr,16000)
                    x=resample_poly(y,16000//g,sr//g).astype('float32')
                    x=torch.tensor(x,device='cuda')[None]
                    arrays['student'].append(model(x,torch.tensor([x.shape[1]],device='cuda')).cpu().numpy()[0])
                    arrays['logf0'].append(feat['acoustic'][-2])
            stage=dest.with_suffix('.partial.npz')
            np.savez_compressed(stage,conditions=list(treatments),**{k:np.asarray(v).reshape(4,len(treatments),-1) for k,v in arrays.items()})
            stage.replace(dest);print(s,flush=True)
    files=[out/f'{s}.npz' for s in people]
    summary={'voices':len(people),'sentences_per_voice':4,'conditions':list(treatments),'models':{}}
    ids=np.repeat(np.arange(len(people)),4);exclude=ids[:,None]==ids[None,:]
    for key in ('map','student','ecapa','logf0'):
        z=np.concatenate([np.load(file)[key] for file in files],axis=0)
        measures={}
        for j,condition in enumerate(treatments):
            baseline=z[:,0 if j==0 or j==1 else 1];changed=z[:,j]
            drift=np.linalg.norm(changed-baseline,axis=1)
            row=dict(median_within_clip_l2=float(np.median(drift)))
            if key=='logf0':
                row['median_signed_semitones']=float(np.median(changed-baseline)*12/np.log(2))
            else:
                da=np.linalg.norm(baseline[:,None]-baseline[None,:],axis=-1)
                db=np.linalg.norm(changed[:,None]-baseline[None,:],axis=-1)
                ia=np.argsort(np.where(exclude,np.inf,da),axis=1)
                ib=np.argsort(np.where(exclude,np.inf,db),axis=1)
                row['same_cross_person_top1']=float(np.mean(ia[:,0]==ib[:,0]))
                row['cross_person_top5_overlap']=float(np.mean([len(set(a[:5])&set(b[:5]))/5 for a,b in zip(ia,ib)]))
            measures[condition]=row
        summary['models'][key]=measures
    write_json(ROOT/'results'/f'source_filter_{args.split}.json',summary)


if __name__=='__main__':
    main()
