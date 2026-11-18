"""Synthetic same-sentence morphology diagnostic; no transformed audio is retained.

A source-timed WORLD parameter interpolation is a controlled intervention. A
coefficient of one is a target-conditioned resynthesis, not a real impression.
"""
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.fft import dct
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from dataset import load_data,unit
from runtime import ROOT,write_json
from rank_map_candidate import Frontends,encode as rank_encode
from infer import encode as base_encode
from wave_map_cli import WaveEncoder


def align(a,b):
    # Compare low-order cepstra after removing each utterance's mean vocal color.
    def content(sp):
        c=dct(np.log(np.maximum(sp[::4],1e-10)),type=2,norm='ortho',axis=1)[:,1:18]
        return unit((c-c.mean(0))/np.maximum(c.std(0),.01))
    lo,hi=content(a),content(b);cost=cdist(lo,hi,'sqeuclidean')
    n,m=cost.shape;dp=np.full((n+1,m+1),np.inf);dp[0,0]=0
    ptr=np.zeros((n,m),dtype='uint8')
    for i in range(n):
        for j in range(m):
            ways=(dp[i,j],dp[i,j+1]+.1,dp[i+1,j]+.1)
            k=np.argmin(ways);dp[i+1,j+1]=cost[i,j]+ways[k];ptr[i,j]=k
    i,j=n-1,m-1;path=[[] for _ in range(n)]
    while i>=0 and j>=0:
        path[i].append(j)
        k=ptr[i,j]
        if k in (0,1):i-=1
        if k in (0,2):j-=1
    ref=[4*np.mean(v) for v in path]
    return np.clip(np.interp(np.arange(len(a)),4*np.arange(n),ref).round().astype(int),0,len(b)-1)


def world_params(path,pw):
    x,sr=sf.read(path,dtype='float64')
    f,t=pw.dio(x,sr,f0_floor=60,f0_ceil=750);f=pw.stonemask(x,f,t,sr)
    sp=pw.cheaptrick(x,f,t,sr);ap=pw.d4c(x,f,t,sr)
    return x,sr,f,sp,ap


@torch.inference_mode()
def main():
    sys.path.insert(0,'/root/autodl-tmp/voxsim_reference/dependencies')
    import pyworld as pw
    data=load_data();front=Frontends(False)
    ck=torch.load(ROOT/'runs/selected.pt',map_location='cpu',weights_only=True)
    par=np.load(ROOT/'runs/rank_map_own.npz')
    student=WaveEncoder(ROOT/'runs/wave_431_noise/best.pt','cuda')
    cache=ROOT/'features/morph_paths';cache.mkdir(exist_ok=True)
    levels=np.linspace(0,1,5)
    rng=np.random.default_rng(30014)
    plans=[]
    for gender in ('female','male'):
        names=[s for s in data['speakers'] if data['manifest']['gender'][s]==gender and data['manifest']['split'][s]=='dev']
        ordered=rng.permutation(names)
        for a,b in zip(ordered,np.roll(ordered,1)):
            for k in range(4):plans.append((a,b,k))
    with tempfile.TemporaryDirectory() as tmp:
        wave=Path(tmp)/'observation.wav'
        for a,b,k in plans:
            dest=cache/f'{a}_{b}_{k}.npz'
            if dest.exists():continue
            name=Path(data['clips'][a][k]).name
            pa=ROOT/'data_jvs'/data['clips'][a][k];pb=ROOT/'data_jvs'/b/'parallel100/wav24kHz16bit'/name
            xa,sr,fa,sa,aa=world_params(pa,pw)
            xb,sb_rate,fb,sb,ab=world_params(pb,pw);assert sr==sb_rate
            ix=align(sa,sb);sa=np.log(sa.clip(1e-10));sb=np.log(sb[ix].clip(1e-10));ab=ab[ix]
            voiced=fa>0
            vb=np.flatnonzero(fb>0)
            if min(len(vb),voiced.sum())<3:raise ValueError('Insufficient voiced support')
            target_logf=np.interp(ix,vb,np.log(fb[vb]))
            out={m:[] for m in ('current','student','own','identity')}
            sources=[pa,pb]
            for level in levels:
                f=fa.copy();f[voiced]=np.exp((1-level)*np.log(fa[voiced])+level*target_logf[voiced])
                y=pw.synthesize(f,np.exp((1-level)*sa+level*sb),np.clip((1-level)*aa+level*ab,0,1),sr)
                y*=np.sqrt(np.mean(xa*xa))/max(np.sqrt(np.mean(y*y)),1e-10)
                file=wave.with_name(f'm{level:.2f}.wav');sf.write(file,y,sr,subtype='FLOAT');sources.append(file)
            for file in sources:
                raw=front.extract(file)
                out['current'].append(base_encode({'acoustic':raw[0]},ck));out['student'].append(student.vector(file))
                out['own'].append(rank_encode(raw,par));out['identity'].append(unit(raw[1]))
            stage=dest.with_suffix('.partial.npz')
            np.savez_compressed(stage,**out);stage.replace(dest);print(a,b,k,flush=True)
    result={}
    for key in ('current','student','own','identity'):
        natural=[];synthetic=[];cross=[];end=[]
        for a,b,k in plans:
            z=np.load(cache/f'{a}_{b}_{k}.npz')[key]
            own,target=z[0],z[1];path=z[2:]
            margin=np.linalg.norm(path-own,axis=1)-np.linalg.norm(path-target,axis=1)
            natural.append(spearmanr(levels,margin).statistic)
            m=np.linalg.norm(path-path[0],axis=1)-np.linalg.norm(path-path[-1],axis=1)
            synthetic.append(spearmanr(levels,m).statistic)
            end.append(float(margin[-1]>margin[0]))
            # Other-sentence target comparison avoids using the matching endpoint itself.
            za=np.mean([np.load(cache/f'{a}_{b}_{j}.npz')[key][0] for j in range(4) if j!=k],axis=0)
            zb=np.mean([np.load(cache/f'{a}_{b}_{j}.npz')[key][1] for j in range(4) if j!=k],axis=0)
            m=np.linalg.norm(path-za,axis=1)-np.linalg.norm(path-zb,axis=1)
            cross.append(spearmanr(levels,m).statistic)
        result[key]=dict(mean_rho_natural_anchors=float(np.mean(natural)),mean_rho_synthetic_anchors=float(np.mean(synthetic)),
            mean_rho_other_sentence_anchors=float(np.mean(cross)),fraction_moving_toward_target=float(np.mean(end)))
    write_json(ROOT/'results/morph_paths.json',dict(paths=len(plans),ordered_voice_pairs=len(set((a,b) for a,b,k in plans)),
        steps=levels.tolist(),models=result,scope='Development signal intervention. No human imitation or listening labels.'))
    print(result,flush=True)


if __name__=='__main__':main()
