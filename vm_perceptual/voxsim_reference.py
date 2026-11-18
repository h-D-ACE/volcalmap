"""Cache official VoxSim pair-score factors on locally available research recordings.

Install the authors' inference module and checkpoint from
https://huggingface.co/spaces/junseok520/VoxSIM and junseok520/voxsim-models.
The per-file dot product exactly factorizes their two-term multi-crop score.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from runtime import ROOT
from dataset import load_data


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def windows(wave):
    n=400*160+240
    padded=np.pad(wave,(0,n-len(wave)+1),mode='wrap') if len(wave)<=n else wave
    starts=np.linspace(0,len(padded)-n,10).astype(int)
    return np.stack([padded[i:i+n] for i in starts])


def load_published(reference,model_name):
    ref=Path(reference).resolve()
    original=Path.cwd();os.chdir(ref);sys.path[:0]=[str(ref),str(ref/'dependencies')]
    expected={'small':('ecapa_tdnn.model','e4f4b0af9cd9a9a1606261d4d3dcee8101d1af55df52531566579d451c06482e'),
              'large':('voxsim_wavlm_ecapa.model','97e790704a0a4cb40035de289b02e3f40d26400d17619ea8af200141e35583df')}
    filename,sha=expected[model_name]
    assert digest(ref/filename)==sha
    state=torch.load(ref/filename,weights_only=True,map_location='cpu')
    if model_name=='small':
        # The released small checkpoint ends at fc6; no final BN parameters. This is a reconstructed baseline.
        from ECAPA import ECAPA_TDNN
        model=ECAPA_TDNN(C=1024,emb_dim=256)
        if not any(k.startswith('__S__.bn6.') for k in state):
            model.bn6=torch.nn.Identity()
        state={k.removeprefix('__S__.'):v for k,v in state.items() if k.startswith('__S__.')}
    else:
        from ssl_ecapa_model import SSL_ECAPA_TDNN
        model=SSL_ECAPA_TDNN(feat_dim=1024,emb_dim=256,feat_type='wavlm_large')
    model.load_state_dict(state,strict=True);del state
    model.eval().cuda()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.chdir(original)
    return model,sha


@torch.inference_mode()
def factors(model,wave):
    full=F.normalize(model(torch.from_numpy(wave[None]).cuda()),dim=-1)
    crops=F.normalize(model(torch.from_numpy(windows(wave)).cuda()),dim=-1)
    return (torch.cat((full[0],crops.mean(0)))/np.sqrt(2.)).cpu().numpy()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',default='/root/autodl-tmp/voxsim_reference')
    p.add_argument('--model',choices=('small','large'),required=True)
    a=p.parse_args()
    model,sha=load_published(a.reference,a.model)
    import soundfile as sf
    import soxr
    data=load_data()
    folder=ROOT/'features'/f'voxsim_{a.model}';folder.mkdir(exist_ok=True)
    for speaker in data['speakers']:
        out=folder/f'{speaker}.npz'
        root=ROOT/'data_jvs'/speaker
        files=sorted([p for style in ('parallel100','falset10') for p in (root/style/'wav24kHz16bit').glob('*.wav')])
        names=[f.relative_to(ROOT/'data_jvs').as_posix() for f in files]
        if out.exists():
            with np.load(out) as f:
                if f['names'].tolist()==names and str(f['weight_sha256'])==sha:continue
        vectors=[]
        for path in files:
            audio,sr=sf.read(path,dtype='float32')
            wave=soxr.resample(audio,sr,16000,quality='HQ') if sr!=16000 else audio
            vectors.append(factors(model,wave))
        tmp=out.with_suffix('.partial.npz')
        np.savez_compressed(tmp,names=names,vectors=vectors,weight_sha256=sha)
        tmp.replace(out)
        print(speaker,len(vectors),flush=True)


if __name__=='__main__':
    main()
