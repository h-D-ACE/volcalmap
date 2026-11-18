"""Evaluate frozen candidates on simultaneous natural English two-microphone speech."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.stats import spearmanr
from scipy.signal import resample_poly
from voxsim_reference import load_published,factors
from wave_student import WaveMap
from paired_correction import Correction
from extract_features import Frontend
from runtime import ROOT,write_json,file_hash
from dataset import unit
from infer import encode


def assess(keys,mic1,mic2):
    d=np.linalg.norm(mic1[:,None]-mic2[None,:],axis=-1)
    same=[k.split('_')[0] for k in keys]
    own=np.equal.outer(same,same)
    d1=np.linalg.norm(mic1[:,None]-mic1[None,:],axis=-1)
    d2=np.linalg.norm(mic2[:,None]-mic2[None,:],axis=-1)
    pairs=np.triu_indices(len(keys),1)
    corpus={}
    for i,s in enumerate(sorted(set(same))):
        idx=np.where(np.asarray(same)==s)[0]
        corpus[s]=float(np.median(np.diag(d)[idx]))
    n1=np.argsort(np.where(own,np.inf,d1),axis=-1)
    n2=np.argsort(np.where(own,np.inf,d2),axis=-1)
    return dict(paired_l2=float(np.median(np.diag(d))),per_voice=corpus,
        same_recording_top1=float(np.mean(d.argmin(1)==np.arange(len(keys)))),
        geometry_srcc=float(spearmanr(d1[pairs],d2[pairs]).statistic),
        cross_voice_top5_overlap=float(np.mean([len(set(a[:5]) & set(b[:5]))/5 for a,b in zip(n1,n2)])))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp'))
    args=p.parse_args()
    parent=args.root;ext,sha=load_published(parent/'voxsim_reference','large')
    import soxr
    wave=WaveMap().cuda().eval()
    wave.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    run=parent/'edge-run/vm_perceptual/runs'
    edge=torch.load(run/'selected.pt',weights_only=True,map_location='cpu')
    cor=torch.load(run/'noise_correction_dev.pt',weights_only=True,map_location='cpu')
    assert cor['base_hash']==edge['data_hash'] and edge['edge_policy']=='trim_zero'
    correct=Correction(cor['width']).eval();correct.load_state_dict(cor['model'])
    frontend=Frontend(device='cuda',use_ecapa=True,use_wavlm=False)
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('edge_frontend',run.parent/'extract_features.py')
    edge_module=module_from_spec(spec);spec.loader.exec_module(edge_module)
    canon=edge_module.Frontend(device='cuda',use_ecapa=False,use_wavlm=False,edge_policy='trim_zero')
    report={}
    for split in ('validation','confirmation'):
        dir=parent/'vm_acoustic'/f'data_vctk_{split}'
        manifest=json.loads((dir/'manifest.json').read_text())
        rows={}
        for r in manifest['files']:
            path=dir/r['file']
            if path.suffix!='.flac':continue
            assert file_hash(path)==r['sha256']
            key,mic=path.stem.rsplit('_',1)
            x,sr=sf.read(path,dtype='float32')
            feature=frontend.extract(path);edgefeat=canon.extract(path)
            edgez=encode(edgefeat,edge)
            inp=np.array(edgefeat['acoustic']+edgefeat['prosody_descriptors'])
            inp=torch.tensor(np.clip((inp-cor['mean'])/cor['scale'],-10,10),dtype=torch.float32)
            newz=correct(inp[None],torch.tensor(edgez)[None])[0].numpy()
            g=math.gcd(16000,sr)
            wav=torch.from_numpy(resample_poly(x,16000//g,sr//g).astype('float32')).cuda()[None]
            length=torch.tensor([wav.shape[1]],device='cuda')
            zs=dict(current=encode(feature,ck),student=wave(wav,length).cpu().numpy()[0],
                identity=unit(np.array(feature['ecapa'])),voxsim_spherical=unit(factors(ext,soxr.resample(x,sr,16000,quality='HQ'))),
                edge=edgez,edge_corrected=newz)
            rows.setdefault(key,{})[mic]=zs
        keys=sorted(k for k,v in rows.items() if set(v)=={'mic1','mic2'})
        report[split]={'pairs':len(keys),'models':{}}
        for k in rows[keys[0]]['mic1']:
            m=[np.stack([rows[n][mi][k] for n in keys]) for mi in ('mic1','mic2')]
            report[split]['models'][k]=assess(keys,*m)
        print(split,len(keys),{k:v['paired_l2'] for k,v in report[split]['models'].items()},flush=True)
    write_json(ROOT/'results/recording_transfer.json',report)


if __name__=='__main__':main()
