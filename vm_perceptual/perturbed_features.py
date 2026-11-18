"""Generate aligned clean and perturbed acoustic features on the GPU training host."""
import argparse,json,tempfile
from pathlib import Path
from runtime import ROOT,write_json
from extract_features import Frontend
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import hashlib


def perturb(x,sr,kind,rng):
    if kind=='clean':return x,sr
    if kind.startswith('noise_'):
        db=int(kind.split('_')[-1])
        noise=rng.standard_normal(len(x))
        noise*=np.sqrt(np.mean(x*x)/np.mean(noise*noise))*10**(-db/20)
        return (x+noise).astype('float32'),sr
    if kind=='band_8k':return resample_poly(x,1,3),8000
    if kind=='band_12k':return resample_poly(x,1,2),12000
    if kind=='tilt':
        return np.r_[x[0],x[1:]-.4*x[:-1]].astype('float32'),sr
    if kind=='echo':
        lag=int(sr*.035)
        y=x.copy();y[lag:]+=.25*x[:-lag]
        return y,sr
    raise ValueError(kind)


def main():
    m=json.loads((ROOT/'data_jvs/manifest.json').read_text())
    front=Frontend(device='cuda',use_ecapa=False,use_wavlm=False,edge_policy='trim_zero')
    modes=['clean','noise_10','noise_20','noise_30','band_8k','band_12k','tilt','echo']
    outdir=ROOT/'features/perturbations';outdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'clip.wav'
        for speaker,split in sorted(m['split'].items()):
            dest=outdir/(speaker+'.npz')
            if dest.exists():continue
            paths=sorted((ROOT/'data_jvs'/speaker/'parallel100/wav24kHz16bit').glob('*.wav'))[:(12 if split=='train' else 4)]
            all_features=[];lengths=[]
            for p in paths:
                x,sr=sf.read(p,dtype='float32')
                if sr!=24000:raise ValueError(p)
                rows=[]
                for mode in modes:
                    seed=int.from_bytes(hashlib.sha256((speaker+p.name+mode).encode()).digest()[:8],'little')
                    y,rate=perturb(x,sr,mode,np.random.default_rng(seed))
                    sf.write(f,y,rate,subtype='FLOAT')
                    try:
                        feat=front.extract(f)
                        rows.append(feat['acoustic']+feat['prosody_descriptors'])
                    except ValueError:
                        rows.append([float('nan')]*44)
                all_features.append(rows);lengths.append(len(x)/sr)
            np.savez_compressed(dest,speaker=speaker,split=split,modes=modes,names=[p.name for p in paths],features=all_features,durations=lengths)
            print(speaker,flush=True)


if __name__=='__main__':
    main()
