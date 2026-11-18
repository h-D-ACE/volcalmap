"""Compare one contiguous observation to four spread-out observations at equal sample budgets."""
import argparse
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from runtime import ROOT, write_json
from dataset import load_data,unit
from train_map import correlations
from infer import encode
from extract_features import Frontend
from wave_student import WaveMap
from voxsim_reference import load_published,factors
from wave_folds import paired_labels
from perceptual_rank import metrics


def pieces(wave, rate, budget, distributed):
    n=min(len(wave),round(rate*budget))
    if n==len(wave):return [wave]
    if not distributed:
        start=(len(wave)-n)//2
        return [wave[start:start+n]]
    # Equal total observed samples. The unused speech between pieces is never encoded.
    lengths=[n//4+(i<n%4) for i in range(4)]
    gaps=np.diff(np.linspace(0,len(wave)-n,6).round().astype(int))
    out=[];pos=0
    for i,length in enumerate(lengths):
        pos+=gaps[i];out.append(wave[pos:pos+length]);pos+=length
    return out


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('dev','test'),required=True)
    a=p.parse_args();data=load_data()
    ext,sha=load_published('/root/autodl-tmp/voxsim_reference','large')
    import soxr
    front=Frontend(device='cuda',use_ecapa=False,use_wavlm=False)
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    student=WaveMap().cuda().eval();student.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    names=[s for s in data['speakers'] if data['manifest']['split'][s]==a.split]
    cache=ROOT/'features'/f'window_aggregation_{a.split}';cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)/'excerpt.wav'
        for name in names:
            output=cache/f'{name}.npz'
            if output.exists():continue
            paths=sorted((ROOT/'data_jvs'/name/'parallel100/wav24kHz16bit').glob('*.wav'))
            paths=paths[4:8] if a.split=='dev' else paths[:4]
            out={k:[] for k in ('current','student','voxsim')}
            for path in paths:
                wave,sr=sf.read(path,dtype='float32')
                for budget,dist in ((0,False),(2,False),(2,True),(4,False),(4,True)):
                    segs=pieces(wave,sr,budget,dist) if budget else [wave]
                    v={k:[] for k in out}
                    for w in segs:
                        sf.write(work,w,sr,subtype='FLOAT')
                        try:
                            z=encode(front.extract(work),ck)
                        except ValueError:
                            z=np.full(16,np.nan)
                        v['current'].append(z)
                        g=math.gcd(sr,16000)
                        x=torch.tensor(resample_poly(w,16000//g,sr//g).astype('float32'),device='cuda')[None]
                        v['student'].append(student(x,torch.tensor([x.shape[1]],device='cuda')).cpu().numpy()[0])
                        v['voxsim'].append(factors(ext,soxr.resample(w,sr,16000,quality='HQ')))
                    for k in v:
                        z=np.stack(v[k]);valid=np.isfinite(z).all(1)
                        out[k].append(unit(z[valid].mean(0)) if valid.any() else np.full(z.shape[1],np.nan))
            stage=output.with_suffix('.partial.npz')
            np.savez_compressed(stage,**{k:np.asarray(v).reshape(4,5,-1) for k,v in out.items()});stage.replace(output)
            print(a.split,name,flush=True)
    labels=paired_labels(data,names,np.arange(len(names)))
    modes=('full','2s_center','2s_spread','4s_center','4s_spread')
    result={}
    for model in ('current','student','voxsim'):
        z=np.stack([np.load(cache/f'{name}.npz')[model] for name in names])
        result[model]={}
        for i,mode in enumerate(modes):
            q=z[:,:,i];valid=np.isfinite(q).all(-1)
            scores=metrics(q,labels) if valid.all() else None
            result[model][mode]=dict(recordings=int(valid.sum()),total=int(valid.size),score=scores,
                median_l2_to_full=float(np.median(np.linalg.norm(q[valid]-z[:,:,0][valid],axis=-1))))
    write_json(ROOT/'results'/f'window_aggregation_{a.split}.json',result)


if __name__=='__main__':main()
