"""Frozen ReDimNet2 B2 identity model as a perceptual timbre baseline on identical audio."""
from pathlib import Path
import sys
import math
import json
import soundfile as sf
import torch
import numpy as np
from scipy.signal import resample_poly
from dataset import load_data,unit
from runtime import ROOT,write_json,file_hash
from gallery_benchmark import quality
from perceptual_rank import metrics
from wave_folds import paired_labels


@torch.inference_mode()
def main():
    ref=Path('/root/autodl-tmp/voxsim_reference')
    commit='c5bbe0b76e37df698c403f8844e41304ceab6307'
    sys.path.insert(0,str(ref/f'redimnet2-{commit}'))
    from redimnet2 import ReDimNet2Wrap
    path=ref/'b2-vox2-lm.pt'
    sha='0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a'
    if file_hash(path)!=sha:raise ValueError('ReDimNet2 checkpoint checksum mismatch')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=ReDimNet2Wrap(**ck['model_config']).cuda().eval()
    model.load_state_dict(ck['state_dict'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=ROOT/'features/redim_b2';cache.mkdir(exist_ok=True)
    data=load_data();codes={}
    for s in data['speakers']:
        files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))
        vectors=[]
        for file in files:
            x,sr=sf.read(file,dtype='float32');g=math.gcd(sr,16000)
            x=torch.from_numpy(resample_poly(x,16000//g,sr//g).astype('float32')).cuda()[None]
            vectors.append(unit(model(x).cpu().numpy()[0]))
        codes[s]=np.array(vectors)
        np.savez_compressed(cache/f'{s}.npz',vectors=codes[s],names=[p.name for p in files],weight_sha256=sha)
    published=json.loads((ROOT/'results/gallery_benchmark.json').read_text())
    gallery=published['gallery'];ratings={}
    for r in data['pairs']:
        ratings[r['speaker_a'],r['speaker_b']]=ratings[r['speaker_b'],r['speaker_a']]=r['score']
    out={}
    for split in ('dev','test'):
        selected=[s for s in data['speakers'] if data['manifest']['split'][s]==split];people=[]
        start=4 if split=='dev' else 0
        perclip=np.stack([codes[s][start:start+4] for s in selected])
        score=metrics(perclip,paired_labels(data,selected,np.arange(len(selected))))
        for s in selected:
            gal=gallery[data['manifest']['gender'][s]]
            y=np.array([ratings[s,n] for n in gal])
            sims=codes[s][:4]@np.stack([codes[n][4:8].mean(0) for n in gal]).T
            people.append([quality(z,y) for z in sims])
        out[split]=dict(single=score,fixed_gallery={k:float(np.mean([[r[k] for r in person] for person in people])) for k in people[0][0]})
    write_json(ROOT/'results/redim_reference.json',dict(source_commit=commit,checkpoint=sha,groups=out))
    print(out,flush=True)


if __name__=='__main__':main()
