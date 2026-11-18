"""Lock a recording-channel recipe on development data and check natural held-out microphones."""
import argparse
import math
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from runtime import ROOT,file_hash,write_json
from ssl_rank_student import SSLMap,targets_for_records
from wave_student import load_records
from robust_ssl import evaluate
from recording_transfer import assess


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/root/autodl-tmp/vm_acoustic/data_vctk_confirmation'))
    args=p.parse_args();recipes={}
    for method in ('unaugmented','channel','consistency'):
        recipes[method]={}
        for seed in (76,177,431):
            f=ROOT/'runs'/f'robust_ssl_{method}_{seed}_1800'/'dev.json'
            rows=json.loads(f.read_text())
            if rows[-1]['step']!=1800:raise ValueError('Incomplete training')
            recipes[method][seed]=max(row['utility'] for row in rows)
    choice=max(recipes,key=lambda r:np.mean(list(recipes[r].values())))
    aug=max(('channel','consistency'),key=lambda r:np.mean(list(recipes[r].values())))
    # Both the unaugmented control and the best augmented treatment were specified before test.
    data,rec,by=load_records();target=targets_for_records(rec,data)
    summary={}
    for method in ('unaugmented',aug):
        seed=sorted(recipes[method],key=recipes[method].get)[1]
        checkpoint=ROOT/'runs'/f'robust_ssl_{method}_{seed}_1800'/'best.pt'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=SSLMap('frozen').cuda().eval()
        for name,param in model.named_parameters():
            if param.requires_grad:param.copy_(ck['state'][name])
        test=evaluate(model,data,rec,by,target,'test',True)
        man=json.loads((args.data/'manifest.json').read_text());paired={}
        for item in man['files']:
            path=args.data/item['file']
            if path.suffix!='.flac':continue
            if file_hash(path)!=item['sha256']:raise ValueError(path)
            key,mic=path.stem.rsplit('_',1)
            wave,sr=sf.read(path,dtype='float32');g=math.gcd(sr,16000)
            x=torch.from_numpy(resample_poly(wave,16000//g,sr//g).astype('float32')).cuda()[None]
            with torch.autocast('cuda',dtype=torch.bfloat16):z=model(x,torch.tensor([x.shape[1]],device='cuda')).cpu().numpy()[0]
            paired.setdefault(key,{})[mic]=z
        names=sorted(k for k,v in paired.items() if set(v)=={'mic1','mic2'})
        mics=[np.stack([paired[n][k] for n in names]) for k in ('mic1','mic2')]
        summary[method]=dict(seed=seed,step=ck['step'],test=test,english_two_mics=assess(names,*mics),version=file_hash(checkpoint))
    write_json(ROOT/'results/robust_ssl_summary.json',dict(development=recipes,choice=choice,results=summary))
    print(choice,{k:v['english_two_mics']['paired_l2'] for k,v in summary.items()},flush=True)


if __name__=='__main__':main()
