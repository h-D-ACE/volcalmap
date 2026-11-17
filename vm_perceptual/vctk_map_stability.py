"""Frozen JVS map on simultaneous VCTK recordings from two microphones."""
from runtime import ROOT, write_json, file_hash
from extract_features import Frontend
from infer import encode
from dataset import unit
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
import torch,json,argparse


def load(folder,frontend):
    manifest=json.loads((folder/'manifest.json').read_text())
    result={}
    for row in manifest['files']:
        path=folder/row['file']
        if path.suffix!='.flac':continue
        if file_hash(path)!=row['sha256']:raise ValueError(path)
        name=path.stem.split('_')
        key='_'.join(name[:2])
        result.setdefault(key,{})[name[-1]]=np.asarray(frontend.extract(path)['acoustic'])
    return {k:v for k,v in result.items() if set(v)=={'mic1','mic2'}}


def run(validation,confirmation):
    frontend=Frontend(device='cuda',use_ecapa=False,use_wavlm=False)
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    calibration=load(validation,frontend); unseen=load(confirmation,frontend)
    people=lambda items:set(s.split('_')[0] for s in items)
    if people(calibration)&people(unseen):raise ValueError('Overlapping speakers')
    bias=np.mean([np.mean([v['mic2']-v['mic1'] for k,v in calibration.items() if k.startswith(s+'_')],axis=0) for s in people(calibration)],axis=0)
    embed=lambda v:encode({'acoustic':v},ck)
    def evaluate(b):
        keys=sorted(unseen)
        x=np.asarray([embed(unseen[k]['mic1']) for k in keys])
        y=np.asarray([embed(unseen[k]['mic2']-b) for k in keys])
        distances=np.linalg.norm(x[:,None]-y[None,:],axis=-1)
        pair=np.diag(distances)
        same=np.array([[k.split('_')[0]==j.split('_')[0] for j in keys] for k in keys])
        repeat=distances[same & ~np.eye(len(keys),dtype=bool)]
        all_pairs=np.triu_indices(len(keys),1)
        dx=np.linalg.norm(x[:,None]-x[None,:],axis=-1)[all_pairs]
        dy=np.linalg.norm(y[:,None]-y[None,:],axis=-1)[all_pairs]
        return dict(paired_median=float(np.median(pair)),paired_p90=float(np.quantile(pair,.9)),other_sentence_same_person_median=float(np.median(repeat)),cross_mic_same_recording_top1=float(np.mean(distances.argmin(0)==np.arange(len(keys)))),geometry_srcc=float(spearmanr(dx,dy).statistic))
    result=dict(calibration_pairs=len(calibration),unseen_pairs=len(unseen),unseen_people=sorted(people(unseen)),raw=evaluate(0),known_mic_bias_correction=evaluate(bias),scope='Paired device stability on previously collected English recordings; no perceptual labels')
    write_json(ROOT/'results/vctk_map_stability.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('calibration',type=Path);p.add_argument('confirmation',type=Path)
    args=p.parse_args();run(args.calibration,args.confirmation)
