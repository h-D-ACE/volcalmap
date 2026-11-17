"""Fit the feature scaling for a frozen JVS mixture and emit single-clip coordinates."""
import argparse,json
from pathlib import Path
from runtime import ROOT, file_hash, write_json
from extract_features import Frontend, resample_poly
from dataset import unit
from transformers import WavLMModel
from math import gcd
import torch
import soundfile as sf
import numpy as np


def fit():
    cache=np.load(ROOT/'features/development_layers.npz')
    weights=json.loads((ROOT/'results/layer_metric_dev.json').read_text())['selected_weights']
    split=json.loads((ROOT/'data_jvs/manifest.json').read_text())['split']
    tr=[i for i,s in enumerate(cache['speakers']) if split[s]=='train']
    arrays={};parts=[]
    for name,w in weights.items():
        if w<=0:continue
        raw=(cache['layers'][tr,:,int(name.split('_')[1])] if name.startswith('wavlm_') else cache[name][tr])
        x=raw.reshape(-1,raw.shape[-1])
        if name!='acoustic':x=unit(x)
        mu=x.mean(0)
        sd=np.maximum(x.std(0),.05) if name=='acoustic' else np.ones_like(mu)
        arrays[name+'_mean']=mu;arrays[name+'_scale']=sd;arrays[name+'_w']=w
        parts.append(np.sqrt(w)*unit((x-mu)/sd))
    coords=np.concatenate(parts,axis=-1)
    mu=coords.mean(0);x=coords-mu
    # Dual PCA on the smaller training-recording Gram matrix.
    vals,u=np.linalg.eigh(x@x.T)
    ids=np.argsort(vals)[::-1][:32]
    arrays['view_mean']=mu;arrays['axes']=(u[:,ids].T@x)/np.sqrt(vals[ids,None])
    arrays['names']=list(k for k,w in weights.items() if w>0)
    out=ROOT/'runs/layer_blend_dev.npz'
    np.savez_compressed(out,**arrays)
    print('trained map dimensionality',coords.shape,'projection',arrays['axes'].shape)


class Encoder:
    def __init__(self,artifact,models,device):
        self.params=np.load(artifact)
        self.front=Frontend(models,device,False,True)
        self.wavlm=WavLMModel.from_pretrained(str(Path(models)/'wavlm'),local_files_only=True).eval().to(device)
        self.device=device

    @torch.inference_mode()
    def __call__(self,path):
        f=self.front.extract(path)
        x,sr=sf.read(path,dtype='float32')
        if sr!=16000:
            g=gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
        layers=self.wavlm(torch.from_numpy(x).to(self.device)[None],output_hidden_states=True).hidden_states
        z=[]
        for name in self.params['names']:
            v=layers[int(name.split('_')[1])][0].mean(0).cpu().numpy() if name.startswith('wavlm_') else np.asarray(f[name])
            if name!='acoustic':v=unit(v)
            z.extend(np.sqrt(self.params[name+'_w'])*unit((v-self.params[name+'_mean'])/self.params[name+'_scale']))
        return np.asarray(z),f


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--audio')
    p.add_argument('--artifact',type=Path,default=ROOT/'runs/layer_blend_dev.npz')
    p.add_argument('--models',type=Path,default=ROOT.parent/'claude_baselines/models')
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,default=ROOT/'results/layer_blend_query.json')
    a=p.parse_args()
    if a.fit:fit()
    if a.audio:
        encoder=Encoder(a.artifact,a.models,a.device)
        z,_=encoder(a.audio);display=(z-encoder.params['view_mean'])@encoder.params['axes'][:3].T
        write_json(a.output,dict(model_version=file_hash(a.artifact),sample_id=file_hash(a.audio),vector=z.tolist(),display=display.tolist(),status='experimental'))
