"""Paired perturbations of the same voice: distances and retrieval gallery changes."""
import tempfile
from pathlib import Path
from runtime import ROOT,write_json
from infer import encode
from extract_features import Frontend
from dataset import load_data
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly


def main():
    data=load_data();ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    frontend=Frontend(device='cuda',use_ecapa=False,use_wavlm=False,edge_policy=ck.get('edge_policy','preserve'))
    gallery=[];names=[]
    for s in data['speakers']:
        if data['manifest']['split'][s]=='test':continue
        for path in data['clips'][s]:
            names.append(path);gallery.append(encode(data['features'][path],ck))
    gallery=np.asarray(gallery)
    metrics={}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'query.wav'
        for s in data['speakers']:
            if data['manifest']['split'][s]!='dev':continue
            for orig in data['clips'][s]:
                x,sr=sf.read(ROOT/'data_jvs'/orig,dtype='float32')
                rng=np.random.default_rng(2026+names.index(orig))
                noise=rng.standard_normal(len(x));noise*=np.sqrt(np.mean(x*x)/np.mean(noise*noise))*.1
                near=lambda z:np.argsort(np.where(np.array(names)==orig,np.inf,np.linalg.norm(gallery-z,axis=1)))[:5]
                clean=encode(frontend.extract(ROOT/'data_jvs'/orig),ck)
                ref=near(clean)
                for mode,(signal,rate) in dict(gain_minus12=(x*.25118864,sr),leading_trailing_silence=(np.pad(x,(sr,sr)),sr),add_noise_20db=(x+noise,sr),resample_to_16k=(resample_poly(x,2,3),16000),telephone_bandwidth=(resample_poly(x,1,3),8000)).items():
                    if sr!=24000:raise ValueError('Protocol uses 24k input')
                    sf.write(path,signal,rate,subtype='FLOAT')
                    try:z=encode(frontend.extract(path),ck)
                    except ValueError:
                        metrics.setdefault(mode,[]).append(None);continue
                    q=near(z);metrics.setdefault(mode,[]).append((float(np.linalg.norm(clean-z)),int(q[0]==ref[0]),len(set(q)&set(ref))/5))
    out={}
    for mode,r in metrics.items():
        a=np.asarray([v for v in r if v is not None])
        out[mode]=dict(valid=len(a),attempted=len(r),median_l2=float(np.median(a[:,0])),p90_l2=float(np.quantile(a[:,0],.9)),same_top1=float(a[:,1].mean()),overlap5=float(a[:,2].mean()))
    print(out);write_json(ROOT/'results/input_stability.json',dict(development_only=True,gallery=len(names),results=out))


if __name__=='__main__':
    main()
