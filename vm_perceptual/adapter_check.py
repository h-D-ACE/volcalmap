"""Can another frontend locate the SAME recordings in our fixed map? Exploratory."""
import json
from runtime import ROOT,write_json
from dataset import load_data,make_input,pca_transform,unit
from infer import encode
from extract_features import Frontend
import numpy as np
import torch

def main():
    data=load_data();ck=torch.load(ROOT/'runs/selected.pt',weights_only=True)
    files=[f for s in data['speakers'] for f in data['clips'][s]]
    splits=np.array([data['manifest']['split'][f.split('/')[0]] for f in files])
    canonical=np.array([encode(data['features'][f],ck) for f in files])
    # Deterministic repeated evaluation is a different property from cross-model agreement.
    again=np.array([encode(data['features'][f],ck) for f in files])
    result=dict(repeated_same_model_max_abs=float(np.max(abs(canonical-again))),adapters={})
    front=Frontend(use_wavlm=ck['representation']=='wavlm3',use_ecapa=ck['representation'] in ('ecapa','fusion'))
    f=ROOT/'data_jvs'/files[0];fa=front.extract(f);fb=front.extract(f)
    result['repeat_full_audio_pipeline']=dict(audio=files[0],
        max_abs_coordinate_difference=float(np.max(abs(encode(fa,ck)-encode(fb,ck)))),
        scope='two CPU runs, fixed audio, versions and parameters; not cross-hardware bit-exact guarantee')
    for rep in ('ecapa','wavlm3'):
        _,vectors,conf=make_input(data,rep)
        X=pca_transform(np.array([vectors[f] for f in files]),conf)
        train=X[splits=='train'];mu=train.mean(0);train=train-mu
        Y=canonical[splits=='train'];ym=Y.mean(0);Y=Y-ym
        candidates=[]
        for ridge in (.1,1.,10.,100.):
            W=np.linalg.solve(train.T@train+ridge*np.eye(train.shape[1]),train.T@Y)
            pred=unit((X-mu)@W+ym)
            err=np.linalg.norm(pred-canonical,axis=1)
            candidates.append((float(np.median(err[splits=='dev'])),ridge,W,pred,err))
        _,ridge,W,pred,err=min(candidates,key=lambda r:r[0])
        test=err[splits=='test']
        q=pred[splits=='test'];g=canonical[splits=='test'];d=np.linalg.norm(q[:,None]-g[None,:],axis=2)
        exact=int(np.sum(np.argmin(d,axis=1)==np.arange(len(q))))
        result['adapters'][rep]=dict(ridge=ridge,development_median=min(r[0] for r in candidates),
            test_clips=len(test),test_median_coordinate_error=float(np.median(test)),test_p90_coordinate_error=float(np.percentile(test,90)),
            same_recording_top1=exact/len(q),exact_coordinate_matches=int(np.sum(test<1e-6)),
            note='Same-recording retrieval among 80 held-out recordings; not identity or perceptual accuracy')
    result['interpretation']=['Same audio + fixed deterministic pipeline gives repeatable coordinates.',
        'Different frontend embeddings do not share units, axes or retained information.',
        'A learned adapter can approximate a canonical map, but lost information cannot be restored by changing axes.',
        'This is a post-lock exploratory diagnostic, not a new independently blinded perceptual test.']
    write_json(ROOT/'results/adapter_check.json',result);print(result)

if __name__=='__main__':main()
