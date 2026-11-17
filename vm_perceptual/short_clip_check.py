"""Post-lock duration robustness audit, no retuning and no new perception labels."""
import json,tempfile
from runtime import ROOT,write_json
from dataset import load_data,unit
from extract_features import Frontend
from infer import encode
import numpy as np
import soundfile as sf
import torch

def main():
    data=load_data();ck=torch.load(ROOT/'runs/selected.pt',weights_only=True)
    front=Frontend(use_wavlm=ck['representation']=='wavlm3',use_ecapa=ck['representation'] in ('ecapa','fusion'))
    rows=[]
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        for i,s in enumerate(data['speakers']):
            if data['split'][i]!='test':continue
            f=data['clips'][s][0];path=ROOT/'data_jvs'/f;y,sr=sf.read(path,dtype='float32')
            full=encode(data['features'][f],ck)
            # Reference consists of OTHER normal clips, not the query itself.
            profile=unit(np.mean([encode(data['features'][g],ck) for g in data['clips'][s][1:]],0))
            for seconds in (1,2,4):
                if len(y)/sr<seconds:rows.append(dict(speaker=s,seconds=seconds,status='source_too_short'));continue
                start=(len(y)-round(sr*seconds))//2;crop=y[start:start+round(sr*seconds)]
                dest=ROOT/tmp/'crop.wav';sf.write(dest,crop,sr,subtype='FLOAT')
                try:
                    feature=front.extract(dest);z=encode(feature,ck)
                    rows.append(dict(speaker=s,seconds=seconds,status='ok',distance_to_full=float(np.linalg.norm(z-full)),
                        distance_to_other_sentences=float(np.linalg.norm(z-profile))))
                except ValueError as e:rows.append(dict(speaker=s,seconds=seconds,status=str(e)))
    metrics={}
    for seconds in (1,2,4):
        r=[v for v in rows if v['seconds']==seconds];ok=[v for v in r if v['status']=='ok']
        metrics[str(seconds)]=dict(attempted=len(r),complete=len(ok),median_to_full=float(np.median([v['distance_to_full'] for v in ok])),
            p90_to_full=float(np.percentile([v['distance_to_full'] for v in ok],90)),
            median_to_other_sentences=float(np.median([v['distance_to_other_sentences'] for v in ok])))
    write_json(ROOT/'results/short_clip_check.json',dict(metrics=metrics,rows=rows,
        note='Center crops, 20 held-out speakers, fixed selected model; stability only, not perceptual accuracy or real-time latency'))
    print(metrics)

if __name__=='__main__':main()
