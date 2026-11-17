"""Stress-test the locked maps on shortened speech; calibration uses development voices."""
import json,tempfile
from pathlib import Path
from runtime import ROOT,write_json
from layer_blend_map import Encoder
from dataset import load_data,unit
from infer import encode
import soundfile as sf
import numpy as np
import torch


def main():
    data=load_data();ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    model=Encoder(ROOT/'runs/layer_blend_dev.npz',ROOT.parent/'claude_baselines/models','cuda')
    records=[];fail=[]
    with tempfile.TemporaryDirectory() as tmp:
        crop=Path(tmp)/'crop.wav'
        for s in data['speakers']:
            sp=data['manifest']['split'][s]
            # One query per speaker per duration; three separate texts form the gallery.
            paths=[ROOT/'data_jvs'/p for p in data['clips'][s]]
            gallery={'blend':[],'selected':[]}
            for path in paths[1:]:
                z,f=model(path);gallery['blend'].append(unit(z));gallery['selected'].append(encode(f,ck))
            gallery={k:unit(np.mean(v,axis=0)) for k,v in gallery.items()}
            z,f=model(paths[0]);full={'blend':unit(z),'selected':encode(f,ck)}
            x,sr=sf.read(paths[0],dtype='float32')
            for duration in (1,2,4):
                size=sr*duration
                if size>len(x):
                    fail.append(dict(voice=s,split=sp,duration=duration,why='shorter source'));continue
                start=(len(x)-size)//2
                sf.write(crop,x[start:start+size],sr,subtype='FLOAT')
                try:
                    u,fc=model(crop)
                except ValueError as e:
                    fail.append(dict(voice=s,split=sp,duration=duration,why=str(e)));continue
                for k,v in [('blend',unit(u)),('selected',encode(fc,ck))]:
                    records.append(dict(voice=s,split=sp,duration=duration,model=k,
                        to_full=float(np.linalg.norm(v-full[k])),to_other_texts=float(np.linalg.norm(v-gallery[k]))))
    out={}
    for k in ('blend','selected'):
        for d in (1,2,4):
            for metric in ('to_full','to_other_texts'):
                get=lambda s:np.asarray([r[metric] for r in records if r['model']==k and r['duration']==d and r['split']==s])
                cal=get('dev');te=get('test')
                # One independent recording per person; finite-sample upper order statistic.
                rank=int(np.ceil(.9*(len(cal)+1)))
                if rank>len(cal) or not len(te):continue
                bound=float(np.sort(cal)[rank-1])
                out[f'{k}_{d}_{metric}']=dict(cal_people=len(cal),test_people=len(te),upper90=bound,test_median=float(np.median(te)),test_covered=int(np.sum(te<=bound)))
    write_json(ROOT/'results/crop_stability.json',dict(summary=out,failures=fail,records=records,
        target='Normal JVS center crops, unit-sphere distance. Development speakers previously used for model research; exploratory coverage.'))
    print(out)


if __name__=='__main__':
    main()
