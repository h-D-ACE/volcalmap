"""Build matched short-window/full-recording supervision without speaker identity targets."""
import hashlib,json,tempfile
from pathlib import Path
from runtime import ROOT
from layer_blend_map import Encoder
from infer import encode
import numpy as np
import soundfile as sf
import torch


def main():
    m=json.loads((ROOT/'data_jvs/manifest.json').read_text())
    front=Encoder(ROOT/'runs/layer_blend_dev.npz',ROOT.parent/'claude_baselines/models','cuda')
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    cache=ROOT/'features/crop_records';cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        crop=Path(tmp)/'wave.wav'
        for speaker,split in sorted(m['split'].items()):
            dest=cache/f'{speaker}.npz'
            if dest.exists():continue
            paths=sorted((ROOT/'data_jvs'/speaker/'parallel100/wav24kHz16bit').glob('*.wav'))
            paths=paths[:12] if split=='train' else paths[4:8] if split=='dev' else paths[:4]
            if len(paths)!=(12 if split=='train' else 4):raise ValueError(speaker)
            rows=[];targets=[];names=[];conditions=[];emb=[];prior=[];failed=0
            for path in paths:
                x,sr=sf.read(path,dtype='float32')
                zfull,f=front(path);target=encode(f,ck)
                for duration in (0,1,2,4):
                    if duration*sr>=len(x):continue
                    for pos in ((.5,) if duration==0 else (.1,.5,.9)):
                        if duration==0:z,obs=zfull,f
                        else:
                            start=int((len(x)-duration*sr)*pos)
                            sf.write(crop,x[start:start+duration*sr],sr,subtype='FLOAT')
                            try:z,obs=front(crop)
                            except ValueError:failed+=1;continue
                        rows.append(obs['acoustic']+obs['prosody_descriptors']+[np.log(obs['duration'])])
                        targets.append(target);emb.append(z);prior.append(encode(obs,ck))
                        names.append(path.name);conditions.append([duration,pos])
            np.savez_compressed(dest,features=np.asarray(rows,np.float32),targets=np.asarray(targets,np.float32),
                emb=np.asarray(emb,np.float32),prior=np.asarray(prior,np.float32),names=names,condition=conditions,split=split,failed=failed)
            print(speaker,len(rows),failed,flush=True)


if __name__=='__main__':
    main()
