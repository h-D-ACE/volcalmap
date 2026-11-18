"""Fixed normal/falsetto diagnostics. Style labels do not specify a perceptual distance."""
import math
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from dataset import load_data, unit
from infer import encode
from runtime import ROOT, write_json
from wave_student import WaveMap, batch


@torch.inference_mode()
def main():
    data=load_data()
    base=torch.load(ROOT/'runs/selected.pt',map_location='cpu',weights_only=True)
    student=WaveMap().cuda().eval()
    ck=torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)
    student.load_state_dict(ck['state'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    names=[s for s in data['speakers'] if data['manifest']['split'][s]=='test']
    result={}; stats={}; clips=[]
    for s in names:
        normal=sorted(data['clips'][s])
        falset=sorted(f for f in data['features'] if f.startswith(s+'/falset10/'))
        assert len(normal)==4 and len(falset)==2
        clips+=normal+falset
    vectors={k:[] for k in ('map','student','ecapa')}
    for path in clips:
        feature=data['features'][path]
        vectors['map'].append(encode(feature,base))
        vectors['ecapa'].append(unit(np.array(feature['ecapa'])))
        x,sr=sf.read(ROOT/'data_jvs'/path,dtype='float32')
        g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g).astype('float32')
        z=student(*batch([{'wave':x}],[0],0))
        vectors['student'].append(z.cpu().numpy()[0])
    for name,z in vectors.items():
        z=np.array(z).reshape(len(names),6,-1);entries=[]
        # Within a person compare only common-text pairs where possible.
        for p,s in enumerate(names):
            files=clips[6*p:6*p+6]
            same=[(i,j) for i in range(4) for j in (4,5) if files[i].split('/')[-1]==files[j].split('/')[-1]]
            cross=[np.linalg.norm(z[p,i]-z[p,j]) for i in range(4) for j in (4,5)]
            normal=[np.linalg.norm(z[p,i]-z[p,j]) for i in range(4) for j in range(i+1,4)]
            style=float(np.median(cross)); control=float(np.median(normal))
            entries.append([style,control,float(np.linalg.norm(z[p,4]-z[p,5]))])
            stats[s]=len(same)
        e=np.asarray(entries);delta=e[:,0]-e[:,1]
        rng=np.random.default_rng(778)
        samples=np.median(delta[rng.integers(len(delta),size=(5000,len(delta)))],axis=1)
        result[name]=dict(median_normal=float(np.median(e[:,1])),median_cross_style=float(np.median(e[:,0])),
            median_falsetto_repeat=float(np.median(e[:,2])),people_larger=int((delta>0).sum()),
            median_paired_difference=float(np.median(delta)),paired_percentile_95=np.percentile(samples,[2.5,97.5]).tolist())
    write_json(ROOT/'results/wave_style_audit.json',dict(speakers=len(names),matched_text_pairs=sum(stats.values()),
        per_map=result,limitation='Natural text and style may change together; no human similarity labels here.'))
    print(result,'matched text:',sum(stats.values()),flush=True)


if __name__=='__main__':
    main()
