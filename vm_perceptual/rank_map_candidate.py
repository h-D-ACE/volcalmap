"""Fit and export the speaker-disjoint study's 16-D ordinal retrieval candidates."""
import argparse
import math
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from scipy.signal import resample_poly
from dataset import load_data,unit
from runtime import ROOT,write_json,file_hash
from perceptual_rank import bank,train_triples,metrics
from wave_folds import paired_labels
from gallery_benchmark import quality

KEYS=('acoustic','identity','wavlm1','voxsim')


def preprocess(values,params):
    parts=[]
    for i,(key,x) in enumerate(zip(params['keys'].tolist(),values)):
        if key!='acoustic':x=unit(x)
        parts.append(unit((x-params[key+'_mean'])/params[key+'_scale']))
    x=np.concatenate(parts,axis=-1)/np.sqrt(len(parts))
    return ((x-params['mean'])@params['axes'].T).astype('float32')


def encode(values,p):
    return unit(preprocess(values,p)@p['weight'].T)


def fit(data,names,blocks,nblock):
    tr=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='train'])
    va=np.array([i for i,s in enumerate(names) if data['manifest']['split'][s]=='dev'])
    params={'keys':np.array(KEYS[:nblock])};parts=[]
    for key,x in zip(KEYS[:nblock],blocks):
        if key!='acoustic':x=unit(x)
        train=x[tr,:4].reshape(-1,x.shape[-1])
        mu=train.mean(0);sd=np.maximum(train.std(0),.05) if key=='acoustic' else np.ones_like(mu)
        params[key+'_mean']=mu;params[key+'_scale']=sd;parts.append(unit((x-mu)/sd))
    x=np.concatenate(parts,axis=-1)/np.sqrt(nblock)
    flat=x[tr,:4].reshape(-1,x.shape[-1]);params['mean']=flat.mean(0)
    _,_,vt=np.linalg.svd((flat-params['mean']).astype('float64'),full_matrices=False)
    params['axes']=vt[:64]
    t=torch.from_numpy(preprocess(blocks,params)).cuda()
    torch.manual_seed(881);w=torch.nn.Parameter(torch.eye(16,64,device='cuda'))
    init=w.detach().clone();opt=torch.optim.AdamW([w],lr=.002)
    triples=train_triples(tr,names,data);labels=paired_labels(data,names,va)
    best=-np.inf;history=[]
    for step in range(1001):
        if step%100==0:
            with torch.no_grad():
                z=F.normalize(F.linear(t[va,4:8],w),dim=-1).cpu().numpy()
                score=metrics(z,labels)['single_srcc'];history.append([step,score])
                if score>best:
                    best=score;chosen=step;params['weight']=w.detach().cpu().numpy().copy()
        if step==1000:break
        z=F.normalize(F.linear(t[tr,:4],w),dim=-1)
        a,b,c=triples[:,torch.randint(triples.shape[1],(512,),device='cuda')]
        j=torch.randint(4,(3,len(a)),device='cuda')
        delta=(z[a,j[0]]*(z[b,j[1]]-z[c,j[2]])).sum(-1)
        loss=F.softplus(-6*delta).mean()+.02*(w-init).square().sum()/16
        opt.zero_grad();loss.backward();opt.step()
    # Store projection for display and the immutable model artifact separately from the public weights.
    z=encode(blocks,params)[tr,:4].reshape(-1,16)
    mean=z.mean(0);_,_,view=np.linalg.svd(z-mean,full_matrices=False)
    params['view_mean']=mean;params['view_axes']=view
    tag='combined' if nblock==4 else 'own'
    path=ROOT/'runs'/f'rank_map_{tag}.npz';np.savez_compressed(path,**params)
    return params,dict(dev_history=history,selected_step=chosen,model_version=file_hash(path))


class Frontends:
    def __init__(self,need_reference,device="cuda"):
        from extract_features import Frontend
        from transformers import WavLMModel
        self.device=device
        self.front=Frontend(device=device,use_ecapa=True,use_wavlm=False)
        self.wavlm=WavLMModel.from_pretrained(str(ROOT.parent/'claude_baselines/models/wavlm'),local_files_only=True).eval().to(device)
        self.ref=None
        if need_reference:
            from voxsim_reference import load_published
            self.ref,_=load_published('/root/autodl-tmp/voxsim_reference','large')

    @torch.inference_mode()
    def extract(self,path,known_reference=None):
        feat=self.front.extract(path);x,sr=sf.read(path,dtype='float32')
        g=math.gcd(sr,16000);w=resample_poly(x,16000//g,sr//g).astype('float32')
        layers=self.wavlm(torch.from_numpy(w).to(self.device)[None],output_hidden_states=True).hidden_states
        values=[np.asarray(feat['acoustic'],dtype='float32'),np.asarray(feat['ecapa'],dtype='float32'),layers[1][0].mean(0).cpu().numpy()]
        if known_reference is not None:values.append(known_reference)
        elif self.ref is not None:
            import soxr
            from voxsim_reference import factors
            values.append(factors(self.ref,soxr.resample(x,sr,16000,quality='HQ')))
        return values


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query',type=Path);ap.add_argument('--variant',choices=('own','combined'),default='own')
    ap.add_argument('--output',type=Path,default=ROOT/'results/rank_map_query.json')
    a=ap.parse_args()
    if a.query:
        p=ROOT/'runs'/f'rank_map_{a.variant}.npz'
        with np.load(p) as params:
            raw=Frontends(a.variant=='combined').extract(a.query)
            z=encode(raw,params)
            write_json(a.output,dict(sample_id=file_hash(a.query),model_version=file_hash(p),timbre=dict(vector=z.tolist()),
                display=((z-params['view_mean'])@params['view_axes'][:3].T).tolist(),research_only=True))
        return
    data=load_data();names,blocks=bank(data)
    candidates={};report={}
    for tag,n in [('own',3),('combined',4)]:
        p,r=fit(data,names,blocks,n);candidates[tag]=p;report[tag]=r
    front=Frontends(False);gallery=json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    refs={tag:{} for tag in candidates}
    for tag,p in candidates.items():
        z=encode(blocks,p)
        for i,s in enumerate(names):refs[tag][s]=z[i]
    for s in data['speakers']:
        if data['manifest']['split'][s]!='test':continue
        files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))
        with np.load(ROOT/'features/voxsim_large'/f'{s}.npz') as f:
            known=dict(zip(f['names'].tolist(),f['vectors']))
        values=[front.extract(path,known[path.relative_to(ROOT/'data_jvs').as_posix()]) for path in files]
        raw=[np.stack(v) for v in zip(*values)]
        for tag,p in candidates.items():refs[tag][s]=encode(raw,p)
    scores={}
    for r in data['pairs']:scores[r['speaker_a'],r['speaker_b']]=scores[r['speaker_b'],r['speaker_a']]=r['score']
    for tag,rows in refs.items():
        for group in ('dev','test'):
            ids=[s for s in data['speakers'] if data['manifest']['split'][s]==group]
            start=4 if group=='dev' else 0
            mat=np.stack([rows[s][start:start+4] for s in ids])
            result=metrics(mat,paired_labels(data,ids,np.arange(len(ids))))
            total=[]
            for s in ids:
                gal=gallery[data['manifest']['gender'][s]]
                sim=rows[s][:4]@np.stack([rows[g][4:8].mean(0) for g in gal]).T
                y=np.array([scores[s,g] for g in gal]);total.extend(quality(row,y) for row in sim)
            result['gallery']={k:float(np.mean([r[k] for r in total])) for k in total[0]}
            report[tag][group]=result
        cache=ROOT/'features'/f'rank_map_{tag}';cache.mkdir(exist_ok=True)
        for s,z in rows.items():np.save(cache/f'{s}.npy',z)
    write_json(ROOT/'results/rank_map_candidate.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
