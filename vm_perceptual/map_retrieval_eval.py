"""Query-level timbre retrieval on people with explicit same-sex JVS similarity ratings.

Queries and gallery profiles use disjoint texts for the eight-new-sentence sets.
Outputs report graded relevance, regret, and pair scores on the actual candidate list.
"""
import itertools
import math
import tempfile
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import rankdata
from runtime import ROOT, write_json
from dataset import load_data,unit
from extract_features import Frontend
from infer import encode
from wave_student import WaveMap


def retrieve(z,groups,rating):
    q=z[:,:4];gallery=z[:,4:].mean(1)
    values={k:[] for k in ('ndcg3','expected_random_ndcg3','regret','top1_rating','random_rating','top1_tied_best','candidate_size')}
    for i,gender in enumerate(groups):
        others=[j for j,g in enumerate(groups) if i!=j and g==gender and np.isfinite(rating[i,j])]
        y=rating[i,others]
        if len(y)<3:continue
        # Linear gains shift the lowest possible JVS rating (-3) to zero.
        gain=y+3;w=1/np.log2(np.arange(2,5));ideal=np.sort(gain)[-3:][::-1]@w
        sim=q[i]@gallery[others].T
        order=np.argsort(-sim,axis=1,kind='stable')
        values['ndcg3'].append(np.mean(gain[order[:,:3]]@w/ideal))
        values['expected_random_ndcg3'].append(gain.mean()*w.sum()/ideal)
        best=y[order[:,0]]
        values['regret'].append(np.mean(y.max()-best))
        values['top1_rating'].append(best.mean())
        values['random_rating'].append(y.mean())
        values['top1_tied_best'].append(np.mean(np.isclose(best,y.max())))
        values['candidate_size'].append(len(others))
    return {k:float(np.mean(v)) for k,v in values.items()}


@torch.inference_mode()
def main():
    data=load_data()
    model=WaveMap().cuda().eval()
    model.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    base=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    front=Frontend(device='cuda',use_ecapa=True,use_wavlm=False)
    report={}
    for split in ('dev','test'):
        names=[];genders=[]
        maps={k:[] for k in ('current','student','identity_ecapa','voxsim_large')}
        for s in data['speakers']:
            if data['manifest']['split'][s]!=split:continue
            files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))[4:12]
            if len(files)!=8:continue
            names.append(s);genders.append(data['manifest']['gender'][s])
            per={k:[] for k in maps}
            with np.load(ROOT/'features/voxsim_large'/f'{s}.npz') as f:
                lookup=dict(zip(f['names'].tolist(),f['vectors']))
            for filename in files:
                relative=filename.relative_to(ROOT/'data_jvs').as_posix()
                per['voxsim_large'].append(lookup[relative])
                feat=front.extract(filename)
                per['current'].append(encode(feat,base))
                per['identity_ecapa'].append(unit(np.array(feat['ecapa'])))
                x,sr=sf.read(filename,dtype='float32');g=math.gcd(16000,sr)
                x=torch.from_numpy(resample_poly(x,16000//g,sr//g).astype('float32')).cuda()[None]
                per['student'].append(model(x,torch.tensor([x.shape[1]],device='cuda')).cpu().numpy()[0])
            for k,v in per.items():maps[k].append(v)
        rating=np.full((len(names),len(names)),np.nan)
        for r in data['pairs']:
            if r['speaker_a'] in names and r['speaker_b'] in names:
                a=names.index(r['speaker_a']);b=names.index(r['speaker_b'])
                rating[a,b]=rating[b,a]=r['score']
        full={}
        for k,rows in maps.items():
            z=np.array(rows)
            forward=retrieve(z,genders,rating)
            reverse=retrieve(np.concatenate((z[:,4:],z[:,:4]),axis=1),genders,rating)
            full[k]={m:.5*(forward[m]+reverse[m]) for m in forward}
        report[split]=dict(n=len(names),female=genders.count('female'),male=genders.count('male'),models=full)
        print(split,full,flush=True)
    write_json(ROOT/'results/map_retrieval_eval.json',report)


if __name__=='__main__':main()
