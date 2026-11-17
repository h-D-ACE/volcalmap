"""Speaker-disjoint perceptual graph and deterministic frozen-feature preprocessing."""
import json,hashlib
from runtime import ROOT
import numpy as np

def load_data():
    manifest=json.loads((ROOT/'data_jvs/manifest.json').read_text())
    features=json.loads((ROOT/'features/jvs_features.json').read_text())
    speakers=sorted(manifest['split']);idx={s:i for i,s in enumerate(speakers)}
    rows=[]
    for gender in ('male','female'):
        scores=json.loads((ROOT/f'data_sources/jvs_{gender}_raw_similarity_data.json').read_text())
        for key,ratings in scores.items():
            a,b=key.split('-');sa,sb=manifest['split'][a],manifest['split'][b]
            if a==b:continue
            rows.append(dict(a=idx[a],b=idx[b],speaker_a=a,speaker_b=b,gender=gender,
                split=sa if sa==sb else 'cross_split_unused',score=float(np.mean(ratings)),
                variance=float(np.var(ratings,ddof=1)),n=len(ratings),ratings=ratings))
    clips={s:sorted([f for f in features if f.startswith(s+'/parallel100/')]) for s in speakers}
    if any(len(v)!=4 for v in clips.values()):raise ValueError('Expected all four predetermined normal clips')
    split=np.array([manifest['split'][s] for s in speakers])
    if tuple(np.sum(split==k) for k in ('train','dev','test'))!=(60,20,20):raise ValueError('Unexpected split')
    return dict(manifest=manifest,features=features,speakers=speakers,clips=clips,split=split,pairs=rows,
        provenance=hashlib.sha256((ROOT/'features/jvs_features.json').read_bytes()+(ROOT/'data_sources/selection.json').read_bytes()+
            (ROOT/'data_sources/jvs_male_raw_similarity_data.json').read_bytes()+(ROOT/'data_sources/jvs_female_raw_similarity_data.json').read_bytes()).hexdigest())

def make_input(data,representation):
    keys={'acoustic':['acoustic'],'ecapa':['ecapa'],'wavlm3':['wavlm3'],'fusion':['acoustic','ecapa']}[representation]
    normal=[f for s in data['speakers'] for f in data['clips'][s]]
    trainfiles=[f for s in data['speakers'] if data['manifest']['split'][s]=='train' for f in data['clips'][s]]
    transforms=[];blocks=[]
    for key in keys:
        M=np.array([data['features'][f][key] for f in trainfiles])
        if key!='acoustic':M=M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-12)
        mu=M.mean(0)
        sd=np.maximum(M.std(0),.05) if key=='acoustic' else np.full(M.shape[1],np.sqrt(np.mean(np.sum((M-mu)**2,axis=1))))
        divisor=float(np.sqrt(M.shape[1])) if key=='acoustic' else 1.
        transforms.append(dict(key=key,mean=mu.tolist(),scale=sd.tolist(),divisor=divisor,normalize=key!='acoustic'))
    conf=dict(representation=representation,blocks=transforms)
    allclip={f:transform(data['features'][f],conf) for f in data['features']}
    profiles=np.array([np.mean([allclip[f] for f in data['clips'][s]],0) for s in data['speakers']])
    tr=profiles[data['split']=='train'];mu=tr.mean(0);_,sv,vt=np.linalg.svd(tr-mu,full_matrices=False)
    conf['pca_mean']=mu.tolist();conf['pca_axes']=vt[:min(32,len(vt))].tolist()
    conf['train_singular_values']=sv.tolist()
    return profiles,allclip,conf

def transform(features,conf,pca=False):
    blocks=[]
    for b in conf['blocks']:
        v=np.array(features[b['key']],dtype=float)
        if b['normalize']:v=v/max(np.linalg.norm(v),1e-12)
        blocks.append((v-b['mean'])/b['scale']/b['divisor'])
    x=np.concatenate(blocks)/np.sqrt(len(blocks))
    return (x-conf['pca_mean'])@np.array(conf['pca_axes']).T if pca else x

def pca_transform(X,conf):return (X-conf['pca_mean'])@np.array(conf['pca_axes']).T

def unit(X):return X/np.maximum(np.linalg.norm(X,axis=-1,keepdims=True),1e-12)
