"""Evaluate locked Japanese-trained maps against independent English VCTK voice ratings."""
import argparse
import itertools
import json
import soundfile as sf
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import torch
from dataset import unit
from infer import encode as base_encode
from rank_map_candidate import Frontends, encode
from runtime import ROOT, PROJECT, file_hash, write_json
from wave_map_cli import WaveEncoder


def corr(x,y):
    a=rankdata(x);a-=a.mean()
    b=rankdata(y);b-=b.mean()
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def score(matrix,ratings,people):
    left,right=np.triu_indices(len(people),1)
    a,b=people[left],people[right]
    distinct=a!=b;a=a[distinct];b=b[distinct]
    if len(np.unique(a.tolist()+b.tolist()))<3:return np.nan
    y=ratings[a,b]
    return .5*(corr(matrix[a,b],y)+corr(matrix[b,a],y))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',type=Path,default=ROOT/'runs')
    ap.add_argument('--device',default='cpu');args=ap.parse_args()
    source=ROOT/'data_sources'
    names=(source/'VCTK_62females.list').read_text().split()
    ratings=np.loadtxt(source/'VCTK_62females.csv',delimiter=',')
    if ratings.shape!=(len(names),len(names)) or not np.array_equal(ratings,ratings.T):
        raise ValueError('Rating matrix and speaker list must be paired and symmetric')
    clips={};sentences={};transcripts={};short=[]
    for folder in ('data','data_vctk_validation','data_vctk_confirmation'):
        directory=PROJECT/'vm_acoustic'/folder
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8-sig'))
        digests={Path(r.get('file') or r['path']).name:r['sha256'] for r in manifest['files']}
        for path in directory.rglob('*.flac'):
            who,text,mic=path.stem.split('_')
            if who not in names:continue
            transcript=path.with_name(f'{who}_{text}.txt')
            for file in (path,transcript):
                if file_hash(file)!=digests[file.name]:raise ValueError(f'Input digest mismatch: {file.name}')
            if sf.info(path).duration<1:
                short.append(path.name);continue
            tx=transcript.read_text(encoding='utf-8').casefold().strip()
            transcripts[who,text]=tx
            if text in ('003','005') and sentences.setdefault(text,tx)!=tx:
                raise ValueError('Different words within a primary text ID')
            if (who,text,mic) in clips:raise ValueError('Duplicate recording key')
            clips[who,text,mic]=path
    people=sorted({w for w,t,m in clips})
    for key in itertools.product(people,('003','005'),('mic1','mic2')):
        if key not in clips:raise ValueError(f'Missing paired recording {key}')
    front=Frontends(False,args.device);maps={}
    ck=torch.load(args.runs/'selected.pt',map_location='cpu',weights_only=True)
    small=WaveEncoder(args.runs/'wave_431_noise/best.pt',args.device)
    robust=WaveEncoder(args.runs/'robust_ssl_consistency_177_1800/best.pt',args.device,'robust_student')
    control=WaveEncoder(args.runs/'robust_ssl_unaugmented_177_1800/best.pt',args.device,'unaugmented_student')
    with np.load(args.runs/'rank_map_own.npz') as params:
        for k,p in clips.items():
            v=front.extract(p)
            maps[k]=dict(current=base_encode({'acoustic':v[0]},ck),student=small.vector(p),own=encode(v,params),
                         ecapa=unit(v[1]),robust_student=robust.vector(p),unaugmented_student=control.vector(p))
    positions=[names.index(s) for s in people];ratings=ratings[np.ix_(positions,positions)]
    n=len(people);idx=np.arange(n)
    rng=np.random.default_rng(38194)
    samples=rng.integers(n,size=(1500,n))
    report={}
    for mic in ('mic1','mic2'):
        values={};resamples={}
        for method in next(iter(maps.values())):
            a,b=[np.stack([maps[s,t,mic][method] for s in people]) for t in ('003','005')]
            matrix=a@b.T
            estimate=score(matrix,ratings,idx)
            boot=np.array([score(matrix,ratings,rows) for rows in samples])
            resamples[method]=boot
            if method=='current':reference=boot
            delta=boot-reference
            loo=[score(matrix,ratings,np.delete(idx,i)) for i in idx]
            values[method]=dict(srcc=estimate,people_resampled_95=np.nanquantile(boot,[.025,.975]).tolist(),
                paired_delta_vs_current_95=np.nanquantile(delta,[.025,.975]).tolist(),leave_one_out=[min(loo),max(loo)])
        values['robust_student']['paired_delta_vs_unaugmented_95']=np.nanquantile(
            resamples['robust_student']-resamples['unaugmented_student'],[.025,.975]).tolist()
        report[mic]=values
    extra={};counts=[]
    for mic in ('mic1','mic2'):
        matrices={k:np.zeros((n,n)) for k in next(iter(maps.values()))}
        for i,j in itertools.combinations(range(n),2):
            a,b=people[i],people[j]
            ids=sorted(t for w,t,m in maps if w==a and m==mic and t not in ('003','005')
                       and (b,t,mic) in maps and transcripts[a,t]==transcripts[b,t])[:4]
            if len(ids)<2:raise ValueError(f'Insufficient disjoint shared texts for {a}, {b}')
            counts.append(len(ids))
            for key,mat in matrices.items():
                mat[i,j]=mat[j,i]=np.mean([maps[a,u,mic][key]@maps[b,v,mic][key]
                                           for u,v in itertools.permutations(ids,2)])
        extra[mic]={key:score(mat,ratings,idx) for key,mat in matrices.items()}
    result=dict(people=people,pairs=n*(n-1)//2,files=len(clips),primary_files=n*4,
        skipped_under_one_second=short,additional_texts=dict(scores=extra,min_per_pair=min(counts),max_per_pair=max(counts),
        protocol='Exclude primary sentences; for each pair use up to four matching transcript IDs, average all cross-text dot products, then score the 55 voice pairs.'),texts=['003','005'],results=report,
        model_versions=dict(current=file_hash(args.runs/'selected.pt'),student=small.version,own=file_hash(args.runs/'rank_map_own.npz'),robust_student=robust.version,unaugmented_student=control.version),
        matrix_sha256=file_hash(source/'VCTK_62females.csv'),speakers_sha256=file_hash(source/'VCTK_62females.list'),
        source='https://sython.org/demo/JSPS-DC1/index.html',
        scope='Previously available VCTK speakers appearing in the published female listener matrix. Two verified common texts, two separate microphones, frozen models. Ratings describe voices, not these particular clip pairs. VCTK microphones had been examined in channel diagnostics.')
    write_json(ROOT/'results/english_listener_check.json',result)
    print(n,{mic:{k:round(v['srcc'],4) for k,v in r.items()} for mic,r in report.items()},flush=True)


if __name__=='__main__':main()
