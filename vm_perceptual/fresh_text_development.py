"""Frozen map on eight fresh texts per JVS development voice."""
from runtime import ROOT, write_json
from dataset import load_data, transform
from train_map import MetricHead, correlations
from extract_features import Frontend
import numpy as np
import torch,json
from pathlib import PurePosixPath


def main():
    data=load_data()
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    head=MetricHead(ck['input_dim'],ck['output_dim'])
    head.load_state_dict(ck['model'])
    front=Frontend(device='cuda',use_ecapa=False,use_wavlm=False)
    selection=json.loads((ROOT/'data_sources/extra_text_selection.json').read_text())
    by_voice={}
    for name in selection:
        rel=PurePosixPath(name).relative_to('jvs_ver1')
        feature=front.extract(ROOT/'data_jvs'/rel)
        by_voice.setdefault(rel.parts[0],{})[rel.name]=transform(feature,ck['preprocessing'],pca=True)
    common=sorted(set.intersection(*(set(f) for f in by_voice.values())))
    if len(common)!=8:raise ValueError(f'Expected 8 common texts; got {common}')
    rows=[r for r in data['pairs'] if r['split']=='dev']
    results={}
    for count in [1,2,4]:
        coords=[]
        for start in range(0,8,count):
            group=common[start:start+count]
            with torch.no_grad():
                coords.append({s:head(torch.tensor(np.mean([v[t] for t in group],0),dtype=torch.float32)).numpy() for s,v in by_voice.items()})
        same=[];different=[]
        for i,za in enumerate(coords):
            for j,zb in enumerate(coords):
                pred=[-np.linalg.norm(za[r['speaker_a']]-zb[r['speaker_b']]) for r in rows]
                (same if i==j else different).append(correlations(rows,pred)['mean_spearman'])
        results[count]=dict(same_texts=float(np.mean(same)),different_texts=float(np.mean(different)),disjoint_assignments=len(different))
    report=dict(voices=len(by_voice),texts=common,rows=len(rows),results=results)
    print(report);write_json(ROOT/'results/fresh_text_development.json',report)


if __name__=='__main__':
    main()
