"""Measure fixed visualization axes against the full experimental voice space."""
import json
from runtime import ROOT,write_json
from dataset import unit
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def main():
    cache=np.load(ROOT/'features/development_layers.npz')
    params=np.load(ROOT/'runs/layer_blend_dev.npz')
    meta=json.loads((ROOT/'data_jvs/manifest.json').read_text())
    people=[i for i,s in enumerate(cache['speakers']) if meta['split'][s]=='dev']
    blocks=[]
    for name in params['names']:
        raw=cache['layers'][people,:,int(name.split('_')[1])] if name.startswith('wavlm_') else cache[name][people]
        if name!='acoustic':raw=unit(raw)
        blocks.append(np.sqrt(params[name+'_w'])*unit((raw-params[name+'_mean'])/params[name+'_scale']))
    z=np.concatenate(blocks,axis=-1).reshape(len(people)*12,-1)
    full=pdist(z);mat=squareform(full);np.fill_diagonal(mat,np.inf)
    order=np.argsort(mat,axis=1)[:,:5]
    out={}
    for d in (1,2,3,5,8,16,32):
        view=(z-params['view_mean'])@params['axes'][:d].T
        dv=pdist(view);m=squareform(dv);np.fill_diagonal(m,np.inf)
        near=np.argsort(m,axis=1)[:,:5]
        out[d]=dict(recall5=float(np.mean([len(set(a)&set(b))/5 for a,b in zip(order,near)])),distance_srcc=float(spearmanr(full,dv).statistic),stress=float(np.linalg.norm(dv-full)/np.linalg.norm(full)))
    write_json(ROOT/'results/blend_projection.json',dict(dev_clips=len(z),full_dim=z.shape[1],views=out));print(out)


if __name__=='__main__':
    main()
