"""Test approximate sliced-W2 coordinates for distributions of speech-frame spectra."""
import json
from runtime import ROOT,write_json
from content_conditioning import frames
from extract_features import Frontend
from dataset import load_data
from nested_people import pair_rows,rank_mean
import numpy as np
from scipy.spatial.distance import pdist


def main():
    data=load_data()
    cache=np.load(ROOT/'features/development_layers.npz')
    names=cache['speakers'].tolist();person={s:i for i,s in enumerate(names)}
    tr=np.asarray([i for i,s in enumerate(names) if data['manifest']['split'][s]=='train'])
    va=np.asarray([i for i,s in enumerate(names) if data['manifest']['split'][s]=='dev'])
    frontend=Frontend(device='cuda',use_wavlm=False,use_ecapa=False)
    sets=[]
    for s in names:
        paths=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))[:12]
        sets.append([frames(frontend,p) for p in paths])
    generator=np.random.default_rng(21016)
    random_dirs=generator.normal(size=(128,19));random_dirs/=np.linalg.norm(random_dirs,axis=1,keepdims=True)
    dirs=np.vstack([np.eye(19),random_dirs]).astype('float32')
    qs=(np.arange(15)+.5)/15
    pitch_hnr=cache['acoustic'][:,:,-2:]
    def measure(fit,hold):
        # Equal observations per training recording determine frame coordinate scaling.
        sub=np.concatenate([sets[i][j][np.linspace(0,len(sets[i][j])-1,32).astype(int)] for i in fit for j in range(4)])
        mu,scale=sub.mean(0),np.maximum(sub.std(0),.01)
        desc=[];mom=[]
        for sp in sets:
            quant=[];summary=[]
            for frame in sp:
                f=(frame-mu)/scale
                quant.append(np.quantile(f@dirs.T,qs,axis=0))
                summary.append(np.r_[f.mean(0),f.std(0)])
            desc.append(quant);mom.append(summary)
        features=np.asarray(desc);mom=np.asarray(mom)
        def norm(x):
            x=x.reshape(len(names),12,-1)
            fit_data=x[fit,:4].reshape(-1,x.shape[-1]);scale=np.sqrt(np.mean(((fit_data-fit_data.mean(0))**2).sum(-1)))
            return (x-fit_data.mean(0))/max(scale,1e-9)
        pp=pitch_hnr[fit,:4].reshape(-1,2)
        aux=(pitch_hnr-pp.mean(0))/np.maximum(pp.std(0),.01)/np.sqrt(2)
        candidates={'moments':norm(mom),'marginal_quantiles':norm(features[:,:,:,:19]),
                    'sliced32':norm(features[:,:,:,19:51]),'sliced128':norm(features[:,:,:,19:])}
        for key in ('moments','sliced128'):
            for w in (.25,.5):
                candidates[f'{key}_source{w}']=np.concatenate((np.sqrt(1-w)*candidates[key],np.sqrt(w)*aux),axis=-1)
        rows=pair_rows(data['pairs'],person,hold)
        a,b=rows['a'],rows['b'];out={}
        for k,z in candidates.items():
            preds=np.stack([-np.linalg.norm(z[a,i]-z[b,j],axis=1) for i in range(4,8) for j in range(8,12)])
            out[k]=rank_mean(rows['y'],preds,rows['groups'])
        return out
    rng=np.random.default_rng(98);gender=data['manifest']['gender']
    halves=[rng.permutation([i for i in tr if gender[names[i]]==g]) for g in ('male','female')]
    folds=[]
    for f in range(3):
        held=np.concatenate([group[f::3] for group in halves])
        folds.append(measure(np.setdiff1d(tr,held),held))
    cv={k:np.mean([fold[k] for fold in folds]) for k in folds[0]}
    choice=max(cv,key=cv.get);dev=measure(tr,va)
    result=dict(method='Fixed spectral projection quantiles; squared Euclidean distance approximates sliced-W2 squared',quantiles=15,directions=128,selection=choice,inner_folds=folds,dev=dev,selected_dev=dev[choice],reference_moments=dev['moments'])
    write_json(ROOT/'results/sliced_spectra.json',result);print(result)


if __name__=='__main__':
    main()
