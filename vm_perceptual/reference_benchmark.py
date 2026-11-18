"""Compare frozen published baselines and the locked small student on identical recordings."""
import itertools
import numpy as np
import torch
from scipy.stats import rankdata
from runtime import ROOT, write_json
from dataset import load_data, unit
from infer import encode
from wave_student import WaveMap,batch,load_records


def weighted_midrank(x, w):
    levels, group=np.unique(x,return_inverse=True)
    counts=np.zeros((len(w),len(levels)))
    for j,g in enumerate(group):
        counts[:,g]+=w[:,j]
    rank=np.cumsum(counts,axis=1)-.5*counts
    return rank[:,group]


def weighted_corr(a,b,w):
    total=np.maximum(w.sum(1,keepdims=True),1.)
    a=a-(a*w).sum(1,keepdims=True)/total
    b=b-(b*w).sum(1,keepdims=True)/total
    den=np.sqrt((w*a*a).sum(1)*(w*b*b).sum(1))
    return np.divide((a*b*w).sum(1),den,out=np.full(len(w),np.nan),where=den>1e-12)


def benchmark(names,data,vectors,seed):
    golds=[]; summaries={}; point={}
    rng=np.random.default_rng(seed)
    for g in ('female','male'):
        people=[s for s in names if data['manifest']['gender'][s]==g]
        idx={s:i for i,s in enumerate(people)}
        pairs=[r for r in data['pairs'] if r['speaker_a'] in idx and r['speaker_b'] in idx]
        a=np.array([idx[r['speaker_a']] for r in pairs]);b=np.array([idx[r['speaker_b']] for r in pairs])
        gold=np.asarray([r['score'] for r in pairs])
        counts=rng.multinomial(len(people),np.full(len(people),1/len(people)),size=1500)
        w=np.vstack((np.ones(len(a)),counts[:,a]*counts[:,b]))
        ranks=weighted_midrank(gold,w)
        for key,items in vectors.items():
            v=np.array([items[s] for s in people])
            for endpoint in ('single','profile'):
                if endpoint=='profile':
                    mean=v.mean(1)
                    preds=[(mean[a]*mean[b]).sum(-1)]
                else:
                    preds=[(v[a,i]*v[b,j]).sum(-1) for i,j in itertools.permutations(range(v.shape[1]),2)]
                result=np.mean([weighted_corr(ranks,weighted_midrank(pred,w),w) for pred in preds],axis=0)
                summaries.setdefault((key,endpoint),[]).append(result)
        golds.append(len(gold))
    combined={k:np.mean(x,axis=0) for k,x in summaries.items()}
    for (k,e),v in combined.items():
        delta=v-combined['current',e]
        ok=delta[1:][np.isfinite(delta[1:])]
        point.setdefault(k,{})[e]=dict(srcc=float(v[0]),difference_vs_current=float(delta[0]),
            paired_95=np.percentile(ok,[2.5,97.5]).tolist(),valid_bootstrap=len(ok))
    return dict(n_speakers=len(names),gender_pairs=golds,models=point)


@torch.inference_mode()
def main():
    data,records,by=load_records()
    ck=torch.load(ROOT/'runs/selected.pt',weights_only=True,map_location='cpu')
    model=WaveMap().cuda().eval()
    model.load_state_dict(torch.load(ROOT/'runs/wave_431_noise/best.pt',weights_only=True)['state'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    summary={}
    for split in ('dev','test'):
        vectors={k:{} for k in ('current','student','identity_ecapa','voxsim_small_reconstructed','voxsim_large')}
        speakers=[s for s in data['speakers'] if data['manifest']['split'][s]==split]
        for s in speakers:
            ids=by[data['speakers'].index(s)]
            assert len(ids)==4
            paths=[f"{s}/parallel100/wav24kHz16bit/{records[i]['name']}" for i in ids]
            vectors['current'][s]=np.array([records[i]['target'] for i in ids])
            vectors['student'][s]=np.array([model(*batch(records,[i],0)).cpu().numpy()[0] for i in ids])
            if split=='test':
                vectors['identity_ecapa'][s]=np.array([unit(np.array(data['features'][p]['ecapa'])) for p in paths])
            else:
                with np.load(ROOT/'features/development_layers.npz') as f:
                    vectors['identity_ecapa'][s]=unit(f['ecapa'][f['speakers'].tolist().index(s),4:8])
            for key,folder in [('voxsim_small_reconstructed','small'),('voxsim_large','large')]:
                with np.load(ROOT/'features'/f'voxsim_{folder}'/f'{s}.npz') as f:
                    lookup=dict(zip(f['names'].tolist(),f['vectors']))
                    vectors[key][s]=np.asarray([lookup[p] for p in paths])
        summary[split]=benchmark(speakers,data,vectors,872)
    write_json(ROOT/'results/reference_benchmark.json',summary)
    for split,block in summary.items():
        print(split,{k:[round(v[e]['srcc'],4) for e in ('single','profile')] for k,v in block['models'].items()},flush=True)


if __name__=='__main__':
    main()
