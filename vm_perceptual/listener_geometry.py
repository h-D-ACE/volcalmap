"""How much geometry can ratings alone identify? Held-out edges in a graph of known voices."""
import numpy as np
import torch
from scipy.stats import spearmanr
from runtime import ROOT,write_json
from dataset import load_data,unit


def edge_trial(rows,names,seed):
    pos={s:i for i,s in enumerate(names)}
    pairs=np.array([(pos[r['speaker_a']],pos[r['speaker_b']]) for r in rows])
    y=np.array([r['score'] for r in rows],np.float32)
    rng=np.random.default_rng(seed);ix=rng.permutation(len(y))
    tr,va,te=np.split(ix,[int(.7*len(y)),int(.85*len(y))])
    a,b=torch.from_numpy(pairs.T).cuda();target=torch.from_numpy(y).cuda()
    # A bias-only model tests whether some voices tend to receive high similarity to many others.
    design=np.zeros((len(rows),len(names)+1));design[:,0]=1
    for k,(p,q) in enumerate(pairs):design[k,1+p]=design[k,1+q]=1
    penalty=np.eye(len(names)+1)*4;penalty[0,0]=0
    coeff=np.linalg.solve(design[tr].T@design[tr]+penalty,design[tr].T@y[tr])
    additive=design@coeff;report={}
    def assess(pred):
        return dict(dev=float(spearmanr(pred[va],y[va]).statistic),test=float(spearmanr(pred[te],y[te]).statistic),
            test_rmse=float(np.sqrt(np.mean((pred[te]-y[te])**2))))
    report['voice_bias']=assess(additive)
    configs=[(d,l) for d in (1,2,3,4,8,16,32) for l in (.01,.1,1.)]
    torch.manual_seed(seed)
    z=torch.nn.Parameter(torch.randn(len(configs),len(names),32,device='cuda')*.05)
    intercept=torch.nn.Parameter(torch.full((len(configs),1),float(y[tr].mean()),device='cuda'))
    mask=torch.tensor([[i<d for i in range(32)] for d,l in configs],device='cuda')[:,None,:]
    reg=torch.tensor([l for d,l in configs],device='cuda')
    opt=torch.optim.Adam([z,intercept],lr=.025)
    for step in range(600):
        loc=z*mask
        pred=intercept-(loc[:,a[tr]]-loc[:,b[tr]]).square().sum(-1)
        loss=(pred-target[tr]).square().mean(1)+reg*loc.square().sum(-1).mean(1)
        opt.zero_grad();loss.sum().backward();opt.step()
    with torch.no_grad():
        loc=z*mask
        predictions=(intercept-(loc[:,a]-loc[:,b]).square().sum(-1)).cpu().numpy()
    for c,p in zip(configs,predictions):report[f'd{c[0]}_l{c[1]}']=assess(p)
    selected=max((k for k in report if k!='voice_bias'),key=lambda k:report[k]['dev'])
    return dict(selected=selected,selected_test=report[selected]['test'],bias_test=report['voice_bias']['test'],
        per_dimension={str(d):report[max((k for k in report if k.startswith(f'd{d}_')),key=lambda k:report[k]['dev'])]['test'] for d in (1,2,3,4,8,16,32)})


def ambiguity(data):
    names=data['speakers'];vec=[]
    for s in names:
        v=np.load(ROOT/'features/rank_map_own'/f'{s}.npy')[:4].mean(0)
        vec.append(unit(v))
    a,b=[np.array([v for v,s in zip(vec,names) if data['manifest']['gender'][s]==g]) for g in ('male','female')]
    d=a@b.T
    axis=unit(b.mean(0))-unit(a.mean(0));axis=unit(axis)
    rot=np.eye(a.shape[1])-2*np.outer(axis,axis);positive=b@rot;negative=-positive
    def summary(z):
        cross=a@z.T
        return dict(median_cross_group_distance=float(np.median(np.sqrt((2-2*cross).clip(0)))),
            cross_group_first_neighbor_retained=float(np.mean(cross.argmax(1)==d.argmax(1))),
            maximum_within_group_dot_change=float(np.max(np.abs(z@z.T-b@b.T))))
    return dict(align_centroids=summary(positive),oppose_centroids=summary(negative),voices=[len(a),len(b)])


def main():
    data=load_data();report={}
    for g in ('female','male'):
        rows=[r for r in data['pairs'] if r['gender']==g];names=sorted({r['speaker_a'] for r in rows}|{r['speaker_b'] for r in rows})
        runs=[edge_trial(rows,names,1830+i) for i in range(12)]
        ds={d:float(np.mean([r['per_dimension'][d] for r in runs])) for d in runs[0]['per_dimension']}
        report[g]=dict(people=len(names),dimensions=ds,dev_selected_mean=float(np.mean([r['selected_test'] for r in runs])),
            additive_mean=float(np.mean([r['bias_test'] for r in runs])),selected=[r['selected'] for r in runs])
    write_json(ROOT/'results/listener_geometry.json',dict(groups=report,cross_group_ambiguity=ambiguity(data),
        scope='Ratings-only edge holdout on known voices. This experiment cannot identify the optimal audio embedding dimension.'))
    print(report,flush=True)


if __name__=='__main__':main()
