"""Re-fit fold-specific teachers and evaluate single-utterance perceptual geometry.

Uses only the original training and development people. Each outer fold uses
different people and sentence IDs from gradient updates. The old test is excluded.
"""
import argparse
import math
import time
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import rankdata
from torch.nn import functional as F
from runtime import ROOT, write_json
from wave_student import WaveMap, batch
from dataset import load_data, unit


def geometry(emb, mask, labels):
    # Split by sex and evaluate all ordered unequal sentence assignments.
    result = []
    for a,b,gold in labels:
        y = rankdata(gold)
        for i in range(4):
            for j in range(4):
                if i == j: continue
                similarity = -(np.square(emb[a,i]-emb[b,j])).sum(-1)
                result.append(np.corrcoef(y,rankdata(similarity))[0,1])
    return float(np.mean(result))


def fold_target(raw, tr):
    train = raw[tr,:4].reshape(-1,raw.shape[-1])
    scaled = (raw-train.mean(0))/np.maximum(train.std(0),.05)
    profiles = scaled[tr,:4].mean(1)
    _,_,axes = np.linalg.svd(profiles-profiles.mean(0),full_matrices=False)
    return unit((scaled-profiles.mean(0))@axes[:16].T).astype(np.float32)


def paired_labels(data,names,indices):
    local = {names[k]:i for i,k in enumerate(indices)}
    labels = []
    for gender in ('male','female'):
        rows = [r for r in data['pairs'] if r['gender']==gender and r['speaker_a'] in local and r['speaker_b'] in local]
        labels.append(([local[r['speaker_a']] for r in rows],[local[r['speaker_b']] for r in rows],[r['score'] for r in rows]))
    return labels


@torch.inference_mode()
def evaluate(model, records, people, start):
    model.eval()
    ids = [p*12+i for p in people for i in range(start,start+4)]
    out={}
    for seconds in (0,1,2,4):
        vectors = [model(*batch(records,ids[k:k+16],seconds)).cpu().numpy() for k in range(0,len(ids),16)]
        out[seconds]=np.concatenate(vectors).reshape(len(people),4,-1)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repeat',type=int,default=0)
    ap.add_argument('--steps',type=int,default=2000)
    args=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=load_data()
    with np.load(ROOT/'features/development_layers.npz') as cache:
        names=cache['speakers'].tolist()
        acoustic=cache['acoustic']; ecapa=unit(cache['ecapa'])
    assert len(names)==80 and all(data['manifest']['split'][n]!='test' for n in names)
    records=[]
    for name in names:
        files=sorted((ROOT/'data_jvs'/name/'parallel100/wav24kHz16bit').glob('*.wav'))[:12]
        assert len(files)==12
        for f in files:
            x,sr=sf.read(f,dtype='float32')
            g=math.gcd(sr,16000)
            records.append({'wave':resample_poly(x,16000//g,sr//g).astype('float32')})
    rng=np.random.default_rng(9000+args.repeat)
    partitions=[rng.permutation([k for k,n in enumerate(names) if data['manifest']['gender'][n]==g]) for g in ('male','female')]
    folds=[np.concatenate([p[i::5] for p in partitions]) for i in range(5)]
    folder=ROOT/'runs'/f'wave_folds_{args.repeat}_{args.steps}'
    folder.mkdir(parents=True,exist_ok=True)
    for fold in range(5):
        te,va=folds[fold],folds[(fold+1)%5]
        tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
        target=fold_target(acoustic,tr)
        vl=paired_labels(data,names,va); tl=paired_labels(data,names,te)
        for recipe in ('acoustic','acoustic_ecapa'):
            path=folder/f'{fold}_{recipe}.json'
            if path.exists():continue
            started=time.monotonic()
            seed=771+10*fold+100*args.repeat
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=WaveMap().cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.steps,eta_min=.00005)
            best=-np.inf
            history=[]
            for step in range(1,args.steps+1):
                model.train()
                who=rng.choice(tr,32,replace=False);which=rng.integers(4,size=32)
                x,n=batch(records,who*12+which,float(rng.choice([1,2,3,4,6])),rng)
                valid=torch.arange(x.shape[1],device='cuda')[None,:]<n[:,None]
                scale=x.square().sum(-1,keepdim=True).div(n[:,None]).sqrt()
                db=torch.empty((32,1),device='cuda').uniform_(12,35)
                x=(x+torch.randn_like(x)*scale*10**(-db/20)*(torch.rand((32,1),device='cuda')<.4))*valid
                z=model(x,n)
                loss=(z-torch.from_numpy(target[who,which]).cuda()).square().sum(-1).mean()
                if recipe=='acoustic_ecapa':
                    e=torch.tensor(ecapa[who,which],device='cuda',dtype=z.dtype)
                    loss=loss+F.mse_loss(z@z.T,e@e.T)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),3)
                opt.step();schedule.step()
                if step%250==0 or step==args.steps:
                    pred=evaluate(model,records,va,4)
                    val=float(np.mean([geometry(v,None,vl) for v in pred.values()]))
                    history.append([step,val])
                    if val>best:
                        best=val;best_step=step
                        state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
            model.load_state_dict(state)
            out=evaluate(model,records,te,8)
            scores={str(k):geometry(v,None,tl) for k,v in out.items()}
            write_json(path,dict(outer=fold,recipe=recipe,seed=seed,train=[names[k] for k in tr],
                dev=[names[k] for k in va],test=[names[k] for k in te],dev_history=history,
                best_step=best_step,outer_scores=scores,acoustic_pca=geometry(target[te,8:12],None,tl),
                ecapa=geometry(ecapa[te,8:12],None,tl),seconds=time.monotonic()-started))
            print(fold,recipe,scores,flush=True)


if __name__=='__main__':
    main()
