"""Channel-robust training of the single-backbone timbre map against frozen clean coordinates."""
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from runtime import ROOT,write_json,file_hash
from wave_student import batch,load_records
from ssl_rank_student import SSLMap,targets_for_records
from wave_folds import paired_labels
from perceptual_rank import metrics
from channel_models import apply_channel


@torch.inference_mode()
def evaluate(net,data,records,by,target,group,withheld=False):
    names=[s for s in data['speakers'] if data['manifest']['split'][s]==group]
    ids=[i for s in names for i in by[data['speakers'].index(s)]]
    labels=paired_labels(data,names,np.arange(len(names)))
    conditions=['clean','equalizer','room','both']
    if withheld:conditions+=['telephone','late_room','pink_noise']
    results={};clean=None
    for kind in conditions:
        vectors=[]
        for start in range(0,len(ids),4):
            x,n=batch(records,ids[start:start+4],0)
            x=apply_channel(x,n,kind,51007+start)
            with torch.autocast('cuda',dtype=torch.bfloat16):vectors.append(net(x,n).cpu().numpy())
        z=np.concatenate(vectors)
        if clean is None:clean=z
        results[kind]={**metrics(z.reshape(len(names),4,-1),labels),
            'mean_teacher_l2':float(np.linalg.norm(z-target[ids],axis=1).mean()),
            'median_clean_drift':float(np.median(np.linalg.norm(z-clean,axis=1)))}
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--recipe',choices=('unaugmented','channel','consistency'),required=True)
    ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--confirm-test',action='store_true')
    a=ap.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,rec,by=load_records();target=targets_for_records(rec,data);yt=torch.from_numpy(target).cuda()
    model=SSLMap('frozen').cuda();learn={k:p for k,p in model.named_parameters() if p.requires_grad}
    folder=ROOT/'runs'/f'robust_ssl_{a.recipe}_{a.seed}_{a.steps}';folder.mkdir(exist_ok=True)
    if a.confirm_test:
        ck=torch.load(folder/'best.pt',weights_only=True)
        with torch.no_grad():
            for name,p in learn.items():p.copy_(ck['state'][name])
        write_json(folder/'test.json',evaluate(model,data,rec,by,target,'test',True));return
    if (folder/'best.pt').exists():raise FileExistsError(folder)
    train=[i for i,s in enumerate(data['speakers']) if data['manifest']['split'][s]=='train']
    opt=torch.optim.AdamW(learn.values(),lr=.0005,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=.00003)
    best=-np.inf;history=[];tic=time.monotonic()
    for step in range(1,a.steps+1):
        model.train();people=rng.choice(train,16,replace=False);ids=[rng.choice(by[p]) for p in people]
        x,n=batch(rec,ids,float(rng.choice([1,2,3,4,5])),rng)
        kind=str(rng.choice(['clean','equalizer','room','both']))
        noisy=apply_channel(x,n,kind,step+a.seed*100000) if a.recipe!='unaugmented' else x
        with torch.autocast('cuda',dtype=torch.bfloat16):z=model(noisy,n)
        y=yt[ids];loss=(z-y).square().sum(-1).mean()+F.mse_loss(z@z.T,y@y.T)
        if a.recipe=='consistency' and kind!='clean':
            with torch.autocast('cuda',dtype=torch.bfloat16):anchor=model(x,n)
            loss=loss+.5*(z-anchor.detach()).square().sum(-1).mean()+.5*(anchor-y).square().sum(-1).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(learn.values(),3);opt.step();sched.step()
        if step%300==0:
            val=evaluate(model,data,rec,by,target,'dev')
            score=float(np.mean([v['single_srcc']-.5*v['mean_teacher_l2'] for v in val.values()]))
            history.append(dict(step=step,utility=score,dev=val,seconds=time.monotonic()-tic))
            if score>best:
                best=score
                torch.save(dict(state={k:p.detach().cpu().clone() for k,p in learn.items()},step=step,recipe=a.recipe,seed=a.seed,
                    teacher=file_hash(ROOT/'runs/rank_map_own.npz')),folder/'best.pt')
            write_json(folder/'dev.json',history);print(a.recipe,a.seed,step,round(score,4),flush=True)


if __name__=='__main__':main()
