"""Check what authentic listening judgments add while keeping frozen audio inputs fixed."""
import argparse
import json
import numpy as np
from dataset import load_rating_graph
from nested_ssl_trial import make_teacher, rate
from perceptual_rank import bank, train_triples, transform
from runtime import ROOT, write_json
from wave_folds import paired_labels


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--permutations',type=int,default=8)
    a=ap.parse_args()
    if a.permutations<1:ap.error('Use at least one permutation')
    data=load_rating_graph(('train','dev'));names,blocks=bank(data,False)
    gender=np.array([data['manifest']['gender'][s] for s in names])
    location=ROOT/'runs'/f'listener_control_{a.permutations}'
    location.mkdir(exist_ok=True)
    records=[]
    for repeat in range(5):
        rng=np.random.default_rng(9000+repeat)
        groups=[rng.permutation(np.flatnonzero(gender==g)) for g in ('male','female')]
        folds=[np.concatenate([g[j::5] for g in groups]) for j in range(5)]
        for k in range(5):
            out=location/f'{repeat}_{k}.json'
            if out.exists():
                records.append(json.loads(out.read_text()));continue
            te,va=folds[k],folds[(k+1)%5];tr=np.setdiff1d(np.arange(len(names)),np.r_[va,te])
            x=np.concatenate(transform(blocks,tr),axis=-1)/np.sqrt(3)
            val=paired_labels(data,names,va);outer=paired_labels(data,names,te)
            trials=[]
            for rep in range(a.permutations+1):
                mapped=np.array(names)
                if rep:
                    shuffle=np.random.default_rng(31541000+repeat*1000+k*100+rep)
                    for g in ('male','female'):
                        subset=tr[gender[tr]==g]
                        mapped[subset]=shuffle.permutation(mapped[subset])
                triples=train_triples(tr,mapped,data)
                z,step=make_teacher(x,tr,va,val,triples,881+k)
                trials.append(dict(permutation=rep,outer_srcc=rate(z[te,8:12],outer),selected_step=step))
            row=dict(repeat=repeat,fold=k,original=trials[0],permuted=trials[1:])
            temp=out.with_suffix('.partial');write_json(temp,row);temp.replace(out);records.append(row)
            print(repeat,k,round(trials[0]['outer_srcc'],3),round(np.mean([v['outer_srcc'] for v in trials[1:]]),3),flush=True)
    real=np.array([r['original']['outer_srcc'] for r in records])
    null=np.array([[s['outer_srcc'] for s in r['permuted']] for r in records])
    delta=real-null.mean(1)
    result=dict(fits=len(records)*(a.permutations+1),folds=records,original=float(real.mean()),
                shuffled=float(null.mean()),mean_paired_difference=float(delta.mean()),
                folds_higher_than_shuffled_mean=int((delta>0).sum()),repeat_deltas=delta.reshape(5,5).mean(1).tolist(),
                scope='Only training voice names in the rating graph are shuffled within gender. Same training audio, optimizer seeds, true development selection and true external-fold labels. Shared corpus, descriptive contrast.')
    write_json(ROOT/'results/listener_label_control.json',result)
    print(result['original'],result['shuffled'],result['repeat_deltas'],flush=True)


if __name__=='__main__':main()
