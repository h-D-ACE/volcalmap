"""How reproducible are the gallery's listening-score neighbors with two disjoint sets of votes?"""
import numpy as np
import json
from dataset import load_data
from runtime import ROOT,write_json


def main():
    data=load_data();gallery=json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    votes={}
    for r in data['pairs']:
        votes[(r['speaker_a'],r['speaker_b'])]=votes[(r['speaker_b'],r['speaker_a'])]=r['ratings']
    rng=np.random.default_rng(83651);report={}
    reps=1500
    for split in ('dev','test'):
        names=[s for s in data['speakers'] if data['manifest']['split'][s]==split]
        by_voice=[]
        for s in names:
            g=gallery[data['manifest']['gender'][s]]
            h=np.array([votes[s,n] for n in g],dtype=float)
            if h.shape!=(10,10):raise ValueError('Expected ten references and ten votes per pair')
            # Split each edge without reusing a vote. No assumption of rater identifiers.
            order=rng.random((reps,10,10)).argsort(-1)
            splitvotes=np.take_along_axis(np.broadcast_to(h,(reps,10,10)),order,axis=-1)
            first=splitvotes[:,:,:5].mean(-1);second=splitvotes[:,:,5:].mean(-1)
            truth=np.sort(second,axis=1)[:,-3]
            a=np.argmax(first+rng.uniform(-1e-8,1e-8,first.shape),axis=1)
            rr=np.arange(reps)
            oracle=(second[rr,a]>=truth)
            # Bayesian bootstrap: raw votes receive random nonnegative weights that sum to one per pair.
            weights=rng.dirichlet(np.ones(10),size=(reps,10))
            draws=(weights*h).sum(-1)
            top3=draws>=np.sort(draws,axis=1)[:,-3,None]
            hard=h.mean(1)
            choices={}
            def vec(key,speaker):
                if key in ('own','combined'):
                    return np.load(ROOT/'features'/f'rank_map_{key}'/f'{speaker}.npy')
                with np.load(ROOT/'features/gallery_vectors'/f'{speaker}.npz') as file:return file[key]
            for key in ('current','student','voxsim','own','combined'):
                scores=vec(key,s)[:4]@np.stack([vec(key,n)[4:8].mean(0) for n in g]).T
                idx=scores.argmax(1)
                choices[key]=dict(estimated_top3=float(top3[:,idx].mean()),
                    empirical_top3=float(np.mean(hard[idx]>=np.sort(hard)[-3])),
                    second_half_top3=float((second[:,idx]>=truth[:,None]).mean()))
            by_voice.append(dict(half_to_half_top3=float(oracle.mean()),random_half_to_half=float((second>=truth[:,None]).mean()),
                probability_full_score_winner_top3=float(top3[:,np.argmax(hard)].mean()),choices=choices))
        report[split]={k:float(np.mean([r[k] for r in by_voice])) for k in by_voice[0] if k!='choices'}
        report[split]['models']={key:{metric:float(np.mean([r['choices'][key][metric] for r in by_voice]))
            for metric in by_voice[0]['choices'][key]} for key in by_voice[0]['choices']}
    write_json(ROOT/'results/rating_repeatability.json',dict(replicates=reps,groups=report,
        note='Within-pair repeatability proxy. Identities of listeners across edges are unknown; this is not an upper bound on machine performance.'))
    print(report,flush=True)


if __name__=='__main__':main()
