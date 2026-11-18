"""Summarize paired unseen-person transfer and microphone experiments by fold."""
import json
from statistics import mean
from runtime import ROOT, write_json


def summarize(root, repeats=3, steps=900):
    methods=('unaugmented','consistency')
    metrics=('outer_srcc','paired_l2','cross_voice_top5_overlap')
    pairs=[]
    for repeat in range(repeats):
        for fold in range(5):
            values={}
            for method in methods:
                folder=root/'runs'/f'channel_folds_{repeat}_{steps}'
                rows=[]
                for seed in (76,431):
                    record=json.loads((folder/f'{fold}_{seed}_{method}.json').read_text())
                    if (record['repeat'],record['fold'],record['seed'],record['recipe'])!=(repeat,fold,seed,method):
                        raise ValueError('Mismatched fold metadata')
                    rows.append(dict(outer_srcc=record['outer_srcc'],**record['english_microphones']))
                values[method]={m:mean(r[m] for r in rows) for m in metrics}
            pairs.append(dict(repeat=repeat,fold=fold,methods=values))
    summary={}
    for metric in metrics:
        delta=[r['methods']['consistency'][metric]-r['methods']['unaugmented'][metric] for r in pairs]
        summary[metric]={m:mean(r['methods'][m][metric] for r in pairs) for m in methods}
        summary[metric].update(paired_delta=mean(delta),
            improved_folds=sum(x<0 if metric=='paired_l2' else x>0 for x in delta),
            repeat_deltas=[mean(delta[r*5:(r+1)*5]) for r in range(repeats)])
    return dict(completed_fits=len(pairs)*4,paired_folds=len(pairs),results=summary,folds=pairs,
        interpretation='Three repeated partitions of the same 80 Japanese voices. All models evaluated on the same 24 English mic pairs from four speakers. Fit selection on clean inner development voices.')


if __name__=='__main__':
    result=summarize(ROOT)
    write_json(ROOT/'results/channel_folds_summary.json',result)
    print(json.dumps(result['results'],indent=2))
