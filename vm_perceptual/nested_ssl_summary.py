"""Summarize matched training-text budgets without treating overlapping folds as independent."""
from itertools import product
import json
from statistics import mean
from runtime import ROOT, write_json


def summarize(root, steps=1200):
    records = {}
    for repeat, fold, seed, count in product(range(5), range(5), (76, 431), (2, 6)):
        key = (repeat, fold, seed, count)
        file = root / 'runs' / f'nested_ssl_{repeat}_{steps}' / f'{fold}_seed{seed}_texts{count}.json'
        row = json.loads(file.read_text())
        if (row['repeat'], row['fold'], row['seed'], len(row['train_texts'])) != key:
            raise ValueError(f'Metadata mismatch in {file}')
        records[key] = row
    durations = {}
    for seconds in ('0', '1', '2', '4'):
        rows = []
        for repeat, fold in product(range(5), repeat=2):
            row = {str(count): mean(records[repeat, fold, seed, count]['outer'][seconds]['srcc']
                                    for seed in (76, 431)) for count in (2, 6)}
            row.update(repeat=repeat, fold=fold, delta=row['6'] - row['2'])
            rows.append(row)
        durations[seconds] = dict(
            two_texts=mean(r['2'] for r in rows), six_texts=mean(r['6'] for r in rows),
            paired_delta=mean(r['delta'] for r in rows),
            folds_improved=sum(r['delta'] > 0 for r in rows),
            repeat_deltas=[mean(r['delta'] for r in rows if r['repeat'] == i) for i in range(5)],
            fold_means=rows)
    teachers = [records[i, j, 76, 2]['teacher_outer'] for i, j in product(range(5), repeat=2)]
    return dict(training_fits=len(records), people=80, repeated_person_splits=25,
                teacher_full=mean(teachers), durations=durations,
                scope='Same development pool reused across repeats. No independent-fold confidence interval.')


if __name__ == '__main__':
    result = summarize(ROOT)
    write_json(ROOT / 'results/nested_ssl_summary.json', result)
    print(json.dumps({k: {m: v for m, v in r.items() if m != 'fold_means'}
                      for k, r in result['durations'].items()}, indent=2))
