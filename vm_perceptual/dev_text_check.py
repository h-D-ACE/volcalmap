"""Development-only sensitivity to available text, using a fixed map."""
import itertools
import json
from runtime import ROOT, write_json
import numpy as np
import torch
from dataset import load_data, transform
from train_map import MetricHead, correlations


def main():
    data = load_data()
    ck = torch.load(ROOT / 'runs/selected.pt', weights_only=True, map_location='cpu')
    if ck['kind'] != 'trained' or ck['data_hash'] != data['provenance']:
        raise ValueError('Expected the frozen trained map and matching feature dataset')
    head = MetricHead(ck['input_dim'], ck['output_dim'])
    head.load_state_dict(ck['model'])
    rows = [row for row in data['pairs'] if row['split'] == 'dev']
    curves = {}
    with torch.no_grad():
        for count in (1, 2, 4):
            coordinates = []
            for subset in itertools.combinations(range(4), count):
                profiles = []
                for speaker in data['speakers']:
                    clips = data['clips'][speaker]
                    features = [transform(data['features'][clips[i]], ck['preprocessing'], pca=True) for i in subset]
                    profiles.append(np.mean(features, axis=0))
                coordinates.append(head(torch.tensor(np.asarray(profiles), dtype=torch.float32)).numpy())
            for condition in ('matched', 'disjoint'):
                scores = []
                subsets = list(itertools.combinations(range(4), count))
                for i, j in itertools.product(range(len(subsets)), repeat=2):
                    if (i != j if condition == 'matched' else bool(set(subsets[i]) & set(subsets[j]))):
                        continue
                    distances = [-np.linalg.norm(coordinates[i][r['a']] - coordinates[j][r['b']]) for r in rows]
                    scores.append(correlations(rows, distances)['mean_spearman'])
                if scores:
                    curves[f'{count}_{condition}'] = dict(assignments=len(scores), mean=float(np.mean(scores)), minimum=min(scores), maximum=max(scores))
    result = dict(development_pairs=len(rows), profile_clips=curves, limitation='Development diagnostics on four parallel texts. Ratings refer to voice pairs. Assignments share speakers and are not independent replicates.')
    write_json(ROOT / 'results/dev_text_check.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
