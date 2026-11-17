"""Development comparison of within-voice whitening for single-utterance retrieval."""
import itertools
from runtime import ROOT, write_json
from dataset import load_data, make_input, unit
from train_map import correlations
import numpy as np


def main():
    data = load_data()
    _, clips, _ = make_input(data, 'acoustic')
    x = np.asarray([[clips[f] for f in data['clips'][s]] for s in data['speakers']])
    train = x[data['split'] == 'train']
    residual = train - train.mean(axis=1, keepdims=True)
    cov = np.cov(residual.reshape(-1, x.shape[-1]), rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    rows = [r for r in data['pairs'] if r['split'] == 'dev']
    a = np.array([r['a'] for r in rows])
    b = np.array([r['b'] for r in rows])
    results = []
    for strength in (0., .25, .5, .75):
        variance = (1-strength)*values.mean() + strength*values
        z = unit((x @ vectors) / np.sqrt(variance))
        scores = []
        for i, j in itertools.permutations(range(4), 2):
            similarity = -np.linalg.norm(z[a, i] - z[b, j], axis=1)
            scores.append(correlations(rows, similarity)['mean_spearman'])
        dev = z[data['split'] == 'dev']
        drift = [np.linalg.norm(dev[:, i]-dev[:, j], axis=1) for i,j in itertools.combinations(range(4),2)]
        results.append(dict(strength=strength, disjoint_text_srcc=float(np.mean(scores)), median_repeat_distance=float(np.median(drift))))
    write_json(ROOT/'results/dev_whitening.json', dict(results=results, train_voices=len(train), dev_pairs=len(rows)))
    print(results)


if __name__ == '__main__':
    main()
