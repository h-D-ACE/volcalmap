"""Size an exploratory neighbor set using cross-text stability on development voices."""
import itertools
import json
import math
import numpy as np
from dataset import load_data
from runtime import ROOT, write_json


def load_vectors(names, method):
    cache = {}
    for speaker in names:
        if method == 'own':
            cache[speaker] = np.load(ROOT / 'features/rank_map_own' / f'{speaker}.npy')
        else:
            with np.load(ROOT / 'features/gallery_vectors' / f'{speaker}.npz') as data:
                cache[speaker] = data[method]
    return cache


def voice_queries(query, gallery, scores):
    outputs = {}
    for count in (1, 2, 3):
        rows = []
        for slots in itertools.combinations(range(4), count):
            point = query[list(slots)].mean(0)
            distances = np.linalg.norm(gallery - point, axis=1)
            order = np.argsort(distances, kind='stable')
            excess = distances - distances[order[0]]
            repeats = [query[i] for i in range(4) if i not in slots]
            ref = [np.linalg.norm(gallery - v, axis=1).argmin() for v in repeats]
            top_score = scores[order[0]]
            rows.append((excess, ref, top_score >= np.sort(scores)[-3]))
        outputs[count] = rows
    return outputs


def evaluate(dev, test):
    result = {}
    for count in (1, 2, 3):
        # Unit of calibration is one person. Protect every alternate sentence among four.
        maximum = sorted(max(float(e[i]) for e, ref, _ in v[count] for i in ref) for v in dev)
        index = math.ceil(.8*(len(maximum)+1))-1
        threshold = maximum[index]
        by_voice = []
        for v in test:
            rows = v[count]
            covered = [e[i] <= threshold for e, ref, _ in rows for i in ref]
            same = [e[i] == 0 for e, ref, _ in rows for i in ref]
            sizes = [np.count_nonzero(e <= threshold) for e, _, _ in rows]
            by_voice.append((all(covered),np.mean(covered),np.mean(same),np.mean(sizes),
                             np.mean([hit for _,_,hit in rows])))
        metrics = np.mean(by_voice,axis=0)
        result[str(count)] = dict(calibration_threshold=threshold, test_people=len(test),
            all_alternates_covered=float(metrics[0]), per_alternate_coverage=float(metrics[1]),
            same_first_as_alternate=float(metrics[2]), mean_candidates=float(metrics[3]),
            first_hits_human_top3=float(metrics[4]))
    return result


def main():
    data = load_data()
    names = data['speakers']
    gallery = json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    labels = {}
    for row in data['pairs']:
        a, b, y = row['speaker_a'], row['speaker_b'], row['score']
        labels[a,b] = labels[b,a] = y
    report = {}
    for method in ('current','student','own','voxsim'):
        cache = load_vectors(names, method)
        rows = {group:[] for group in ('dev','test','extra')}
        for s in names:
            split = data['manifest']['split'][s]
            if split not in ('dev','test'):continue
            ids = gallery[data['manifest']['gender'][s]]
            references = np.stack([cache[t][4:8].mean(0) for t in ids])
            ratings = np.array([labels[s,t] for t in ids])
            # Use the same four query text IDs for development and test; all differ from reference IDs.
            rows[split].append(voice_queries(cache[s][:4],references,ratings))
            if split=='test' and len(cache[s])>=12:
                rows['extra'].append(voice_queries(cache[s][8:12],references,ratings))
        report[method] = {split:evaluate(rows['dev'],rows[split]) for split in ('test','extra')}
    write_json(ROOT/'results/repeat_neighbor_sets.json',dict(gallery_size=10,design_coverage=.8,
        description='Empirical coverage on two sets of texts. Models used development data previously; no distribution-free guarantee.',
        distance='Euclidean distance to the arithmetic mean reference. May differ from dot-product gallery ranking.',results=report))
    print({k:v['test'] for k,v in report.items()}, flush=True)


if __name__=='__main__':main()
