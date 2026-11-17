"""Fixed-model recording map. Retrieval always uses the full timbre vector."""
import argparse
import json
from pathlib import Path
from runtime import ROOT, file_hash, write_json
import numpy as np


def search(index, query, k=5, include_self=False):
    if index['model_version'] != query['model_version']:
        raise ValueError('Query and map use different models; rebuild or re-encode')
    if k < 1:
        raise ValueError('k must be positive')
    q = np.asarray(query['timbre']['vector'], dtype=float)
    if q.shape != (index['dimensions'],) or not np.isfinite(q).all():
        raise ValueError('Invalid query vector')
    matches = []
    for record in index['records']:
        if not include_self and record['sample_id'] == query['sample_id']:
            continue
        v = np.asarray(record['vector'], dtype=float)
        if v.shape != q.shape or not np.isfinite(v).all():
            raise ValueError('Invalid map vector')
        matches.append(dict(record_id=record['record_id'], sample_id=record['sample_id'],
                            distance=float(np.linalg.norm(v-q)), display=record['display']))
    matches.sort(key=lambda row: (row['distance'], row['record_id']))
    return dict(model_version=index['model_version'], research_only=True,
                metric='Euclidean distance in full timbre coordinates',
                display_projection_version=index['projection_version'],
                self_matches_excluded=not include_self, matches=matches[:k],
                limitation='Neighbor rank is experimental perceptual similarity, not identity or calibrated probability')


def build(features_path, checkpoint, projection):
    import torch
    from infer import encode
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    view = json.loads(Path(projection).read_text(encoding='utf-8'))
    version = file_hash(checkpoint)
    if view['checkpoint_sha256'] != version:
        raise ValueError('Projection belongs to a different model')
    features = json.loads(Path(features_path).read_text(encoding='utf-8'))
    records = []
    for name, feature in sorted(features.items()):
        z = encode(feature, ck)
        display = ((z-view['mean']) @ np.asarray(view['axes'][:3]).T).tolist()
        records.append(dict(record_id=name, sample_id=feature['sha256'],
                            vector=z.tolist(), display=display))
    if not records:
        raise ValueError('Empty map')
    return dict(model_version=version, projection_version=file_hash(projection),
                dimensions=len(records[0]['vector']), research_only=True,
                feature_source_sha256=file_hash(features_path), records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--features', default=str(ROOT/'features/jvs_features.json'))
    b.add_argument('--checkpoint', default=str(ROOT/'runs/selected.pt'))
    b.add_argument('--projection', default=str(ROOT/'results/projection.json'))
    b.add_argument('--output', default=str(ROOT/'results/map_index.json'))
    s = sub.add_parser('search')
    s.add_argument('query')
    s.add_argument('--index', default=str(ROOT/'results/map_index.json'))
    s.add_argument('--output', default=str(ROOT/'results/neighbors.json'))
    s.add_argument('--k', type=int, default=5)
    s.add_argument('--include-self', action='store_true')
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.features, args.checkpoint, args.projection)
    else:
        result = search(json.loads(Path(args.index).read_text(encoding='utf-8')),
                        json.loads(Path(args.query).read_text(encoding='utf-8')),
                        args.k, args.include_self)
    write_json(args.output, result)
    print(json.dumps(dict(output=args.output, records=len(result.get('records', [])),
                          matches=result.get('matches', [])), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
