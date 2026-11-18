"""Compare a development-selected waveform student to the fixed map on matched clips."""
import numpy as np
import torch
from runtime import ROOT, write_json
from wave_student import WaveMap, batch, load_records
from train_map import correlations


@torch.inference_mode()
def main():
    data, records, by = load_records()
    model = WaveMap().cuda().eval()
    tag = 'wave_431_noise'
    ck = torch.load(ROOT/'runs'/tag/'best.pt', weights_only=True)
    model.load_state_dict(ck['state'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {'student': tag, 'step': ck['step'], 'parameters': ck['params'], 'groups': {}}
    for group in ('dev', 'test'):
        ids = [i for i, rec in enumerate(records) if rec['split'] == group]
        pairs = [p for p in data['pairs'] if p['split'] == group]
        lookup = {i: j for j, i in enumerate(ids)}
        cache = {}
        for i in ids:
            s = data['speakers'][records[i]['speaker']]
            if s not in cache:
                with np.load(ROOT/'features/crop_records'/f'{s}.npz') as f:
                    cache[s] = {(str(n), int(c[0])): z for n,c,z in zip(f['names'], f['condition'], f['prior']) if c[1] == .5}
        results = {}
        for sec in (0,1,2,4):
            base = []
            for i in ids:
                rec = records[i]
                key = (rec['name'], sec if len(rec['wave']) > sec*16000 else 0)
                z = cache[data['speakers'][rec['speaker']]].get(key)
                base.append(np.full(16,np.nan) if z is None else z)
            base = np.array(base)
            student = np.concatenate([model(*batch(records, ids[k:k+24], sec)).cpu().numpy() for k in range(0,len(ids),24)])
            row = {'matched': int(np.isfinite(base).all(1).sum()), 'total': len(ids)}
            for name, emb in [('map',base), ('student',student)]:
                scores = []
                for a in range(4):
                    for b in range(4):
                        if a == b: continue
                        ia = [lookup[by[p['a']][a]] for p in pairs]
                        ib = [lookup[by[p['b']][b]] for p in pairs]
                        valid = np.isfinite(base[ia]).all(1) & np.isfinite(base[ib]).all(1)
                        subset = [p for p,v in zip(pairs, valid) if v]
                        pred = -np.linalg.norm(emb[ia][valid]-emb[ib][valid],axis=1)
                        scores.append(correlations(subset,pred)['mean_spearman'])
                row[name] = float(np.mean(scores))
            results[str(sec)] = row
        out['groups'][group] = results
    write_json(ROOT/'results/wave_comparison.json', out)
    print(out, flush=True)


if __name__ == '__main__':
    main()
