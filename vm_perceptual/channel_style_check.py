"""Check whether device robustness preserves natural voice changes and single-file geometry."""
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from dataset import load_data
from runtime import ROOT, file_hash, write_json
from ssl_rank_student import SSLMap


def waveform(path):
    audio, sr = sf.read(path, dtype='float32')
    factor = math.gcd(sr, 16000)
    return torch.from_numpy(resample_poly(audio, 16000//factor, sr//factor).astype('float32')).cuda()


def style_statistics(z, names, clips):
    per_voice = []
    for s in names:
        normal, falset = clips[s]
        dist = lambda a,b: float(np.linalg.norm(z[a] - z[b]))
        normal_change = np.median([dist(a,b) for i,a in enumerate(normal) for b in normal[i+1:]])
        different = np.median([dist(a,b) for a in normal for b in falset])
        per_voice.append(dict(speaker=s, normal=float(normal_change), style=float(different),
                              falset_repeat=dist(*falset), delta=float(different-normal_change)))
    deltas = np.array([x['delta'] for x in per_voice])
    rng = np.random.default_rng(778)
    sample = np.median(rng.choice(deltas, size=(5000,len(deltas))), axis=1)
    return dict(median_normal=float(np.median([r['normal'] for r in per_voice])),
                median_normal_falset=float(np.median([r['style'] for r in per_voice])),
                median_paired_delta=float(np.median(deltas)), people_positive=int(sum(deltas>0)),
                paired_speaker_interval=np.quantile(sample,[.025,.975]).tolist(),
                per_voice=per_voice)


@torch.inference_mode()
def main():
    data=load_data()
    names=[s for s in data['speakers'] if data['manifest']['split'][s]=='test']
    clips={s:(sorted(data['clips'][s]), sorted(f for f in data['features'] if f.startswith(s+'/falset10/'))) for s in names}
    paths=[path for s in names for group in clips[s] for path in group]
    assert len(set(paths))==len(paths)==120 and all(len(f)==2 and len(n)==4 for n,f in clips.values())
    waves={path:waveform(ROOT/'data_jvs'/path) for path in paths}
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=SSLMap('frozen').cuda().eval()
    results={}
    for method in ('unaugmented','consistency'):
        dest=ROOT/'runs'/f'robust_ssl_{method}_177_1800/best.pt'
        ck=torch.load(dest,weights_only=True,map_location='cpu')
        for key,param in model.named_parameters():
            if param.requires_grad:param.copy_(ck['state'][key])
        vectors={}
        def run(items):
            lengths=torch.tensor([len(waves[p]) for p in items],device='cuda')
            audio=torch.nn.utils.rnn.pad_sequence([waves[p] for p in items],batch_first=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(audio,lengths).cpu().numpy()
        for path in paths:vectors[path]=run([path])[0]
        # Deliberately mix durations within batches; all original audio samples stay the same.
        rng=np.random.default_rng(249)
        shuffled=rng.permutation(paths).tolist()
        batch_vectors={}
        for i in range(0,len(paths),4):
            group=shuffled[i:i+4]
            batch_vectors.update(zip(group,run(group)))
        single=np.stack([vectors[p] for p in paths])
        batched=np.stack([batch_vectors[p] for p in paths])
        sim=single@single.T; padded=batched@single.T
        owners=[p.split('/')[0] for p in paths]
        exclude=np.equal.outer(owners,owners)
        sim[exclude]=-np.inf;padded[exclude]=-np.inf
        n1=sim.argmax(1);n2=padded.argmax(1)
        drift=np.linalg.norm(single-batched,axis=1)
        results[method]=dict(style=style_statistics(vectors,names,clips), model_sha256=file_hash(dest),
            single_vs_batch=dict(median_l2=float(np.median(drift)),max_l2=float(max(drift)),
                                 cross_speaker_top1_equal=float(np.mean(n1==n2))))
    matched=sum(Path(a).name==Path(b).name for n,f in clips.values() for a in n for b in f)
    report=dict(speakers=len(names),clips=len(paths),matched_text_style_pairs=matched,results=results,
                limitation='Falsetto is a natural style category, not a human distance label; text can change.')
    write_json(ROOT/'results/channel_style_check.json',report)
    print({k:(r['style']['median_paired_delta'],r['single_vs_batch']) for k,r in results.items()},flush=True)


if __name__=='__main__':
    main()
