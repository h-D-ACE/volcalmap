"""Compare deployed singleton students on the same natural voice and gallery endpoints."""
import json
import numpy as np
from dataset import load_rating_graph
from gallery_benchmark import quality
from reference_benchmark import benchmark
from runtime import ROOT,write_json
from wave_map_cli import WaveEncoder


def main():
    data=load_rating_graph()
    split=data['manifest']['split'];gender=data['manifest']['gender']
    gallery=json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    targets={s for group in gallery.values() for s in group}
    encoders={k:WaveEncoder(ROOT/'runs'/run/'best.pt','cuda',k) for k,run in (
        ('robust_student','robust_ssl_consistency_177_1800'),('unaugmented_student','robust_ssl_unaugmented_177_1800'))}
    labels={}
    for r in data['pairs']:
        a,b=r['speaker_a'],r['speaker_b'];labels[a,b]=labels[b,a]=r['score']
    vectors={k:{} for k in ('current',*encoders)}
    for s in data['speakers']:
        if split[s]=='train' and s not in targets:continue
        with np.load(ROOT/'features/gallery_vectors'/f'{s}.npz') as old:
            files=old['files'].tolist();base=old['current']
        for k in vectors:vectors[k][s]={}
        slots=range(len(files)) if split[s]=='test' else range(4,8)
        for i in slots:
            vectors['current'][s][i]=base[i]
            path=ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit'/files[i]
            for k,encoder in encoders.items():vectors[k][s][i]=encoder.vector(path)
    report={}
    for stage,start,stop in [('dev',4,8),('test',0,4),('fresh',4,12)]:
        group='dev' if stage=='dev' else 'test'
        people=[s for s in data['speakers'] if split[s]==group and stop-1 in vectors['current'][s]]
        rows={k:{s:np.stack([v[s][i] for i in range(start,stop)]) for s in people} for k,v in vectors.items()}
        scores=benchmark(people,data,rows,2762)
        if group=='test':
            windows=range(0,4) if stage=='test' else range(8,12)
            for k,bank in vectors.items():
                values=[]
                for s in people:
                    group_ids=gallery[gender[s]]
                    refs=np.stack([np.mean([bank[g][i] for i in range(4,8)],axis=0) for g in group_ids])
                    y=np.array([labels[s,g] for g in group_ids])
                    values.extend(quality(bank[s][i]@refs.T,y) for i in windows)
                scores['models'][k]['gallery_hit3']=float(np.mean([r['hit3'] for r in values]))
        report[stage]=scores
    write_json(ROOT/'results/robust_retrieval_check.json',dict(groups=report,versions={k:e.version for k,e in encoders.items()},
        scope='Deployed single-file float32 path. Same existing JVS labels. Fresh gallery is restricted to last four texts to exclude gallery reference texts.'))
    for group,block in report.items():
        print(group,{k:(round(v['single']['srcc'],4),v.get('gallery_hit3')) for k,v in block['models'].items()},flush=True)


if __name__=='__main__':main()
