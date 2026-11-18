"""Independent query encoding for two development-selected distilled models, with fixed gallery tests."""
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from runtime import ROOT,write_json,file_hash
from dataset import load_data
from reference_student import AttentiveMap
from ssl_rank_student import SSLMap
from reference_benchmark import benchmark
from gallery_benchmark import quality


@torch.inference_mode()
def main():
    data=load_data();names=data['speakers']
    lock={}
    for family,modelnames,steps in [('rankstudent',('mean','attention','attention_relation'),3600),('ssl_rank',('frozen','rank4'),1200)]:
        recipes={}
        for method in modelnames:
            per={}
            for seed in (76,177,431):
                rows=json.loads((ROOT/'runs'/f'{family}_{method}_{seed}_{steps}'/'dev.json').read_text())
                if rows[-1]['step']!=steps:raise ValueError('Incomplete candidate training')
                per[seed]=max(row['utility'] for row in rows)
            recipes[method]=per
        choice=max(recipes,key=lambda k:np.mean(list(recipes[k].values())))
        middle=sorted(recipes[choice],key=recipes[choice].get)[1]
        lock[family]=dict(recipe=choice,seed=middle,steps=steps,dev=recipes)
    assert (lock['rankstudent']['recipe'],lock['rankstudent']['seed'])==('attention_relation',76)
    assert (lock['ssl_rank']['recipe'],lock['ssl_rank']['seed'])==('frozen',431)
    models={'rankstudent':AttentiveMap(),'ssl_rank':SSLMap('frozen')}
    for k,model in models.items():
        rec=lock[k];path=ROOT/'runs'/f"{k}_{rec['recipe']}_{rec['seed']}_{rec['steps']}"/'best.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if k=='rankstudent':model.load_state_dict(ck['state'])
        else:
            for name,param in model.named_parameters():
                if param.requires_grad:param.copy_(ck['state'][name])
        model.cuda().eval()
        rec['params']=sum(p.numel() for p in model.parameters());rec['weight_sha256']=file_hash(path)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    all_z={k:{} for k in (*models,'current','own')}
    for s in names:
        with np.load(ROOT/'features/gallery_vectors'/f'{s}.npz') as f:
            all_z['current'][s]=f['current']
        all_z['own'][s]=np.load(ROOT/'features/rank_map_own'/f'{s}.npy')
        files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))
        for k,net in models.items():
            out=ROOT/'features'/f'confirmed_{k}';out.mkdir(exist_ok=True)
            dest=out/f'{s}.npz'
            if not dest.exists():
                vals=[]
                for file in files:
                    wave,sr=sf.read(file,dtype='float32');g=math.gcd(16000,sr)
                    x=torch.from_numpy(resample_poly(wave,16000//g,sr//g).astype('float32')).cuda()[None]
                    n=torch.tensor([x.shape[1]],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=k=='ssl_rank'):vals.append(net(x,n).cpu().numpy()[0])
                stage=dest.with_suffix('.partial.npz')
                np.savez_compressed(stage,names=[p.name for p in files],vectors=vals,version=lock[k]['weight_sha256']);stage.replace(dest)
            with np.load(dest) as f:
                if str(f['version'])!=lock[k]['weight_sha256']:raise ValueError('Cache version mismatch')
                all_z[k][s]=f['vectors']
    gallery=json.loads((ROOT/'results/gallery_benchmark.json').read_text())['gallery']
    ratings={}
    for r in data['pairs']:ratings[r['speaker_a'],r['speaker_b']]=ratings[r['speaker_b'],r['speaker_a']]=r['score']
    result={}
    for key,start,end in [('dev',4,8),('test',0,4),('fresh_test',4,12)]:
        group='dev' if key=='dev' else 'test'
        subset=[s for s in names if data['manifest']['split'][s]==group and len(all_z['current'][s])>=end]
        restricted={k:{s:zz[s][start:end] for s in subset} for k,zz in all_z.items()}
        res=benchmark(subset,data,restricted,25206)
        for k,zz in all_z.items():
            stats=[]
            for s in subset:
                gal=gallery[data['manifest']['gender'][s]];target=np.array([ratings[s,g] for g in gal])
                # Equal evaluation texts for every model. Fresh queries use texts 9..12, gallery 5..8.
                q=zz[s][8:12] if key=='fresh_test' else zz[s][:4]
                gvec=np.stack([zz[g][4:8].mean(0) for g in gal])
                stats.extend(quality(row,target) for row in q@gvec.T)
            res['models'][k]['gallery']={m:float(np.mean([s[m] for s in stats])) for m in stats[0]}
        result[key]=res
    write_json(ROOT/'results/distilled_confirmation.json',dict(selection=lock,groups=result))


if __name__=='__main__':main()
