"""One locked held-out evaluation, node resampling, dimensions, and style challenge."""
import argparse,json,hashlib
from pathlib import Path
from runtime import ROOT,write_json,file_hash
from dataset import load_data,make_input,pca_transform,unit
from train_map import MetricHead,correlations
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import spearmanr

def score(z,rows,alpha,beta):
    d2=np.array([np.sum((z[r['a']]-z[r['b']])**2) for r in rows])
    return 3*np.tanh(alpha-beta*d2)

def baseline_fit(z,rows):
    target=np.array([r['score'] for r in rows])
    fit=least_squares(lambda ab:score(z,rows,ab[0],np.exp(ab[1]))-target,[.8,0.],bounds=([-5,-5],[5,5]),loss='huber')
    return float(fit.x[0]),float(np.exp(fit.x[1]))

def boot_ci(rows,preds,repeats=2000,reference=None):
    # Resample SPEAKER NODES separately within gender, then induce weighted edges.
    rng=np.random.default_rng(20261117);samples=[];differences=[]
    for _ in range(repeats):
        vals=[];refvals=[]
        for gender in ('male','female'):
            rr=[(i,r) for i,r in enumerate(rows) if r['gender']==gender]
            nodes=sorted({r[k] for _,r in rr for k in ('a','b')})
            chosen=rng.choice(nodes,len(nodes),replace=True);counts={n:int(np.sum(chosen==n)) for n in nodes}
            y=[];p=[];rp=[]
            for i,r in rr:
                w=counts[r['a']]*counts[r['b']]
                y.extend([r['score']]*w);p.extend([float(preds[i])]*w)
                if reference is not None:rp.extend([float(reference[i])]*w)
            if len(y)>=3 and np.std(y)>0 and np.std(p)>0:
                vals.append(float(spearmanr(y,p).statistic))
                if reference is not None and np.std(rp)>0:refvals.append(float(spearmanr(y,rp).statistic))
        if len(vals)==2:
            samples.append(float(np.mean(vals)))
            if len(refvals)==2:differences.append(float(np.mean(vals)-np.mean(refvals)))
    return dict(interval=np.percentile(samples,[2.5,97.5]).tolist(),valid_resamples=len(samples),
        difference_vs_locked_baseline=np.percentile(differences,[2.5,97.5]).tolist() if differences else None,
        method='stratified speaker-node bootstrap, induced edge multiplicity; descriptive small-sample interval')

def geometry(z,train_mask,test_mask):
    tr=z[train_mask];mu=tr.mean(0);_,sv,axes=np.linalg.svd(tr-mu,full_matrices=False)
    ev=sv**2;effective=float(ev.sum()**2/np.sum(ev**2)) if np.sum(ev**2)>0 else 0.
    q=z[test_mask];full=np.linalg.norm(q[:,None]-q[None,:],axis=2);np.fill_diagonal(full,np.inf)
    n=len(q);k=min(5,n-1);truth=np.argsort(full,axis=1)[:,:k];i,j=np.triu_indices(n,1)
    views={}
    for dim in (1,2,3,5,8,16):
        if dim>axes.shape[0] or dim>z.shape[1]:continue
        p=(q-mu)@axes[:dim].T;dist=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(dist,np.inf)
        near=np.argsort(dist,axis=1)[:,:k]
        views[str(dim)]=dict(neighbor_recall=float(np.mean([len(set(a)&set(b))/k for a,b in zip(truth,near)])),
            distance_spearman=float(spearmanr(full[i,j],dist[i,j]).statistic),
            stress=float(np.sqrt(np.sum((full[i,j]-dist[i,j])**2)/np.sum(full[i,j]**2))))
    return dict(effective_dimension=effective,views=views),dict(mean=mu.tolist(),axes=axes.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--confirm-heldout',action='store_true');a=p.parse_args()
    if not a.confirm_heldout:p.error('Use --confirm-heldout only after locking development selection')
    output=ROOT/'results';output.mkdir(exist_ok=True)
    if (output/'evaluation.json').exists():raise RuntimeError('Held-out already evaluated; archive/version a genuinely new experiment, do not retune this test')
    data=load_data();dev=json.loads((ROOT/'runs/development.json').read_text())
    if dev['data_hash']!=data['provenance']:raise ValueError('Development data mismatch')
    torch.set_num_threads(2)
    rows={s:[r for r in data['pairs'] if r['split']==s] for s in ('train','dev','test')}
    candidates=[];coordinates={};clip_coordinates={};checkpoints={}
    for rep in sorted({r['representation'] for r in dev['candidates']}):
        X,clips,conf=make_input(data,rep)
        for dim in (None,3,8,16):
            z=unit(X if dim is None else pca_transform(X,conf)[:,:dim])
            alpha,beta=baseline_fit(z,rows['train']);name=f'{rep}_baseline_'+('full' if dim is None else str(dim))
            val=correlations(rows['dev'],score(z,rows['dev'],alpha,beta))
            candidates.append(dict(name=name,kind='baseline',dev_spearman=val['mean_spearman'],alpha=alpha,beta=beta))
            coordinates[name]=z
            clip_coordinates[name]={f:unit(v if dim is None else pca_transform(v[None],conf)[0,:dim]) for f,v in clips.items()}
            checkpoints[name]=dict(kind='baseline',edge_policy=data['edge_policy'],representation=rep,preprocessing=conf,pca_dim=dim,alpha=alpha,beta=beta,data_hash=data['provenance'],research_only=True)
    for r in dev['candidates']:
        ck=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True)
        if ck['data_hash']!=data['provenance']:raise ValueError('Checkpoint data mismatch')
        X,clips,conf=make_input(data,r['representation']);xp=pca_transform(X,conf)
        model=MetricHead(ck['input_dim'],ck['output_dim']);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():
            z=model(torch.tensor(xp,dtype=torch.float32)).numpy()
            zz=model(torch.tensor(np.array([pca_transform(v[None],conf)[0] for v in clips.values()]),dtype=torch.float32)).numpy()
        alpha=float(model.alpha.detach());beta=float(torch.exp(model.log_beta.detach().clamp(-5,5)))
        name=r['name'];candidates.append(dict(name=name,kind='trained',dev_spearman=r['dev_spearman'],alpha=alpha,beta=beta))
        coordinates[name]=z;clip_coordinates[name]=dict(zip(clips,zz));checkpoints[name]=dict(ck,kind='trained')
    # Choose including untrained baselines BEFORE examining any held-out score.
    winner=max(candidates,key=lambda r:r['dev_spearman'])
    best_baseline=max((r for r in candidates if r['kind']=='baseline'),key=lambda r:r['dev_spearman'])
    write_json(output/'locked_selection.json',dict(winner=winner,best_baseline=best_baseline,candidates=candidates,data_hash=data['provenance']))
    torch.save(checkpoints[winner['name']],ROOT/'runs/selected.pt')
    results={}
    for r in candidates:
        predictions=score(coordinates[r['name']],rows['test'],r['alpha'],r['beta'])
        results[r['name']]=dict(kind=r['kind'],development_spearman=r['dev_spearman'],test=correlations(rows['test'],predictions))
    selected_z=coordinates[winner['name']];selected_clip=clip_coordinates[winner['name']]
    pred=score(selected_z,rows['test'],winner['alpha'],winner['beta'])
    reference=score(coordinates[best_baseline['name']],rows['test'],best_baseline['alpha'],best_baseline['beta'])
    interval=boot_ci(rows['test'],pred,reference=reference)
    geo,view=geometry(selected_z,data['split']=='train',data['split']=='test')
    # No labels claim that falsetto should be close/far: report measured displacement only.
    repeat=[];falset=[]
    for i,s in enumerate(data['speakers']):
        if data['split'][i]!='test':continue
        for f in data['clips'][s]:
            others=[selected_clip[g] for g in data['clips'][s] if g!=f]
            # Both sides use the SAME clip aggregation convention; no self-inclusion.
            repeat.append(float(np.linalg.norm(selected_clip[f]-unit(np.mean(others,0)))))
        for f,v in selected_clip.items():
            if f.startswith(s+'/falset10/'):
                normal=unit(np.mean([selected_clip[g] for g in data['clips'][s]],0))
                falset.append(dict(speaker=s,path=f,distance_to_normal=float(np.linalg.norm(v-normal))))
    write_json(output/'projection.json',dict(**view,model_version=winner['name'],checkpoint_sha256=file_hash(ROOT/'runs/selected.pt'),evaluation=geo))
    report=dict(winner=winner,best_baseline=best_baseline,models=results,selected_node_bootstrap=interval,projection=geo,
        pair_counts={k:len(v) for k,v in rows.items()},split_counts={s:int(np.sum(data['split']==s)) for s in rows},
        normal_leave_one_clip_out=dict(n=len(repeat),median=float(np.median(repeat)),p90=float(np.percentile(repeat,90))),
        falsetto_challenge=dict(n=len(falset),median=float(np.median([r['distance_to_normal'] for r in falset])),rows=falset,
            note='No human falsetto similarity labels: displacement is diagnostic, neither accuracy nor desired separation'),
        caveats=['Only same-gender normal-voice ratings; cross-gender map distances not validated',
                 'Four clips per normal voice profile; published ratings refer to corpus voice pairs, not these exact clips',
                 'Research-only dataset; not product-ready or a tone-normalization model'],data_hash=data['provenance'])
    write_json(output/'evaluation.json',report)
    pairs=[dict(**r,prediction=float(p)) for r,p in zip(rows['test'],pred)]
    write_json(output/'heldout_predictions.json',pairs)
    lines=['# JVS感知音色坐标：冻结测试报告','','模型选择仅看开发集，候选包含不训练基线。不同人听感评分是目标，不是身份分类。',
        f"选择：{winner['name']}；开发Spearman={winner['dev_spearman']:.4f}。",
        '', '|方法|开发均值SRCC|测试男性SRCC|测试女性SRCC|测试均值SRCC|','|---|---:|---:|---:|---:|']
    for name,m in results.items():
        t=m['test'];lines.append(f"|{name}|{m['development_spearman']:.4f}|{t['male']['spearman']:.4f}|{t['female']['spearman']:.4f}|{t['mean_spearman']:.4f}|")
    lines += ['',f"选定模型按说话者节点重采样的测试均值SRCC 95%区间：{interval['interval']}。只有每性别10位测试人，不应过度解释。",
        f"逐句留一稳定性中位距离：{report['normal_leave_one_clip_out']['median']:.4f}；假声到正常档案中位：{report['falsetto_challenge']['median']:.4f}。假声没有感知真值，这不是成功率。",
        '', '## 显示压缩（保留原高维几何，不代表感知准确率）','', '|维数|5近邻保留|距离秩相关|stress|','|---|---:|---:|---:|']
    for d,m in geo['views'].items():lines.append(f"|{d}|{m['neighbor_recall']:.3f}|{m['distance_spearman']:.3f}|{m['stress']:.3f}|")
    lines+=['','## 范围','',*['- '+s for s in report['caveats']],
        '- 原始语音和特征/权重不进入GitHub源码包；首次独立测试之后不再据此调参。']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('SELECTED',winner,'TEST',results[winner['name']]['test'],'CI',interval,flush=True)

if __name__=='__main__':main()
