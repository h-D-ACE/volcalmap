"""Summaries of controlled repeated-person studies; preserve per-repeat means and denominators."""
import json
import numpy as np
from runtime import ROOT,write_json


def read(directory,file):
    return json.loads((ROOT/'runs'/directory/file).read_text())


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),repeat_means=a.reshape(5,-1).mean(1).tolist(),n=int(a.size))


def main():
    budget={};listwise={};psd={};cov={};distill={}
    for repeat in range(5):
        for fold in range(5):
            bud=read(f'annotation_budget_{repeat}',f'{fold}.json')
            for row in bud['trials']:
                key=f"{row['people']}voices_{row['ratings']}votes"
                budget.setdefault(key,[]).append(row['outer']['single_srcc'])
            budget.setdefault('pca16_unlabeled',[]).append(bud['unlabeled']['single_srcc'])
            lis=read(f'listwise_{repeat}',f'{fold}.json')
            for key,value in lis.items():
                row=value if key.endswith('fixed') else value['outer_gallery']
                for m in ('hit3','ndcg3','regret'):
                    listwise.setdefault(key+'_'+m,[]).append(row[m])
            p=read(f'psd_metric_{repeat}',f'{fold}.json')
            for kind,choice in p['choices'].items():
                psd.setdefault(kind,[]).append(p['candidates'][choice]['outer'])
            c=read(f'covariance_geometry_{repeat}',f'{fold}.json')
            for k,choice in [('mean','mean'),('chosen',c['choice']),('diag',c['diagonal_choice']),('full',c['full_choice'])]:
                cov.setdefault(k,[]).append(c['candidates'][choice]['outer']['single_srcc'])
            for pool in ('mean','attention'):
                for recipe in ('pca','gram'):
                    f=read(f'refstudent_{repeat}_{pool}_{recipe}_2000',f'{fold}.json')
                    for sec,z in f['outer'].items():
                        distill.setdefault(pool+'_'+recipe+'_'+sec+'s',[]).append(z['single_srcc'])
                    if pool=='mean' and recipe=='pca':distill.setdefault('voxsim_teacher_full',[]).append(f['reference_full']['single_srcc'])
    out={k:{name:stats(a) for name,a in values.items()} for k,values in [('annotation_budget',budget),('gallery_loss',listwise),
        ('quadratic_metric',psd),('frame_covariance',cov),('reference_distillation',distill)]}
    write_json(ROOT/'results/final_study_summary.json',out)
    print({k:{s:round(a['mean'],3) for s,a in v.items()} for k,v in out.items()},flush=True)


if __name__=='__main__':main()
