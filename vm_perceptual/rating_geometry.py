"""Audit listener disagreement and Euclidean feasibility, not an audio model."""
import json
from runtime import ROOT,write_json
import numpy as np
from scipy.stats import spearmanr

def main():
    result={};rng=np.random.default_rng(20261117)
    for gender in ('male','female'):
        raw=json.loads((ROOT/f'data_sources/jvs_{gender}_raw_similarity_data.json').read_text())
        people=sorted({s for key in raw for s in key.split('-')});index={s:i for i,s in enumerate(people)}
        S=np.eye(len(people));variances=[];ratings=[]
        for key,rr in raw.items():
            a,b=key.split('-');S[index[a],index[b]]=S[index[b],index[a]]=np.mean(rr)/3
            ratings.append(np.array(rr,dtype=float));variances.append(np.var(rr,ddof=1))
        eig=np.linalg.eigvalsh(S)
        agreements=[]
        for _ in range(100):
            left=[];right=[]
            for rr in ratings:
                rr=rng.permutation(rr);mid=len(rr)//2;left.append(rr[:mid].mean());right.append(rr[mid:].mean())
            agreements.append(float(spearmanr(left,right).statistic))
        result[gender]=dict(speakers=len(people),pairs=len(raw),ratings=sum(len(r) for r in ratings),
            median_rating_sd=float(np.sqrt(np.median(variances))),split_half_srcc_median=float(np.median(agreements)),
            minimum_gram_eigenvalue=float(eig[0]),negative_gram_eigenvalues=int(np.sum(eig< -1e-8)),
            negative_eigen_mass_fraction=float(-eig[eig<0].sum()/np.sum(abs(eig))))
    result['interpretation']=['Mean scores divided by 3 plus unit diagonal are not automatically a positive semidefinite cosine Gram matrix.',
        'Negative eigenvalues do not disprove a perceptual map: rating scale may be nonlinear, listener noise and context matter.',
        'Split-half agreement uses independent rating subsets per pair; listener IDs absent, not a universal human ceiling.',
        'Male/female graphs disconnected: cross-gender geometry lacks supervision and must not be claimed calibrated.']
    write_json(ROOT/'results/rating_geometry.json',result);print(result)

if __name__=='__main__':main()
