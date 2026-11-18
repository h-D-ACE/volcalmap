"""Exact Euclidean distance from a query to a fixed gallery's nearest-neighbor boundary."""
import numpy as np


def boundary_radius(query,gallery):
    q=np.asarray(query,dtype=np.float64);g=np.asarray(gallery,dtype=np.float64)
    if q.ndim!=1 or g.ndim!=2 or len(g)<2 or g.shape[1]!=len(q) or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError('Expected a finite query and at least two gallery points of matching dimension')
    delta=g-q;d2=np.einsum('ij,ij->i',delta,delta)
    first=int(np.argmin(d2));sep=np.linalg.norm(g-g[first],axis=1)
    other=np.arange(len(g))!=first
    if np.any(other & (sep==0)):return first,0.
    margin=(d2[other]-d2[first])/(2*sep[other])
    return first,float(np.min(margin))
