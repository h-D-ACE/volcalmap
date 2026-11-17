"""Model-independent product contract: timbre-first, optional prosody matching.

This module does not pretend acoustic probes or an untrained network are a
validated voice map. A future approved encoder supplies versioned coordinates.
"""
from dataclasses import dataclass
from math import isfinite,sqrt

@dataclass(frozen=True)
class VoicePoint:
    sample_id:str
    model_version:str
    timbre:tuple[float,...]
    prosody:tuple[float,...]|None=None
    status:str="valid"

    def __post_init__(self):
        if not self.timbre or any(not isfinite(v) for v in self.timbre):raise ValueError("Invalid timbre coordinates")
        if self.prosody is not None and (not self.prosody or any(not isfinite(v) for v in self.prosody)):raise ValueError("Invalid prosody coordinates")

def distance(a,b):
    if len(a)!=len(b):raise ValueError("Coordinate dimension mismatch")
    return sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))

def search(query,gallery,*,match_prosody=False,timbre_radius=None,timbre_top_k=20,limit=10):
    if query.status!="valid":raise ValueError("Query has insufficient evidence")
    if timbre_top_k<1 or limit<1:raise ValueError("Positive result counts required")
    if timbre_radius is not None and (not isfinite(timbre_radius) or timbre_radius<0):raise ValueError("Invalid radius")
    if match_prosody and query.prosody is None:raise ValueError("Query prosody unavailable")
    hits=[]
    for p in gallery:
        if p.model_version!=query.model_version:raise ValueError("Mixed model versions cannot share coordinates")
        if p.status!="valid":continue
        dt=distance(query.timbre,p.timbre)
        if timbre_radius is not None and dt>timbre_radius:continue
        hits.append(dict(sample_id=p.sample_id,timbre_distance=dt,prosody_distance=None,point=p))
    hits.sort(key=lambda r:(r["timbre_distance"],r["sample_id"]))
    hits=hits[:timbre_top_k]
    if match_prosody:
        hits=[h for h in hits if h["point"].prosody is not None]
        for h in hits:h["prosody_distance"]=distance(query.prosody,h["point"].prosody)
        hits.sort(key=lambda r:(r["prosody_distance"],r["timbre_distance"],r["sample_id"]))
    return [{k:v for k,v in h.items() if k!="point"} for h in hits[:limit]]

def validate_contract():
    q=VoicePoint("query","test-only",(0.,0.),(1.,0.))
    gallery=[VoicePoint("near_other_style","test-only",(.1,0.),(0.,1.)),
             VoicePoint("near_same_style","test-only",(.2,0.),(1.,0.)),
             VoicePoint("far_same_style","test-only",(5.,5.),(1.,0.))]
    assert search(q,gallery,timbre_top_k=2)[0]["sample_id"]=="near_other_style"
    matched=search(q,gallery,match_prosody=True,timbre_top_k=2)
    assert matched[0]["sample_id"]=="near_same_style"
    assert all(h["sample_id"]!="far_same_style" for h in matched)
    assert distance(gallery[1].prosody,gallery[2].prosody)==0
    assert all(h["prosody_distance"] is None for h in search(q,gallery))
    print("Contract checks passed: timbre gate, style switch, cross-timbre shared style coordinates")

if __name__=="__main__":validate_contract()
