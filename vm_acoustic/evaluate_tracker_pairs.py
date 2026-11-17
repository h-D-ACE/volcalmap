"""Frozen path tracker on all preselected paired speech, no quality rejection."""
import json
from collections import defaultdict,Counter
from acoustics import ROOT,np,read_audio
from formant_tracker import CandidateBank,track
from run_probe import save_json,csv_write,summary

OUT=ROOT/"results_sentence_validation";DATA=ROOT/"data_vctk_validation"

def aggregate(bank,t,source,shift):
    duration=t["end"]-t["start"]
    grid=np.linspace(t["start"]+.1*duration+shift,t["end"]-.1*duration+shift,max(3,int(duration*.8/.005)+1))
    ff,bb=bank.at(grid);f,states=track(ff,bb,1.)
    times=np.array(source.get("times",[]))
    values=source["values"].copy()
    if len(times)<3:return values,False
    for j in range(3):
        vals=np.interp(times,grid,f[:,j])
        vals=vals[np.isfinite(vals)&(vals>0)]
        if len(vals)<3:values[j+1]=None;continue
        vals=np.sort(np.log(vals));cut=int(len(vals)*.1)
        values[j+1]=float(np.mean(vals[cut:len(vals)-cut] if cut else vals))
    return values,all(v is not None for v in values)

def main():
    data=json.loads((OUT/"measurements.json").read_text());scale=np.array(json.loads((OUT/"calibration.json").read_text())["scale"]["scale"])
    rows=[];coverage=defaultdict(Counter)
    for p in data["pairs"]:
        a,sr=read_audio(DATA/f"{p['speaker']}_{p['utterance']}_mic1.flac");b,sr2=read_audio(DATA/f"{p['speaker']}_{p['utterance']}_mic2.flac")
        ba,bb=CandidateBank(a,sr),CandidateBank(b,sr2)
        for i,t in enumerate(p["tokens"]):
            c=coverage[p["group"]];c["candidate"]+=1
            sa,sb=t["reference_selected"]["a"],t["reference_selected"]["b"]
            va,ga=aggregate(ba,t,sa,0.);vb,gb=aggregate(bb,t,sb,p["lag"])
            original=all(sa["mask"]+sb["mask"]);good=ga and gb
            c["fixed_complete"]+=int(original);c["path_complete"]+=int(good)
            if not(original and good):continue
            c["common"]+=1
            rows.append(dict(speaker=p["speaker"],utterance=p["utterance"],group=p["group"],index=i,vowel=t["vowel"],
                 long=t["end"]-t["start"]>=.18,
                 fixed=float(np.linalg.norm((np.array(sb["values"])-sa["values"])/scale)/np.sqrt(6)),
                 path=float(np.linalg.norm((np.array(vb)-va)/scale)/np.sqrt(6))))
        print("path",p["speaker"],p["utterance"],flush=True)
    metrics={}
    for stratum in ("all","long"):
        metrics[stratum]={}
        for g in coverage:
            bucket=defaultdict(list)
            for r in rows:
                if r["group"]==g and (stratum=="all" or r["long"]):bucket[r["speaker"],r["utterance"]].append(r)
            metrics[stratum][g]={k:summary([float(np.median([r[k] for r in rr])) for rr in bucket.values()]) for k in ("fixed","path")}
    save_json(OUT/"tracker_metrics.json",dict(metrics=metrics,coverage={g:dict(c) for g,c in coverage.items()},strength=1.,
        note="Frozen adult-development strength. Existing VCTK evaluation previously inspected; method comparison is exploratory."))
    csv_write(OUT/"tracker_distances.csv",rows)
    print(metrics,flush=True)

if __name__=="__main__":main()
