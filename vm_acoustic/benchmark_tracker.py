"""Select path strength on annotated development set; report exploratory test."""
import json
from acoustics import ROOT,np,read_audio
from formant_tracker import candidates,track
from run_probe import save_json

def main():
    data=ROOT/"data_reference";out=ROOT/"results_reference"
    rows=json.loads((data/"subset.json").read_text())["rows"]
    results=[]
    for r in rows:
        audio,sr=read_audio(data/r["path"])
        times=np.array(r["reference_times"])[1:]
        # Track at 5ms, then interpolate to original reference points.
        grid=np.arange(r["start"]+.1*r["duration"],r["end"]-.1*r["duration"]+.0025,.005)
        ff,bb=candidates(audio,sr,grid)
        for strength in (.1,1.,10.):
            tracked,states=track(ff,bb,strength)
            pred=np.array([np.interp(times,grid,tracked[:,j]) for j in range(3)]).T
            ref=np.array(r["reference_formants"])[1:]
            valid=ref>0
            err=abs(pred-ref)/np.maximum(ref,1)
            err=np.where(np.isfinite(err),err,1.)
            results.append(dict(id=r["id"],split=r["split"],vowel=r["vowel"],strength=strength,
                median=float(np.median(err[valid])),p90=float(np.percentile(err[valid],90)),
                within10=float((err[valid]<=.1).mean()),estimate=pred.tolist(),states=states.tolist()))
    scores={str(s):float(np.mean([r["median"]+.5*r["p90"] for r in results if r["split"]=="dev" and r["strength"]==s])) for s in (.1,1.,10.)}
    chosen=float(min(scores,key=scores.get))
    metric={}
    for split in ("dev","test"):
        rr=[r for r in results if r["split"]==split and r["strength"]==chosen]
        metric[split]=dict(median=float(np.median([r["median"] for r in rr])),p90_of_p90=float(np.percentile([r["p90"] for r in rr],90)),within10=float(np.mean([r["within10"] for r in rr])))
    save_json(out/"tracker_experiment.json",dict(chosen_strength=chosen,development_scores=scores,metrics=metric,rows=results,
        note="Test speakers were previously inspected for other algorithms; this is exploratory, not a new blind validation"))
    print(scores,chosen,metric,flush=True)

if __name__=="__main__":main()
