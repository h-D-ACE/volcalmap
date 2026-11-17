"""Untouched child speakers: fixed adult-developed tracker challenge."""
import hashlib
import io
import json
import zipfile
from reference_benchmark import DATA,OUT,parse,VOWELS,formant_track
from formant_tracker import candidates,track
from acoustics import np,read_audio
from run_probe import save_json

def main():
    times=parse(DATA/"timedata.dat",5);formants=parse(DATA/"bigdata.dat",30)
    speakers={g:sorted({n[:3] for n in times if n[0]==g and all(n[:3]+v in times and n[:3]+v in formants for v in VOWELS)})[:6] for g in ("b","g")}
    rows=[]
    with zipfile.ZipFile(DATA/"h95-alldata.zip") as z:
        with zipfile.ZipFile(io.BytesIO(z.read("kids.zip"))) as zz:
            names={n.rsplit("/",1)[-1].lower():n for n in zz.namelist()}
            for ss in speakers.values():
                for s in ss:
                    for v in VOWELS:
                        name=s+v;b=zz.read(names[name+".wav"]);path=DATA/"audio"/(name+".wav");path.write_bytes(b)
                        t=times[name];start,end=t[0]/1000,t[1]/1000
                        refs=np.array(formants[name][5:]).reshape(8,3)[1:]
                        rt=start+(end-start)*np.arange(2,9)/10
                        y,sr=read_audio(path)
                        grid=np.arange(start+.1*(end-start),end-.1*(end-start)+.0025,.005)
                        ff,bb=candidates(y,sr,grid);pathf,states=track(ff,bb,1.)
                        pred=np.array([np.interp(rt,grid,pathf[:,j]) for j in range(3)]).T
                        fixed=formant_track(y,sr,dict(ceiling=5500,window=.04),rt)
                        for method,estimate in (("fixed40",fixed),("path",pred)):
                            valid=refs>0;err=abs(estimate-refs)/np.maximum(refs,1);err=np.where(np.isfinite(err),err,1.)
                            rows.append(dict(id=name,speaker=s,vowel=v,method=method,sha256=hashlib.sha256(b).hexdigest(),
                                median=float(np.median(err[valid])),p90=float(np.percentile(err[valid],90)),within10=float((err[valid]<=.1).mean()),
                                estimate=estimate.tolist(),truth=refs.tolist(),times=rt.tolist()))
    metrics={}
    for method in ("fixed40","path"):
        rr=[r for r in rows if r["method"]==method]
        metrics[method]=dict(n=len(rr),speakers=len({r["speaker"] for r in rr}),median=float(np.median([r["median"] for r in rr])),
            p90_of_p90=float(np.percentile([r["p90"] for r in rr],90)),within10=float(np.mean([r["within10"] for r in rr])))
    save_json(OUT/"tracker_child_challenge.json",dict(selection="First six complete boys and girls, all six vowels; none used in development",strength=1.,metrics=metrics,rows=rows))
    print(metrics,flush=True)

if __name__=="__main__":main()
