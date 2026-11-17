"""Prepare disjoint stage-0 data; keep all 16 benchmark speakers out of fitting."""
import hashlib
import io
import json
import zipfile
from reference_benchmark import DATA,parse,VOWELS
from run_probe import save_json

def main():
    times=parse(DATA/"timedata.dat",5);f=parse(DATA/"bigdata.dat",30)
    reference=json.loads((DATA/"subset.json").read_text())["rows"]
    excluded={r["speaker"] for r in reference}
    groups={g:sorted({n[:3] for n in times if n[0]==g and n[:3] not in excluded}) for g in ("m","w")}
    splits={}
    for group,ss in groups.items():
        n=len(ss);nt=int(n*.7);nd=int(n*.15)
        splits.update({s:"train" if i<nt else "dev" if i<nt+nd else "test" for i,s in enumerate(ss)})
    rows=[];missing=[]
    with zipfile.ZipFile(DATA/"h95-alldata.zip") as z:
        for group,nested in (("m","men.zip"),("w","women.zip")):
            with zipfile.ZipFile(io.BytesIO(z.read(nested))) as zz:
                names={n.rsplit("/",1)[-1].lower():n for n in zz.namelist()}
                for speaker in groups[group]:
                    for vowel in VOWELS:
                        name=speaker+vowel
                        if name not in times or name not in f or name+".wav" not in names:
                            missing.append(name);continue
                        b=zz.read(names[name+".wav"])
                        (DATA/"audio"/(name+".wav")).write_bytes(b)
                        vals=f[name];tt=times[name]
                        rows.append(dict(id=name,speaker=speaker,vowel=vowel,split=splits[speaker],path="audio/"+name+".wav",
                            sha256=hashlib.sha256(b).hexdigest(),start=tt[0]/1000,end=tt[1]/1000,
                            reference_formants=[vals[5+i*3:8+i*3] for i in range(8)]))
    manifest=dict(stage="physical_formant_measurement_only",source=json.loads((DATA/"source.json").read_text()),
        benchmark_speakers_excluded=sorted(excluded),selection="All remaining adult speakers, six fixed vowels; sorted speaker 70/15/15 split within group",
        missing=missing,rows=rows)
    save_json(DATA/"training_manifest.json",manifest)
    counts={s:dict(speakers=len({r['speaker'] for r in rows if r['split']==s}),tokens=sum(r['split']==s for r in rows)) for s in ("train","dev","test")}
    save_json(DATA/"training_counts.json",counts)
    print(counts)

if __name__=="__main__":main()
