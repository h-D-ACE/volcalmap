"""Boundary review assets, same-denominator legacy coverage and contract checks."""
import json
import hashlib
from collections import Counter,defaultdict

from acoustics import ROOT,np,read_audio,extract,pm
from run_probe import save_json
from reliability_probe import measure
from measurement_v2 import Measurement
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=ROOT/"results_sentence_validation"
DATA=ROOT/"data_vctk_validation"

def legacy_coverage(pairs):
    # Exactly the previous three-phone candidate set and fixed old scale.
    scale=np.array(json.loads((ROOT/"results/calibration.json").read_text())["scale"])
    counts=defaultdict(Counter);long=[];rows=[]
    for p in pairs:
        y,sr=read_audio(DATA/f"{p['speaker']}_{p['utterance']}_mic1.flac")
        try:record=extract(y,sr,return_frames=True)
        except ValueError:record=None
        f=pm.Sound(y,sampling_frequency=sr).to_formant_burg(time_step=.01,max_number_of_formants=5,maximum_formant=5500,window_length=.025,pre_emphasis_from=50)
        for i,t in enumerate(p["tokens"]):
            duration=t["end"]-t["start"]
            if duration>=.18:
                long.append(dict(speaker=p["speaker"],utterance=p["utterance"],group=p["group"],index=i,
                    word=t["word"],vowel=t["vowel"],start=t["start"],end=t["end"],duration=duration,
                    mic2_shift_s=p["lag"],text=p["text"],boundary_source="mic1 HMM; mic2 waveform registration",
                    mic1_file=f"{p['speaker']}_{p['utterance']}_mic1.flac",mic2_file=f"{p['speaker']}_{p['utterance']}_mic2.flac",
                    audio_sha256=p["audio_sha256"],status="requires_phonetic_review",distance_used_for_selection=False))
            if t["vowel"] not in ("AE","EH","IY"):continue
            c=counts[p["group"]];c["candidates"]+=1
            complete=all(t["reference_selected"]["a"]["mask"]+t["reference_selected"]["b"]["mask"])
            c["v2_complete_both_mics"]+=int(complete)
            reason=None
            center=(t["start"]+t["end"])/2
            if duration<.12:reason="short"
            elif record is None:reason="v1_whole_utterance_gate"
            else:
                v=measure(record,center,f);shift=[measure(record,center+s,f) for s in (-.01,.01)]
                if v is None or any(q is None for q in shift):reason="too_few_joint_frames"
                elif max(v["formant_log_iqr"])>.12:reason="formant_iqr"
                elif max(np.linalg.norm((np.array(q["values"])-v["values"])/scale)/np.sqrt(6) for q in shift)>.25:reason="boundary_shift"
            c["legacy_mic1_gate_pass"]+=int(reason is None)
            c[reason or "legacy_pass"]+=1
            rows.append(dict(speaker=p["speaker"],utterance=p["utterance"],index=i,vowel=t["vowel"],reason=reason,v2_complete=complete))
    save_json(OUT/"coverage_audit.json",dict(counts={g:dict(c) for g,c in counts.items()},rows=rows,
        note="Same AE/EH/IY candidates; legacy mic1 gate is optimistic (does not require mic2 availability); not accuracy truth"))
    save_json(OUT/"long_vowel_manifest.json",dict(selection="ALL aligned AE/EH/IY/AA/AH/UW >=180ms; no distance or trajectory exclusion",rows=long))
    return long,counts

def reference_plots():
    ref=ROOT/"data_reference"
    rows=json.loads((ref/"subset.json").read_text())["rows"]
    evals=json.loads((ROOT/"results_reference/evaluations.json").read_text())
    # First test adult of each group, three fixed contrasting vowels.
    speakers=[min(r["speaker"] for r in rows if r["split"]=="test" and r["speaker"][0]==g) for g in ("m","w")]
    selected=[r for r in rows if r["speaker"] in speakers and r["vowel"] in ("ae","iy","uw")]
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout="constrained")
    for ax,r in zip(axes.flat,selected):
        y,sr=read_audio(ref/r["path"])
        ax.specgram(y,NFFT=512,Fs=sr,noverlap=450,cmap="magma",vmin=-105,vmax=-35)
        ax.axvline(r["start"],color="white",ls="--");ax.axvline(r["end"],color="white",ls="--")
        truth=np.array(r["reference_formants"]);times=r["reference_times"]
        for j in range(3):ax.plot(times,np.where(truth[:,j]>0,truth[:,j],np.nan),"wo",ms=3,label="reference" if j==0 else None)
        for method,color in (("c5500_w25","cyan"),("c5500_w40","lime")):
            rr=next(x for x in evals if x["id"]==r["id"] and x["method"]==method)
            for j in range(3):ax.plot(rr["times"],np.array(rr["estimate"])[:,j],color=color,lw=1,label=method if j==0 else None)
        ax.set_ylim(0,4000);ax.set_xlim(max(0,r["start"]-.08),r["end"]+.08)
        ax.set_title(f"{r['id']} duration {r['duration']*1000:.0f} ms");ax.set_xlabel("Time (s)");ax.set_ylabel("Hz");ax.legend(fontsize=7)
    fig.savefig(ROOT/"results_reference/boundary_reference_audit.png",dpi=140);plt.close(fig)

def long_plots(long):
    # Every selected long vowel is included, paginated; not only successes.
    for page in range((len(long)+5)//6):
        rr=long[page*6:page*6+6]
        fig,axes=plt.subplots(3,2,figsize=(14,10))
        fig.subplots_adjust(left=.07,right=.98,bottom=.07,top=.94,hspace=.45,wspace=.25)
        for ax,r in zip(axes.flat,rr):
            y,sr=read_audio(DATA/r["mic1_file"])
            start=max(0,r["start"]-.08);end=min(len(y)/sr,r["end"]+.08)
            ax.specgram(y[int(start*sr):int(end*sr)],NFFT=1024,Fs=sr,noverlap=900,
                xextent=(start,end),cmap="magma",vmin=-110,vmax=-35)
            m=Measurement(y,sr)
            times=np.linspace(r["start"],r["end"],30)
            for j in (1,2,3):
                fs=[m.formants.get_value_at_time(j,float(t)) for t in times]
                ax.plot(times,fs,"c.",ms=3)
            ax.axvline(r["start"],color="white",ls="--");ax.axvline(r["end"],color="white",ls="--")
            ax.set_ylim(0,4000);ax.set_xlabel("Time (s)");ax.set_ylabel("Hz")
            ax.set_title(f"{r['speaker']}_{r['utterance']} {r['word']}/{r['vowel']} {r['duration']*1000:.0f}ms")
        for ax in list(axes.flat)[len(rr):]:ax.set_visible(False)
        for index,ax in enumerate(axes.flat):
            ax.set_position([.08+.47*(index%2),.71-.30*(index//2),.39,.22])
            ax.set_yticks([0,1000,2000,3000,4000])
            ax.set_yticklabels(["0","1000","2000","3000","4000"])
            ax.tick_params(labelsize=9)
        fig.savefig(OUT/f"long_vowels_page_{page+1}.png",dpi=130,bbox_inches="tight",pad_inches=.2);plt.close(fig)

def validate_training():
    from train_physics import read_manifest
    x,y,rows,manifest=read_manifest(ROOT/"data_reference/training_manifest.json")
    excluded=set(manifest["benchmark_speakers_excluded"])
    assert not excluded & {r["speaker"] for r in rows}
    assert np.isfinite(x).all()
    assert np.isfinite(y).any(axis=0).all()
    save_json(ROOT/"results_reference/training_manifest_validation.json",dict(hash_and_split_checks="passed",shape=list(x.shape),
         missing_targets=int((~np.isfinite(y)).sum()),benchmark_speakers_excluded=True))

def main():
    pairs=json.loads((OUT/"measurements.json").read_text())["pairs"]
    long,counts=legacy_coverage(pairs)
    reference_plots();long_plots(long);validate_training()
    print("Long vowels",len(long),"coverage",dict(counts),flush=True)

if __name__=="__main__":main()
