"""Auditable plots, known transition boundaries and extraction invariants."""
import json
from collections import Counter

from acoustics import ROOT, np, extract, read_audio, pm, sf
from run_probe import save_json
from reliability_probe import OUT, METHODS, synth, tracks, values_at, measure, lag_seconds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def transition_test():
    result = []
    examples=OUT/"synthetic_examples"
    examples.mkdir(exist_ok=True)
    for f0 in (110,190,280):
        # Piecewise stationary segments with an explicitly known EH interval.
        # Hard transitions are an experimental intervention, not natural speech.
        chunks = [synth(f,f0,40,[3500,4500],i+10)[0] for i,f in enumerate(([700,1700,2600],[500,1900,2700],[300,2300,3000]))]
        y = np.concatenate([chunks[0][:12000],chunks[1][:2880],chunks[2][:12000]])
        sr=24000
        sf.write(examples/f"known_EH_boundary_f0_{f0}.wav",y,sr,subtype="FLOAT")
        center=.56  # middle interval is exactly .500--.620 s
        for method in METHODS:
            f=tracks(y,sr,method)
            for shift in (0,.01,.03,.06):
                fs=values_at(f,center+shift+np.arange(-3,4)*.01)
                estimate=np.mean(np.sort(fs,axis=0)[1:-1],axis=0)
                result.append(dict(f0=f0,method=method,shift_s=shift,true_bounds_s=[.5,.62],
                    estimated_hz=estimate.tolist(), relative_error=(abs(estimate-[500,1900,2700])/[500,1900,2700]).tolist()))
    save_json(OUT/"known_boundary_intervention.json",result)
    return result


def plots():
    # Predefined contrast: calibration examples and known held-out F2 failure.
    examples=[("p228","001","please","IY"),("p228","001","stella","EH"),
              ("p229","001","stella","EH"),("p229","002","these","IY")]
    fig,axes=plt.subplots(4,2,figsize=(15,13),layout="constrained")
    notes=[]
    for row,(speaker,utt,word,vowel) in enumerate(examples):
        paths=[ROOT/"data"/f"{speaker}_{utt}_{mic}.flac" for mic in ("mic1","mic2")]
        (a,sr),(b,srb)=[read_audio(p) for p in paths]
        lag,corr=lag_seconds(a,b,sr)
        phones=[json.loads((ROOT/"results_vowels/alignments"/(p.name+".json")).read_text())["phones"] for p in paths]
        pa=next(p for p in phones[0] if p["word"]==word and p["vowel"]==vowel)
        pb=next(p for p in phones[1] if p["word"]==word and p["vowel"]==vowel)
        start=max(0,pa["start"]-.12);end=min(len(a)/sr,pa["end"]+.12)
        center=(pa["start"]+pa["end"])/2
        wave,spec=axes[row]
        t=np.arange(int(start*sr),int(end*sr))/sr
        wave.plot(t,a[int(start*sr):int(end*sr)],lw=.55,label="mic1")
        wave.set_title(f"{speaker}_{utt} {word}/{vowel}; lag={lag*1000:.2f} ms, r={corr:.3f}")
        wave.set_ylabel("Amplitude")
        spec.specgram(a[int(start*sr):int(end*sr)],NFFT=1024,Fs=sr,noverlap=900,
                      xextent=(start,end),cmap="magma",vmin=-100,vmax=-30)
        spec.set_ylim(0,4000);spec.set_ylabel("Hz")
        times=np.arange(start+.025,end-.025,.005)
        for ceiling,color in ((5500,"cyan"),(5000,"lime")):
            f=pm.Sound(a,sampling_frequency=sr).to_formant_burg(time_step=.01,max_number_of_formants=5,
                maximum_formant=ceiling,window_length=.025,pre_emphasis_from=50)
            fs=values_at(f,times)
            for i in range(3):
                spec.plot(times,fs[:,i],".",color=color,ms=2,label=f"{ceiling} Hz F1-3" if i==0 else None)
        for ax in (wave,spec):
            for k in ("start","end"):
                ax.axvline(pa[k],color="blue",lw=1.4,label="mic1 HMM boundary" if k=="start" else None)
                ax.axvline(pb[k]-lag,color="red",ls="--",lw=1.2,label="mic2 boundary after lag" if k=="start" else None)
            ax.axvspan(center-.03,center+.03,color="gray",alpha=.2)
            ax.set_xlim(start,end);ax.set_xlabel("Time (s)")
        wave.legend(fontsize=7);spec.legend(fontsize=7)
        notes.append(dict(speaker=speaker,utterance=utt,word=word,vowel=vowel,
                          mic1_bounds=[pa["start"],pa["end"]],mic2_registered_bounds=[pb["start"]-lag,pb["end"]-lag]))
    fig.savefig(OUT/"boundary_tracks.png",dpi=150)
    plt.close(fig)
    save_json(OUT/"plot_examples.json",notes)


def validate():
    y,sr=synth([500,1900,2700],130,45,[3500,4500],91)
    f=tracks(y,sr,"old")
    record=extract(y,sr,return_frames=True)
    a,b=measure(record,.40,f),measure(record,.401,f)
    assert a and b
    assert np.max(abs(np.array(a["values"][:4])-b["values"][:4]))<.03
    lag,corr=lag_seconds(y,np.r_[np.zeros(240),y],sr)
    assert abs(lag-.01)<.0002 and corr>.9
    cal=json.loads((OUT/"calibration.json").read_text())
    for model in cal.values():
        assert all(s in ("p225","p226","p227","p228") and u in ("001","002","003","004") for s,u,k in model["fit_keys"])
    trials=json.loads((OUT/"trials.json").read_text())
    audit=json.loads((OUT/"audit.json").read_text())
    save_json(OUT/"validation.json",dict(gain_independent_registration_sign_test="passed (+10ms)",
        submillisecond_stationary_core_check="passed (F0-F3 log change < .03)",
        no_heldout_calibration_keys="passed", rejection_counts=dict(Counter(p["reason"] or "accepted" for p in audit)),
        caution="Visual review is assistant inspection, not human phonetic ground truth"))


if __name__=="__main__":
    transition_test()
    plots()
    validate()
    print("Transition interventions, plots and checks completed")
