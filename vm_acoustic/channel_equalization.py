"""Train-only smooth microphone magnitude correction, evaluated at fixed tokens.

Not a universal microphone inverse. No held-out selection of correction strength.
"""
import json
from collections import defaultdict
from math import gcd
from acoustics import ROOT,np,read_audio
from scipy import signal,ndimage
from measurement_v2 import Measurement
from run_probe import save_json,csv_write,summary

DATA=ROOT/"data_vctk_validation"
OUT=ROOT/"results_sentence_validation"

def fit(pairs):
    spectra=[];keys=[]
    for p in pairs:
        if p["group"]!="cal_old_text":continue
        pp=[]
        for mic in ("mic1","mic2"):
            y,sr=read_audio(DATA/f"{p['speaker']}_{p['utterance']}_{mic}.flac")
            g=gcd(sr,24000);y=signal.resample_poly(y,24000//g,sr//g)
            f,power=signal.welch(y,fs=24000,nperseg=2048,noverlap=1024)
            pp.append(power)
        ratio=10*np.log10(np.maximum(pp[0],1e-14)/np.maximum(pp[1],1e-14))
        ratio-=np.median(ratio[(f>=500)&(f<=3000)]) # gain is not the target
        spectra.append(ratio);keys.append([p["speaker"],p["utterance"]])
    db=ndimage.gaussian_filter1d(np.median(spectra,axis=0),sigma=8)
    db=np.clip(db,-6,6)
    db[f<80]=db[np.searchsorted(f,80)]
    # Unity outside measured speech band; short raised-cosine taper above 5k.
    blend=np.clip((6000-f)/1000,0,1)
    blend=.5-.5*np.cos(np.pi*blend)
    db*=blend
    fir=signal.firwin2(257,f,10**(db/20),fs=24000)
    model=dict(sample_rate=24000,fit_keys=keys,frequencies=f.tolist(),target_gain_db=db.tolist(),
               fir=fir.tolist(),max_gain_db=6,smoothing_sigma_bins=8,
               note="mic2->mic1, magnitude-only, trained on cal_old_text; no test tuning")
    save_json(OUT/"channel_filter.json",model)
    return model

def main():
    data=json.loads((OUT/"measurements.json").read_text())
    pairs=data["pairs"];model=fit(pairs)
    calibration=json.loads((OUT/"calibration.json").read_text())
    scale=np.array(calibration["scale"]["scale"])
    rows=[];counts=defaultdict(lambda:dict(candidates=0,original_complete=0,eq_complete=0,common_complete=0))
    for p in pairs:
        y,sr=read_audio(DATA/f"{p['speaker']}_{p['utterance']}_mic2.flac")
        g=gcd(sr,24000);y=signal.resample_poly(y,24000//g,sr//g)
        eq=signal.fftconvolve(y,np.array(model["fir"]),mode="same")
        m=Measurement(eq,24000,**data["method"]["config"])
        for i,t in enumerate(p["tokens"]):
            c=counts[p["group"]];c["candidates"]+=1
            a=t["reference_selected"]["a"];b=t["reference_selected"]["b"]
            corrected=m.token(t["start"]+p["lag"],t["end"]+p["lag"])
            good_a=all(a["mask"]);good_b=all(b["mask"]);good_eq=all(corrected["mask"])
            c["original_complete"]+=int(good_a and good_b);c["eq_complete"]+=int(good_a and good_eq)
            if not(good_a and good_b and good_eq):continue
            c["common_complete"]+=1
            av,bv,ev=[np.array(x["values"]) for x in (a,b,corrected)]
            da=(bv-av)/scale;de=(ev-av)/scale
            row=dict(speaker=p["speaker"],utterance=p["utterance"],group=p["group"],index=i,vowel=t["vowel"],
                long=t["end"]-t["start"]>=.18,original=float(np.linalg.norm(da)/np.sqrt(6)),
                equalized=float(np.linalg.norm(de)/np.sqrt(6)))
            for j in range(6):row[f"original_abs_z{j}"]=float(abs(da[j]));row[f"equalized_abs_z{j}"]=float(abs(de[j]))
            rows.append(row)
        print("EQ",p["speaker"],p["utterance"],flush=True)
    csv_write(OUT/"equalization_distances.csv",rows)
    metrics={};sentences=[]
    for stratum in ("all","long"):
        metrics[stratum]={}
        for group in sorted(counts):
            bysentence=defaultdict(list)
            for r in rows:
                if r["group"]==group and (stratum=="all" or r["long"]):bysentence[r["speaker"],r["utterance"]].append(r)
            ss=[dict(speaker=s,utterance=u,group=group,stratum=stratum,
                original=float(np.median([r["original"] for r in rr])),equalized=float(np.median([r["equalized"] for r in rr]))) for (s,u),rr in bysentence.items()]
            sentences.extend(ss)
            before=summary([r["original"] for r in ss]);after=summary([r["equalized"] for r in ss])
            # Speaker-cluster bootstrap is descriptive with only four people.
            speakers=sorted({r["speaker"] for r in ss});rng=np.random.default_rng(710);boots=[]
            if speakers:
                for _ in range(1000):
                    sampled=rng.choice(speakers,len(speakers),replace=True)
                    rr=[r for s in sampled for r in ss if r["speaker"]==s]
                    boots.append(np.median([r["equalized"] for r in rr])-np.median([r["original"] for r in rr]))
            metrics[stratum][group]=dict(original=before,equalized=after,speakers=len(speakers),
                difference_bootstrap_95=np.percentile(boots,[2.5,97.5]).tolist() if boots else None,
                improvement_fraction=1-after["median"]/before["median"] if before["median"] else None)
    csv_write(OUT/"equalization_sentence_distances.csv",sentences)
    save_json(OUT/"equalization_metrics.json",dict(coverage=dict(counts),metrics=metrics))
    lines=["# 已知双麦的音频层频响校正", "", "只用标定人001–006的同次录音估计mic2→mic1平滑幅频响应；固定±6dB限幅、257抽头线性相位FIR、80–6000Hz范围，离线卷积补偿时延。",
           "不拟合测试距离，不改变元音边界，不依赖声纹/身份标签。相位、非线性、未知设备和混响并未消除。全部方法使用相同完整token集合和同一尺度。",
           "本轮是预先指定的单个物理校正对照，未在测试上搜索滤波强度。", ""]
    for stratum,gg in metrics.items():
        lines += [f"## {stratum}","", "|条件|句对|原距离|频响校正|变化改善比例|按人bootstrap差值95%区间|", "|---|---:|---:|---:|---:|---|"]
        for g,m in gg.items():
            if m["original"]["n"]:
                lines.append(f"|{g}|{m['original']['n']}|{m['original']['median']:.4f}|{m['equalized']['median']:.4f}|{m['improvement_fraction']:.1%}|{m['difference_bootstrap_95']}|")
    lines += ["", "每组只有至多4人，bootstrap不能代表总体可靠区间；误差下降不等于音色感知相似度改善。覆盖与每参数误差见JSON/CSV。"]
    (OUT/"equalization_report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":main()
