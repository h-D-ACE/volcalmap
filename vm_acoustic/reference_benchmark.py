"""Independent reference boundaries/formants; choose configuration on dev only."""
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict

from acoustics import ROOT,np,pm,read_audio
from run_probe import save_json,csv_write

DATA=ROOT/"data_reference"
OUT=ROOT/"results_reference"
VOWELS=["ae","eh","iy","ah","uh","uw"]
CONFIGS={f"c{c}_w{w}":dict(ceiling=c,window=w/1000) for c in (4500,5000,5500,6000,6500) for w in (25,40)}
BASE="c5500_w25"

def parse(path,n):
    rows={}
    for line in path.read_text(errors="replace").splitlines():
        parts=line.split()
        if len(parts)==n and re.fullmatch(r"[mwbg]\d\d[a-z]{2}",parts[0]):
            rows[parts[0]]=list(map(float,parts[1:]))
    return rows

def build_subset():
    times=parse(DATA/"timedata.dat",5)
    formants=parse(DATA/"bigdata.dat",30)
    if not formants:
        # Header: file, duration,F0,F1-3,8*3 contours at 10--80%.
        raise ValueError("bigdata format mismatch")
    speakers={g:sorted({n[:3] for n in times if n[0]==g and all(n[:3]+v in times and n[:3]+v in formants for v in VOWELS)})[:8] for g in ("m","w")}
    selected=[s+v for ss in speakers.values() for s in ss for v in VOWELS]
    out=DATA/"audio";out.mkdir(exist_ok=True)
    rows=[]
    with zipfile.ZipFile(DATA/"h95-alldata.zip") as archive:
        for group,nested in (("m","men.zip"),("w","women.zip")):
            with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as z:
                names={n.rsplit("/",1)[-1].lower():n for n in z.namelist()}
                for name in selected:
                    if name[0]!=group: continue
                    b=z.read(names[name+".wav"])
                    (out/(name+".wav")).write_bytes(b)
                    vals=formants[name]
                    start,end,c1,c2=np.array(times[name])/1000
                    references=np.array(vals[5:]).reshape(8,3)
                    assert end>start and abs((end-start)*1000-vals[0])<2
                    rows.append(dict(id=name,speaker=name[:3],vowel=name[3:],
                        split="dev" if name[:3] in speakers[group][:4] else "test",
                        path="audio/"+name+".wav",sha256=hashlib.sha256(b).hexdigest(),
                        start=start,end=end,center1=c1,center2=c2,
                        reference_f0=vals[1],reference_times=(start+(end-start)*np.arange(1,9)/10).tolist(),
                        reference_formants=references.tolist(),duration=end-start))
    save_json(DATA/"subset.json",dict(selection="First eight complete adult speakers of each group, six prespecified vowels; no formant-error or duration exclusions",rows=rows))
    return rows

def formant_track(y,sr,config,t):
    f=pm.Sound(y,sampling_frequency=sr).to_formant_burg(time_step=.005,max_number_of_formants=5,
             maximum_formant=config["ceiling"],window_length=config["window"],pre_emphasis_from=50)
    return np.array([[f.get_value_at_time(i,float(ti)) for i in (1,2,3)] for ti in t])

def main():
    OUT.mkdir(exist_ok=True)
    rows=build_subset()
    evaluations=[]
    for r in rows:
        y,sr=read_audio(DATA/r["path"])
        truth=np.array(r["reference_formants"])
        # Score central 20--80% at their annotated times, not a pooled median.
        t=np.array(r["reference_times"])[1:]
        truth=truth[1:]
        for name,c in CONFIGS.items():
            fs=formant_track(y,sr,c,t)
            mask=(truth>0)&np.isfinite(fs)&(fs>0)
            err=np.where(mask,abs(fs-truth)/np.maximum(truth,1),1.)
            # Reference zeros are missing truth, not estimation errors.
            available=truth>0
            loss=err[available]
            evaluations.append(dict(id=r["id"],speaker=r["speaker"],vowel=r["vowel"],split=r["split"],method=name,
                 median_error=float(np.median(loss)),p90_error=float(np.percentile(loss,90)),
                 valid_rate=float(mask[available].mean()),within10_rate=float((err[available]<=.1).mean()),
                 all_f2_within10=bool(np.all(err[:,1][available[:,1]]<=.1)),
                 estimate=fs.tolist(),truth=truth.tolist(),times=t.tolist(),
                 median_per_formant=[float(np.median(err[:,i][available[:,i]])) for i in range(3)]))
    scores={m:float(np.mean([r["median_error"]+.5*r["p90_error"] for r in evaluations if r["split"]=="dev" and r["method"]==m])) for m in CONFIGS}
    chosen=min(scores,key=scores.get)
    save_json(OUT/"locked_method.json",dict(chosen=chosen,config=CONFIGS[chosen],dev_scores=scores,
        baseline=BASE,selection="Mean per-token median relative error + 0.5*p90, development speakers only",
        reference_sha256=hashlib.sha256((DATA/"bigdata.dat").read_bytes()).hexdigest(),
        scope="Adult hVd natural speech; human-edited LPC reference, not anatomical ground truth"))
    save_json(OUT/"evaluations.json",evaluations)
    summaries={}
    for split in ("dev","test"):
        summaries[split]={}
        for method in CONFIGS:
            rr=[r for r in evaluations if r["split"]==split and r["method"]==method]
            summaries[split][method]=dict(n=len(rr),median_error=float(np.median([r["median_error"] for r in rr])),
                p90_token_error=float(np.percentile([r["p90_error"] for r in rr],90)),
                point_within10_rate=float(np.mean([r["within10_rate"] for r in rr])),
                valid_rate=float(np.mean([r["valid_rate"] for r in rr])),
                token_all_f2_within10_rate=float(np.mean([r["all_f2_within10"] for r in rr])))
    save_json(OUT/"metrics.json",summaries)
    durations=np.array([r["duration"] for r in rows])
    lines=["# 有独立标注的自然元音验证", "", "Hillenbrand 1995，自然/hVd/词，16位成人×6个预定元音=96条；8人开发/8人测试。",
        "按说话者编号预选，没有依据算法误差、元音长度或听众识别率删样本。参考为原研究编辑过的LPC轮廓，独立于本实现但不是解剖真值。",
        f"元音时长中位{np.median(durations)*1000:.1f}ms，范围{durations.min()*1000:.1f}–{durations.max()*1000:.1f}ms；≥180ms {int((durations>=.18).sum())}/96。",
        f"只用开发集选择 {chosen}。在独立标注的20–80%时刻比较F1–F3，参考零值为缺失。",
        "", "|划分|设置|条数|每条中位误差的中位|每条p90误差的p90|点误差≤10%|可计算率|", "|---|---|---:|---:|---:|---:|---:|"]
    for split in summaries:
        for m in dict.fromkeys([BASE,chosen]):
            s=summaries[split][m]
            lines.append(f"|{split}|{m}|{s['n']}|{s['median_error']:.4f}|{s['p90_token_error']:.4f}|{s['point_within10_rate']:.3f}|{s['valid_rate']:.3f}|")
    lines += ["", "全部音频均保留；可计算率和误差≤10%是不同概念。本数据没有双麦，不能把人工设备滤波实验写成真实双麦验证。"]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print("Selected",chosen,"scores",scores,flush=True)
    print("Test",summaries["test"][chosen],flush=True)

if __name__=="__main__": main()
