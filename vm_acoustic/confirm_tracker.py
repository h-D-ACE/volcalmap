"""Frozen path algorithm confirmation on four newly downloaded speakers."""
import json
from collections import Counter,defaultdict
from acoustics import ROOT,np,read_audio
from build_sentence_validation import align
from reliability_probe import lag_seconds
from measurement_v2 import Measurement
from formant_tracker import CandidateBank
from evaluate_tracker_pairs import aggregate
from run_probe import save_json,csv_write,summary

DATA=ROOT/"data_vctk_confirmation";OUT=ROOT/"results_confirmation"

def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((DATA/"manifest.json").read_text())
    # All constants, normalization, and strength are already frozen.
    scale=np.array(json.loads((ROOT/"results_sentence_validation/calibration.json").read_text())["scale"]["scale"])
    rows=[];failures=[];counts=Counter()
    for speaker in manifest["speakers"]:
        for utt in manifest["utterances"]:
            files=[DATA/f"{speaker}_{utt}_{m}.flac" for m in ("mic1","mic2")];text=DATA/f"{speaker}_{utt}.txt"
            if not all(f.exists() for f in [*files,text]):failures.append([speaker,utt,"source_missing"]);continue
            a,sr=read_audio(files[0]);b,sr2=read_audio(files[1]);assert sr==sr2
            try:pp=align(a,sr,text.read_text())
            except Exception as e:failures.append([speaker,utt,str(e)]);continue
            lag,corr=lag_seconds(a,b,sr)
            ma,mb=Measurement(a,sr),Measurement(b,sr)
            ba,bb=CandidateBank(a,sr),CandidateBank(b,sr)
            for i,t in enumerate(pp):
                counts["candidate"]+=1
                long=t["end"]-t["start"]>=.18;counts["long_candidate"]+=int(long)
                sa=ma.token(t["start"],t["end"]);sb=mb.token(t["start"]+lag,t["end"]+lag)
                va,ga=aggregate(ba,t,sa,0);vb,gb=aggregate(bb,t,sb,lag)
                fixed=all(sa["mask"]+sb["mask"]);tracked=ga and gb
                counts["fixed_complete"]+=int(fixed);counts["path_complete"]+=int(tracked)
                if not(fixed and tracked):continue
                counts["common_complete"]+=1;counts["long_common_complete"]+=int(long)
                rows.append(dict(speaker=speaker,utterance=utt,index=i,word=t["word"],vowel=t["vowel"],long=long,
                    fixed=float(np.linalg.norm((np.array(sb["values"])-sa["values"])/scale)/np.sqrt(6)),
                    path=float(np.linalg.norm((np.array(vb)-va)/scale)/np.sqrt(6))))
            print("confirmation",speaker,utt,flush=True)
    csv_write(OUT/"distances.csv",rows)
    metrics={}
    for stratum in ("all","long"):
        groups=defaultdict(list)
        for r in rows:
            if stratum=="all" or r["long"]:groups[r["speaker"],r["utterance"]].append(r)
        ss=[dict(speaker=s,utterance=u,**{k:float(np.median([r[k] for r in rr])) for k in ("fixed","path")}) for (s,u),rr in groups.items()]
        metric={k:summary([r[k] for r in ss]) for k in ("fixed","path")}
        metric["per_speaker"]={s:{k:summary([r[k] for r in ss if r["speaker"]==s]) for k in ("fixed","path")} for s in manifest["speakers"]}
        rng=np.random.default_rng(710);boots=[]
        for _ in range(1000):
            sample=rng.choice(manifest["speakers"],4,replace=True)
            selected=[r for s in sample for r in ss if r["speaker"]==s]
            if selected:boots.append(np.median([r["path"] for r in selected])-np.median([r["fixed"] for r in selected]))
        metric["difference_cluster_bootstrap_95"]=np.percentile(boots,[2.5,97.5]).tolist() if boots else None
        metrics[stratum]=metric
    result=dict(coverage=dict(counts),metrics=metrics,failures=failures,
        configuration="5500/40ms fixed baseline; candidate ceilings 4500..6500/25ms, bandwidth+continuity strength1; no test fitting")
    save_json(OUT/"metrics.json",result)
    lines=["# 冻结算法的新增说话者确认", "", "p244/p250/p252/p254，预定001/003/005/008/011/012，共24句对。配置、尺度、路径强度在下载本集前已经固定。",
        "只用mic1自动对齐，mic2按波形时差投影；没有根据误差筛选片段。",f"失败/缺失 {len(failures)}；覆盖 {dict(counts)}。",
        "", "|层|句对|固定40ms|路径法|按人bootstrap差值95%区间|", "|---|---:|---:|---:|---|"]
    for s,m in metrics.items():lines.append(f"|{s}|{m['fixed']['n']}|{m['fixed']['median']:.4f}|{m['path']['median']:.4f}|{m['difference_cluster_bootstrap_95']}|")
    lines += ["", "只有4人。若区间跨0或个别人退化，不宣称总体显著改善。不会用本集再调路径强度。",
              "全套测量仍非跨日/未知设备或感知音色真值；这只验证已知双麦重复性。"]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(result,flush=True)

if __name__=="__main__":main()
