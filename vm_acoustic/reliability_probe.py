"""Controlled alignment/estimator interventions and paired-mic reliability.

No held-out real-data tuning. See locked config and retained coverage in report.
"""
import hashlib
import json
from collections import defaultdict
from itertools import product
from math import gcd

from acoustics import ROOT, np, pm, read_audio, fit_calibration, coordinates
from scipy import signal
from run_probe import CAL_SPEAKERS, CAL_TEXTS, group_name, save_json, csv_write, summary
from vowel_probe import token_values

OUT = ROOT / "results_reliability"
PHONES = {"AE", "EH", "IY"}
METHODS = {"old": (5500, .025), "wide5000": (5000, .04),
           "wide5500": (5500, .04), "wide6000": (6000, .04)}
CONFIG = dict(phones=sorted(PHONES), min_phone_s=.12, core_s=.06,
              min_frames=5, max_center_formant_log_iqr=.12,
              max_shift_distance=.25, min_calibration_profiles=4,
              selection="mic1 only; no mic2 parameter-distance gate", lag_search_s=.15,
              methods=METHODS, synthetic_selection="development only: median error + 0.5*p90 error + invalid fraction")


def tracks(y, sr, method):
    ceiling, window = METHODS[method]
    snd = pm.Sound(y, sampling_frequency=sr)
    formant = snd.to_formant_burg(time_step=.01, max_number_of_formants=5,
                                 maximum_formant=ceiling, window_length=window, pre_emphasis_from=50)
    return formant


def values_at(formant, times):
    return np.array([[formant.get_value_at_time(i, float(t)) for i in (1, 2, 3)] for t in times])


def synth(target, f0, snr, upper, seed):
    sr = 24000
    t = np.arange(int(sr*.95))/sr
    # Independent of Praat: known digital all-pole transfer function.
    excitation = sum(np.sin(2*np.pi*f0*k*t + .13*k)/k for k in range(1, int(11000/f0)))
    den = np.array([1.])
    for f, bw in zip([*target, *upper], [65, 90, 120, 180, 220]):
        r = np.exp(-np.pi*bw/sr)
        den = np.convolve(den, [1, -2*r*np.cos(2*np.pi*f/sr), r*r])
    y = signal.lfilter([1], den, excitation)[int(sr*.15):]
    y *= .4/np.max(np.abs(y))
    noise = np.random.default_rng(seed).normal(size=len(y))
    y += noise * np.sqrt(np.mean(y*y)/np.mean(noise*noise)) * 10**(-snr/20)
    return y, sr


def synthetic():
    rows = []
    vowels = {"AE":[700,1700,2600], "EH":[500,1900,2700],
              "IY":[300,2300,3000], "UW_stress":[330,900,2200]}
    for split, f0s in (("dev", [90,160,240]), ("test", [110,190,280])):
        for (vowel, target), f0, upper, snr in product(vowels.items(), f0s, ([3500,4500],[3900,5100]), (25,45)):
            y, sr = synth(target, f0, snr, upper, 42 if split=="dev" else 84)
            # Both intervals are safely INSIDE stationary ground-truth vowel.
            # Moving a boundary here isolates numerical/estimator variability.
            for method in METHODS:
                f = tracks(y, sr, method)
                t = np.arange(.30, .50, .01)
                base = values_at(f, t)
                valid = np.isfinite(base).all(1) & (base[:,0]>150) & (base[:,0]<base[:,1]) & (base[:,1]<base[:,2])
                err = np.abs(base[valid]-target)/target
                shifted = values_at(f, t+.01)
                stable = valid & np.isfinite(shifted).all(1)
                shift_error = np.max(np.abs(np.median(shifted[stable],0)-np.median(base[stable],0))/target) if stable.any() else None
                rows.append(dict(split=split, vowel=vowel, f0=f0, upper=list(upper), snr=snr, method=method,
                    frame_count=len(t), valid_count=int(valid.sum()), median_relative_error=float(np.median(err)) if valid.any() else 1.,
                    p90_relative_error=float(np.percentile(err,90)) if valid.any() else 1.,
                    per_formant_error=np.median(err,0).tolist() if valid.any() else [1.,1.,1.],
                    boundary_shift_max_formant_relative_change=float(shift_error) if shift_error is not None else None))
    scores = {}
    for method in METHODS:
        rr = [r for r in rows if r["split"]=="dev" and r["method"]==method]
        scores[method] = float(np.median([r["median_relative_error"] for r in rr]) +
                              .5*np.percentile([r["p90_relative_error"] for r in rr],90) +
                              np.mean([1-r["valid_count"]/r["frame_count"] for r in rr]))
    chosen = min(scores, key=scores.get)
    save_json(OUT / "synthetic.json", dict(rows=rows, dev_scores=scores, chosen=chosen,
        ground_truth="digital pole frequencies; NOT exact spectral peak positions or anatomical truth"))
    return chosen, rows, scores


def lag_seconds(a, b, sr):
    # Low sample rate and band-pass limit cost; lag is registration, not a
    # fit to parameter distance. Positive lag means mic2 event occurs later.
    a, b = signal.resample_poly(a, 1, sr//8000), signal.resample_poly(b, 1, sr//8000)
    sos = signal.butter(3, [200,3000], fs=8000, btype="bandpass", output="sos")
    a, b = signal.sosfiltfilt(sos,a), signal.sosfiltfilt(sos,b)
    corr = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b),len(a), mode="full")
    mask = abs(lags)<=int(.15*8000)
    lag = lags[mask][np.argmax(abs(corr[mask]))]
    peak = float(np.max(abs(corr[mask])) / np.sqrt(np.sum(a*a)*np.sum(b*b)))
    return float(lag/8000), peak


def measure(record, center, formant):
    tt, vv = np.array(record["frames"]["times"]), np.array(record["frames"]["values"])
    # Fixed relative quadrature grid: a sub-ms registration shift must not
    # change which 10-ms frame happens to enter an interval. Do not bridge
    # missing voiced regions or extrapolate.
    times = center + np.arange(-3,4)*.01
    right = np.searchsorted(tt,times)
    if np.any(right==0) or np.any(right==len(tt)):
        return None
    left = right-1
    if np.any(tt[right]-tt[left]>.015):
        return None
    v = np.array([np.interp(times,tt,vv[:,i]) for i in range(6)]).T
    fs = values_at(formant, times)
    valid = np.isfinite(fs).all(1) & (fs[:,0]>150) & (fs[:,0]<fs[:,1]) & (fs[:,1]<fs[:,2]) & (fs[:,2]<5000)
    if valid.sum()<5:
        return None
    v = v[valid]
    v[:,1:4] = np.log(fs[valid])
    # Winsor-free 1/7 trimmed mean limits outliers and median rank switching.
    estimate = np.mean(np.sort(v,axis=0)[1:-1],axis=0)
    return dict(values=estimate.tolist(), times=times[valid].tolist(), frame_values=v.tolist(),
                formant_log_iqr=(np.percentile(v[:,1:4],75,axis=0)-np.percentile(v[:,1:4],25,axis=0)).tolist())


def main():
    OUT.mkdir(exist_ok=True)
    chosen, synthetic_rows, scores = synthetic()
    print("Synthetic dev selection", scores, chosen, flush=True)
    records = json.loads((ROOT/"results/features.json").read_text())
    lookup = {(r["speaker"],r["utterance"],r["mic"]):r for r in records if "error" not in r}
    scale = np.array(json.loads((ROOT/"results/calibration.json").read_text())["scale"])
    trials, audit = [], []
    for (speaker, utt, mic), a in sorted(lookup.items()):
        if mic != "mic1" or (speaker,utt,"mic2") not in lookup:
            continue
        b = lookup[speaker,utt,"mic2"]
        pa, pb = [ROOT/"results_vowels/alignments"/(r["file"]+".json") for r in (a,b)]
        if not pa.exists() or not pb.exists():
            continue
        aa, bb = [json.loads(p.read_text())["phones"] for p in (pa,pb)]
        bphones = {(p["word_index"],p["phone_index"],p["word"],p["vowel"]):p for p in bb}
        ya,sr = read_audio(ROOT/"data"/a["file"])
        yb,srb = read_audio(ROOT/"data"/b["file"])
        assert sr == srb and sr % 8000 == 0
        lag, corr = lag_seconds(ya,yb,sr)
        methods = sorted({"old",chosen})
        tracks_a = {m:tracks(ya,sr,m) for m in methods}
        tracks_b = {m:tracks(yb,sr,m) for m in methods}
        for phone in aa:
            if phone["vowel"] not in PHONES:
                continue
            key = (phone["word_index"],phone["phone_index"],phone["word"],phone["vowel"])
            other = bphones.get(key)
            center = (phone["start"]+phone["end"])/2
            info = dict(speaker=speaker, utterance=utt, vowel=phone["vowel"], word=phone["word"],
                        key=list(key), start=phone["start"], end=phone["end"], center=center,
                        lag_s=lag, waveform_correlation=corr, group=group_name(a))
            if not other or phone["end"]-phone["start"]<CONFIG["min_phone_s"]:
                audit.append(dict(info,accepted=False,reason="missing match or phone shorter than 120ms"))
                continue
            info["boundary_disagreement_after_lag_s"] = max(abs(other[k]-phone[k]-lag) for k in ("start","end"))
            if corr < .3:
                audit.append(dict(info,accepted=False,reason="registration correlation below 0.3"))
                continue
            measured = {}
            for m in methods:
                # independent alignment baseline, and single-reference registered core
                measured[m] = dict(a=measure(a,center,tracks_a[m]),
                    b_independent=measure(b,(other["start"]+other["end"])/2,tracks_b[m]),
                    b_registered=measure(b,center+lag,tracks_b[m]))
            legacy_a, legacy_b = token_values(a,[phone]),token_values(b,[other])
            legacy = [legacy_a[key]["values"],legacy_b[key]["values"]] if key in legacy_a and key in legacy_b else None
            ref = measured[chosen]["a"]
            shifts = [measure(a,center+s,tracks_a[chosen]) for s in (-.01,.01)]
            reason = None
            if ref is None or any(v is None for v in shifts):
                reason = "mic1 fewer than five valid core frames"
            elif max(ref["formant_log_iqr"]) > CONFIG["max_center_formant_log_iqr"]:
                reason = "mic1 core formant instability"
            else:
                sensitivity = max(float(np.linalg.norm((np.array(v["values"])-ref["values"])/scale)/np.sqrt(6)) for v in shifts)
                info["mic1_boundary_shift_distance"] = sensitivity
                if sensitivity>CONFIG["max_shift_distance"]:
                    reason = "mic1 boundary perturbation sensitivity"
            audit.append(dict(info,accepted=reason is None,reason=reason))
            # Keep pre-gate observations too: quantify rejection, not only survivors.
            trials.append(dict(info,accepted=reason is None, measurements=measured,legacy_values=legacy))
        print(speaker,utt,"registered",round(lag,5),flush=True)
    save_json(OUT/"trials.json",trials)
    save_json(OUT/"audit.json",audit)
    save_json(OUT/"config.json",dict(CONFIG,chosen=chosen,script_sha256=hashlib.sha256((ROOT/"reliability_probe.py").read_bytes()).hexdigest()))
    evaluate(trials, lookup, scale, chosen, audit, synthetic_rows, scores)


def evaluate(trials, lookup, scale, chosen, audit, synthetic_rows, scores):
    # Lock a single shared observation set across all interventions.
    methods = sorted({"old",chosen})
    complete = [p for p in trials if p["legacy_values"] is not None and all(all(v is not None for v in p["measurements"][m].values()) for m in methods)]
    usable = [p for p in complete if p["accepted"]]
    rows = []
    for p in usable:
        r = {k:p[k] for k in ("speaker","utterance","vowel","word","group")}
        for m in methods:
            mm = p["measurements"][m]
            for reg in ("independent","registered"):
                r[m+"_"+reg] = float(np.linalg.norm((np.array(mm["b_"+reg]["values"])-mm["a"]["values"])/scale)/np.sqrt(6))
        a,b = [lookup[p["speaker"],p["utterance"],mic] for mic in ("mic1","mic2")]
        r["sentence_raw"] = float(np.linalg.norm((np.array(b["values"])-a["values"])/scale)/np.sqrt(6))
        r["legacy_token_median"] = float(np.linalg.norm((np.array(p["legacy_values"][1])-p["legacy_values"][0])/scale)/np.sqrt(6))
        rows.append(r)
    # Fit vowel-conditioned correction on admitted cal profiles only.
    models = {}
    for vowel in PHONES:
        pp = [p for p in usable if p["vowel"]==vowel and p["speaker"] in CAL_SPEAKERS and p["utterance"] in CAL_TEXTS]
        if len(pp)>=4 and len({p["speaker"] for p in pp})>=2:
            models[vowel] = fit_calibration([(p["measurements"][chosen]["a"]["values"],p["measurements"][chosen]["b_registered"]["values"]) for p in pp])
            models[vowel]["fit_keys"] = [[p["speaker"],p["utterance"],p["key"]] for p in pp]
    for p,r in zip(usable,rows):
        if p["vowel"] in models:
            m = models[p["vowel"]]
            mm = p["measurements"][chosen]
            delta = np.array(mm["b_registered"]["values"])-mm["a"]["values"]
            r["chosen_registered_bias_common_scale"] = float(np.linalg.norm((delta-m["mic2_bias"])/scale)/np.sqrt(6))
        else:
            r["chosen_registered_bias_common_scale"] = None
    save_json(OUT/"calibration.json",models)
    csv_write(OUT/"paired_distances.csv",rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["speaker"],r["utterance"],r["group"]].append(r)
    sentence_rows = []
    measures = [k for k in rows[0] if k not in ("speaker","utterance","vowel","word","group")]
    for (speaker,utt,g), rr in grouped.items():
        sentence_rows.append(dict(speaker=speaker,utterance=utt,group=g,tokens=len(rr),
            **{k:float(np.median([r[k] for r in rr])) if all(r[k] is not None for r in rr) else None for k in measures}))
    csv_write(OUT/"sentence_distances.csv",sentence_rows)
    metrics = {g:{k:summary([r[k] for r in sentence_rows if r["group"]==g and r[k] is not None]) for k in measures}
               for g in sorted(set(r["group"] for r in sentence_rows))}
    coverage = {g:dict(candidates=sum(p["group"]==g for p in audit),accepted=sum(p["group"]==g for p in usable),
                       utterance_pairs=sum(r["group"]==g for r in sentence_rows)) for g in metrics}
    ungated = {}
    for group in metrics:
        pairs = defaultdict(list)
        for p in complete:
            if p["group"]!=group:
                continue
            mm=p["measurements"][chosen]
            d=float(np.linalg.norm((np.array(mm["b_registered"]["values"])-mm["a"]["values"])/scale)/np.sqrt(6))
            pairs[p["speaker"],p["utterance"]].append(d)
        ungated[group]=summary([float(np.median(v)) for v in pairs.values()])
    save_json(OUT/"metrics.json",dict(metrics=metrics,coverage=coverage,ungated_registered=ungated))
    lines = ["# 测量可靠性：固定核心、时间配准、合成真值对照", "", f"合成开发集选择 {chosen}；参数预先记录于config.json。",
             "仅AE/EH/IY、≥120ms、固定中央60ms七点相对时间网格、至少5有效点、逐维去掉两端后均值；不跨>15ms缺口插值。mic1质量门控与双麦距离无关。",
             "两麦用波形相关估计统一时间差；独立边界与配准边界在同一token上比较。",
             "mic1质量门控可能偏向容易样本；覆盖率是结果的一部分。自动边界尚非人工真值。",
             "", "## 合成验证", "", "数字全极点合成，4种频率配置×3个F0×2种高频共振峰×2种SNR，每个split 48条；dev/test的F0与噪声种子不同，结构相同。不是独立解剖真值。",
             "", "|split|method|跨音频中位相对误差|音频p90误差的90分位|10ms移动的最大F1-F3相对变化中位数|", "|---|---|---:|---:|---:|"]
    for split,m in product(("dev","test"),METHODS):
        rr = [r for r in synthetic_rows if r["split"]==split and r["method"]==m]
        lines.append(f"|{split}|{m}|{np.median([r['median_relative_error'] for r in rr]):.4f}|{np.percentile([r['p90_relative_error'] for r in rr],90):.4f}|{np.median([r['boundary_shift_max_formant_relative_change'] for r in rr]):.4f}|")
    keys = ["sentence_raw","legacy_token_median","old_independent",chosen+"_registered","chosen_registered_bias_common_scale"]
    lines += ["", "## 实录双麦：共同保留token，句内中位后跨句中位，全部同一原始尺度", "", "|条件|候选/保留token|句对|整句|上一轮元音中位|固定核心独立边界|固定核心配准|配准+bias（有效句对数）|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g,mm in metrics.items():
        c=coverage[g]
        vals=[mm[k]["median"] for k in keys]
        rendered=["NA" if v is None else f"{v:.4f}" for v in vals]
        rendered[-1]+=f" (n={mm[keys[-1]]['n']})"
        lines.append(f"|{g}|{c['candidates']}/{c['accepted']}|{c['utterance_pairs']}|"+"|".join(rendered)+"|")
    lines += ["", "bias列可能因标定不足缺失；具体分母见metrics.json，不能直接与不同分母列比较。",
              "筛选后整句基线也仅取相同句对，并非原来62对总体；不能把子集改善等同总体改善。",
              "旧估计独立→配准主要检验时间边界/配准效应；旧→新配准固定时刻检验估计器效应。误差有交互，并非严格可相加的因果分解。",
              "请结合audit.json中的失败与覆盖率以及边界/轨迹图核验。"]
    lines += ["", "本轮合成开发集未支持替换原Burg设置；改进的是取样网格、汇总、配准与拒绝机制，不宣称产生了更准确的新共振峰算法。",
              "实现过程已查看这批留出结果并做过网格修正，属于探索性排错，下一轮正式验收应使用新数据。",
              "", "## 质量门控前的完整token子集（可用性门槛仍存在）", ""]
    for g,v in ungated.items():
        lines.append(f"- {g}: n={v['n']} 句对，固定核心配准中位距离 {v['median']:.4f}")
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Done",len(usable),"usable",flush=True)


if __name__ == "__main__":
    main()
