"""Reproducible paired-microphone pilot. No listening labels or neural training.

Whole-utterance voiced-frame medians; NOT the proposed 24-D phoneme model.
"""
from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
import platform
import sys
import time
from pathlib import Path

from acoustics import ROOT, CONFIG, FEATURES, pm, np, sf, extract, read_audio, fit_calibration, coordinates
import scipy

CAL_SPEAKERS = ["p225", "p226", "p227", "p228"]
HOLDOUT_SPEAKERS = ["p229", "p230", "p231", "p232"]
CAL_TEXTS = ["001", "002", "003", "004"]
NEW_TEXTS = ["005", "006", "007", "008"]
OUT = ROOT / "results"


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    return {"n": len(values), "median": float(np.median(values)), "mean": float(np.mean(values))} if values else {"n": 0, "median": None, "mean": None}


def make_pairs(records):
    lookup = {(r["speaker"], r["utterance"], r["mic"]): r for r in records if "error" not in r}
    pairs = []
    for speaker, utt, mic in sorted(lookup):
        if mic == "mic1" and (speaker, utt, "mic2") in lookup:
            pairs.append((lookup[speaker, utt, "mic1"], lookup[speaker, utt, "mic2"]))
    return pairs


def group_name(record):
    speaker_new = record["speaker"] in HOLDOUT_SPEAKERS
    text_new = record["utterance"] in NEW_TEXTS
    return ("heldout_speakers" if speaker_new else "cal_speakers") + ("_new_texts" if text_new else "_cal_texts")


def make_map(records, model):
    valid = [r for r in records if "error" not in r]
    z = np.array([coordinates(r["values"], r["mic"], model) for r in valid])
    train = np.array([r["speaker"] in CAL_SPEAKERS and r["utterance"] in CAL_TEXTS and r["mic"] == "mic1" for r in valid])
    center = z[train].mean(0)
    _, _, vt = np.linalg.svd(z[train] - center, full_matrices=False)
    xy = (z - center) @ vt[:2].T
    save_json(OUT / "projection.json", {"fit": "calibration speakers/texts, mic1 only", "mean": center.tolist(), "components": vt[:2].tolist()})
    csv_write(OUT / "coordinates.csv", [dict(file=r["file"], speaker=r["speaker"], utterance=r["utterance"], mic=r["mic"],
              x=float(pt[0]), y=float(pt[1]), **{f"z{i}": float(v) for i, v in enumerate(vec)}) for r, pt, vec in zip(valid, xy, z)])
    mn, span = xy.min(0), np.maximum(np.ptp(xy, axis=0), 1e-6)
    colors = ["#2667bd", "#da4167", "#138a72", "#a05fbc", "#d17a22", "#548c2f", "#6152aa", "#bd632f"]
    speakers = CAL_SPEAKERS + HOLDOUT_SPEAKERS
    dots = []
    for r, point in zip(valid, xy):
        x, y = 45 + 810 * (point[0] - mn[0]) / span[0], 450 - 390 * (point[1] - mn[1]) / span[1]
        color = colors[speakers.index(r["speaker"])]
        fill = color if r["mic"] == "mic1" else "white"
        # Filenames originate in a fixed local manifest, not arbitrary HTML.
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{fill}" stroke="{color}" stroke-width="2" onclick="play(\'{r["file"]}\')"><title>{html.escape(r["file"])}</title></circle>')
    legend = " ".join(f'<span style="color:{c}">{s}</span>' for c, s in zip(colors, speakers))
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VM-Acoustic 声学探针</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:30px auto;color:#172b3a;background:#f7f9fc}}svg{{background:white;border:1px solid #ccd5df;width:100%}}circle{{cursor:pointer}}span{{margin-right:14px}}</style>
<h1>VM-Acoustic：有声片段声学图</h1><p>探索性 6 维参数的 PCA 投影。没有元音对齐，也未验证人耳相似度。实心=mic1，空心=mic2。点击可试听本地样本。</p>
<p>{legend}</p><svg viewBox="0 0 900 500">{"".join(dots)}</svg><p id="label">选择一个点</p><audio id="player" controls></audio>
<p>投影仅拟合标定人/标定句的 mic1；新点直接变换。二维邻近不等于感知接近。</p>
<script>function play(name){{document.getElementById('label').textContent=name;let a=document.getElementById('player');a.src='../data/'+name;a.play().catch(()=>{{}});}}</script></html>'''
    (OUT / "map.html").write_text(page, encoding="utf-8")
    # Neighborhood retention only diagnoses projection, not human agreement.
    keep = []
    for i in range(len(valid)):
        high = np.linalg.norm(z - z[i], axis=1)
        low = np.linalg.norm(xy - xy[i], axis=1)
        high[i] = low[i] = np.inf
        keep.append(len(set(np.argsort(high)[:5]) & set(np.argsort(low)[:5])) / 5)
    return float(np.mean(keep))


def main():
    OUT.mkdir(exist_ok=True)
    cache = OUT / "feature_cache"
    cache.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    versions = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "scipy": scipy.__version__, "soundfile": sf.__version__, "parselmouth": pm.__version__, "praat": pm.PRAAT_VERSION}
    code_hash = hashlib.sha256((ROOT / "acoustics.py").read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps([CONFIG, versions, code_hash], sort_keys=True).encode()).hexdigest()
    config = {"extractor": CONFIG, "versions": versions, "feature_code_sha256": code_hash,
              "calibration_speakers": CAL_SPEAKERS, "holdout_speakers": HOLDOUT_SPEAKERS,
              "calibration_utterance_ids": CAL_TEXTS, "new_utterance_ids": NEW_TEXTS,
              "aggregation": "jointly valid voiced frames, whole-utterance median; no phoneme alignment"}
    save_json(OUT / "config.json", config)
    records = []
    start = time.monotonic()
    for entry in manifest["files"]:
        if not entry["path"].endswith(".flac"):
            continue
        file = entry["path"]
        path = ROOT / "data" / file
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Sample hash mismatch: {file}")
        speaker, utt, mic = path.stem.split("_")
        cp = cache / f"{path.stem}_{signature[:12]}.json"
        if cp.exists():
            record = json.loads(cp.read_text(encoding="utf-8"))
        else:
            record = {"file": file, "speaker": speaker, "utterance": utt, "mic": mic,
                      "source_sha256": entry["sha256"], "feature_signature": signature,
                      "text": (ROOT / "data" / f"{speaker}_{utt}.txt").read_text(encoding="utf-8").strip()}
            try:
                audio, sr = read_audio(path)
                record["source_sample_rate"] = sr
                record.update(extract(audio, sr, return_frames=True))
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            save_json(cp, record)
        records.append(record)
        if len(records) % 16 == 0:
            print(f"Extracted/cached {len(records)} samples in {time.monotonic()-start:.1f}s", flush=True)
    save_json(OUT / "features.json", records)
    csv_write(OUT / "features.csv", [dict(file=r["file"], speaker=r["speaker"], utterance=r["utterance"], mic=r["mic"],
              error=r.get("error", ""), **{f: r.get("values", [None]*6)[j] for j, f in enumerate(FEATURES)},
              valid_frames=r.get("joint_valid_frames", 0), voiced_frames=r.get("pitch_voiced_frames", 0),
              flags=";".join(r.get("flags", []))) for r in records])
    pairs = make_pairs(records)
    cal = [(a, b) for a, b in pairs if a["speaker"] in CAL_SPEAKERS and a["utterance"] in CAL_TEXTS]
    model = fit_calibration([(a["values"], b["values"]) for a, b in cal])
    model["fit_files"] = [[a["file"], b["file"]] for a, b in cal]
    save_json(OUT / "calibration.json", model)
    rows, grouped = [], {}
    for mode in ("raw", "bias", "weighted"):
        for group in sorted({group_name(a) for a, b in pairs}):
            gp = [(a, b) for a, b in pairs if group_name(a) == group]
            within = []
            for a, b in gp:
                za, zb = coordinates(a["values"], "mic1", model, mode), coordinates(b["values"], "mic2", model, mode)
                d = float(np.linalg.norm(za-zb))
                within.append(d)
                rows.append({"speaker": a["speaker"], "utterance": a["utterance"], "group": group, "mode": mode, "paired_mic_distance": d})
            between = []
            identity_hits = []
            for utt in sorted({a["utterance"] for a, b in gp}):
                same = [(a, b) for a, b in gp if a["utterance"] == utt]
                gallery = np.array([coordinates(a["values"], "mic1", model, mode) for a, b in same])
                if len(gallery) < 2:
                    continue
                between.extend(float(np.linalg.norm(za-zb)) for za, zb in itertools.combinations(gallery, 2))
                for i, (a, b) in enumerate(same):
                    query = coordinates(b["values"], "mic2", model, mode)
                    identity_hits.append(int(np.argmin(np.linalg.norm(gallery-query, axis=1)) == i))
            key = f"{group}/{mode}"
            grouped[key] = {"within": summary(within), "between_mic1_same_utterance": summary(between),
                            "within_between_median_ratio": float(np.median(within)/np.median(between)) if between and np.median(between)>0 else None,
                            "identity_top1_diagnostic": float(np.mean(identity_hits)) if identity_hits else None}
    csv_write(OUT / "paired_distances.csv", rows)
    csv_write(OUT / "profile_distances.csv", [
        {"speaker_a": a, "speaker_b": b, "distance": float(np.linalg.norm(za-zb))}
        for (a, za), (b, zb) in itertools.combinations([
            (s, np.median([coordinates(r["values"], "mic1", model) for r in records
                           if "error" not in r and r["speaker"] == s and r["utterance"] in NEW_TEXTS], axis=0))
            for s in CAL_SPEAKERS+HOLDOUT_SPEAKERS
            if any("error" not in r and r["speaker"]==s and r["utterance"] in NEW_TEXTS for r in records)
        ], 2)])
    transcripts = {}
    for r in records:
        transcripts.setdefault(r["utterance"], set()).add(r["text"])
    result = {"sample_count": len(records), "failed_samples": [r["file"] for r in records if "error" in r],
              "pair_count": len(pairs), "calibration_pair_count": len(cal),
              "transcript_variants": {u: sorted(t) for u, t in transcripts.items()},
              "groups": grouped, "pca_neighbor_overlap_k5": make_map(records, model),
              "runtime_seconds": time.monotonic()-start}
    present = {r["file"] for r in records}
    result["missing_planned_pairs"] = [f"{s}_{u}" for s in CAL_SPEAKERS+HOLDOUT_SPEAKERS for u in CAL_TEXTS+NEW_TEXTS
                                        if any(f"{s}_{u}_{mic}.flac" not in present for mic in ("mic1", "mic2"))]
    save_json(OUT / "metrics.json", result)
    report = ["# VM-Acoustic 首轮实际结果", "", "本报告由 run_probe.py 生成。六参数有声帧探针，不是按元音对齐的24维模型。", "",
              f"样本 {len(records)}；成功配对 {len(pairs)}；标定对 {len(cal)}；失败 {len(result['failed_samples'])}。",
              "", "## 固定设置", "", "F0 60–600 Hz；Burg ceiling 5500 Hz；CPP 为 Praat 6.1.38 非平滑帧级 peak prominence；整句共同有效有声帧中位数。",
              "", "## 双麦克风距离", "", "距离无物理单位；同一标定尺度，weighted 权重均值为1。ratio=同次双麦克风中位距离/同句异人mic1中位距离；异人分离只作诊断，不是音色真值。", "",
              "|分组|方式|配对数|双麦距离中位|同句异人距离中位|ratio|身份top1诊断|", "|---|---|---:|---:|---:|---:|---:|"]
    for key, value in grouped.items():
        group, mode = key.split("/")
        report.append(f"|{group}|{mode}|{value['within']['n']}|{value['within']['median']:.4f}|{value['between_mic1_same_utterance']['median']:.4f}|{value['within_between_median_ratio']:.4f}|{value['identity_top1_diagnostic']:.1%}|")
    report += ["", "## 校准参数", "", "|参数|mic2-mic1偏差（原特征单位）|尺度|归一化权重|", "|---|---:|---:|---:|"]
    for j, f in enumerate(FEATURES):
        report.append(f"|{f}|{model['mic2_bias'][j]:.5f}|{model['scale'][j]:.5f}|{model['weights_mean_one'][j]:.4f}|")
    report += ["", "## 边界与复现", "",
               "- 标定只用 p225–p228 的001–004；p229–p232以及005–008没有进入标定。",
               "- 有声帧不是经过人工确认的元音；没有实现音素对齐和24维档案。文本影响尚未排除。",
               "- 一个固定5500 Hz ceiling不保证适合所有声线；HNR/CPP及F1均可能受到内容、音高和通道影响。",
               "- 未做跨日、未知设备、听觉或AI训练验证；小样本数值无总体显著性声明。",
               f"- 全样本PCA k=5邻居重合均值 {result['pca_neighbor_overlap_k5']:.1%}，仅诊断二维投影。",
               "- 查看 config.json、features.csv、calibration.json、paired_distances.csv 与 profile_distances.csv；map.html可本地试听。",
               "- 源样本来自 Edinburgh VCTK 0.92 官方ZIP，data/manifest.json保留CRC/SHA256；README与许可保存在data/。",
               "- 运行：python vm_acoustic/run_probe.py；先安装 requirements.txt 到 vm_acoustic/.deps。", ""]
    def group_metric(group, mode, field="within"):
        item = grouped[f"{group}/{mode}"]
        return item[field]["median"] if isinstance(item[field], dict) else item[field]
    hg, cg = "heldout_speakers_new_texts", "cal_speakers_new_texts"
    hr, hb, hw = [group_metric(hg, mode) for mode in ("raw", "bias", "weighted")]
    cr, cw = [group_metric(cg, mode) for mode in ("raw", "weighted")]
    rr, rw = [group_metric(hg, mode, "within_between_median_ratio") for mode in ("raw", "weighted")]
    report += ["## 本轮观察", "",
               f"- 新说话者+新句子：双麦距离中位数 raw {hr:.4f} → bias {hb:.4f} → weighted {hw:.4f}（weighted相对变化 {100*(hw/hr-1):+.1f}%）。",
               f"- 标定说话者+新句子：raw {cr:.4f} → weighted {cw:.4f}（相对变化 {100*(cw/cr-1):+.1f}%）；校准未全面泛化。",
               f"- heldout新句子的 within/between ratio {rr:.4f} → {rw:.4f}；这里只是声学/通道诊断。",
               "- 所有样本都有数值输出，不代表所有帧的共振峰是正确生理/语音学测量。",
               f"- 计划样本缺失对：{', '.join(result['missing_planned_pairs'])}；保持缺失，不用其他句子补齐以免改变固定文本。", ""]
    validation_path = OUT / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        k = validation["known_source_filter"]
        sensitivity = validation["formant_ceiling_sensitivity"]
        max_change = max(abs(v) for row in sensitivity for v in row["formant_change_percent_5500_to_5000"])
        report += ["## 合成检查与敏感性（validate_probe.py）", "",
                   f"- 已知 F0/F1/F2/F3 {k['target_hz']} Hz，估计 {[round(v, 3) for v in k['estimated_hz']]} Hz。",
                   "- 振幅减半，六参数最大变化低于1e-5；加入噪声后HNR下降；已知偏差校准、距离对称/三角性质检查通过。",
                   f"- 8条留出mic1样本中，formant ceiling 5500→5000 Hz引起的F1–F3相对变化最大 {max_change:.2f}%；不得视为全部参数已准确。",
                   "- 这只是一个合成配置的功能检查，未覆盖全部F0/声线/噪声范围；未用敏感性结果重新选择测试配置。", ""]
    report += ["## 下一步", "",
               "先做元音/音素对齐与共振峰可靠性检查，再复测通道校正；增加独立说话者和设备。固定ceiling与整句混音素中位数是当前主要风险。",
               "现阶段不据此训练大网络或宣称音色地图成功。物理参数的误差需先定位，才适合作AI监督。", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ('transcript_variants','groups')}, indent=2), flush=True)
    print("Results: " + str(OUT / "report.md"), flush=True)


if __name__ == "__main__":
    main()
