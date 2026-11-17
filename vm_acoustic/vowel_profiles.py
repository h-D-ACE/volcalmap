"""Aggregate matched vowel tokens within each utterance before calibration.

Companion to vowel_probe.py. No parameter search or held-out model selection.
"""
import json
from collections import defaultdict
from itertools import combinations

from acoustics import ROOT, FEATURES, np, fit_calibration, coordinates
from run_probe import CAL_SPEAKERS, CAL_TEXTS, save_json, csv_write, summary

OUT = ROOT / "results_vowels"


def main():
    data = json.loads((OUT / "tokens.json").read_text())
    groups = defaultdict(list)
    for p in data["matched_pairs"]:
        groups[p["speaker"], p["utterance"], p["vowel"]].append(p)
    profiles = []
    for (speaker, utt, vowel), pp in groups.items():
        profiles.append(dict(speaker=speaker, utterance=utt, vowel=vowel, group=pp[0]["group"],
            a=np.median([p["a"] for p in pp], axis=0).tolist(),
            b=np.median([p["b"] for p in pp], axis=0).tolist(), tokens=len(pp)))
    models = {}
    for v in sorted(set(p["vowel"] for p in profiles)):
        cal = [p for p in profiles if p["vowel"] == v and p["speaker"] in CAL_SPEAKERS and p["utterance"] in CAL_TEXTS]
        if len(cal) >= 4 and len(set(p["speaker"] for p in cal)) >= 2:
            models[v] = fit_calibration([(p["a"], p["b"]) for p in cal])
            models[v]["fit_keys"] = [[p["speaker"], p["utterance"], v] for p in cal]
    original = json.loads((ROOT / "results/calibration.json").read_text())
    records = json.loads((ROOT / "results/features.json").read_text())
    lookup = {(r["speaker"], r["utterance"], r["mic"]): r for r in records if "error" not in r}
    profile_rows, sentence = [], defaultdict(list)
    for p in profiles:
        if p["vowel"] not in models:
            continue
        m = models[p["vowel"]]
        r = {k: p[k] for k in ("speaker", "utterance", "vowel", "group", "tokens")}
        delta = np.array(p["b"])-p["a"]
        for mode in ("raw", "bias", "weighted"):
            r[mode] = float(np.linalg.norm(coordinates(p["a"], "mic1", m, mode)-coordinates(p["b"], "mic2", m, mode)))
        r["common_scale_raw"] = float(np.linalg.norm(delta / original["scale"])/np.sqrt(6))
        r["common_scale_bias"] = float(np.linalg.norm((delta-m["mic2_bias"])/original["scale"])/np.sqrt(6))
        for i, feature in enumerate(FEATURES):
            r["abs_delta_"+feature] = float(abs(delta[i]))
        profile_rows.append(r)
        sentence[p["speaker"], p["utterance"], p["group"]].append(r)
    rows = []
    for (speaker, utt, group), vv in sentence.items():
        a, b = [lookup[speaker, utt, mic] for mic in ("mic1", "mic2")]
        r = dict(speaker=speaker, utterance=utt, group=group, vowels=len(vv))
        for mode in ("raw", "bias", "weighted"):
            r["sentence_"+mode] = float(np.linalg.norm(coordinates(a["values"], "mic1", original, mode)-coordinates(b["values"], "mic2", original, mode)))
        for key in ("raw", "bias", "weighted", "common_scale_raw", "common_scale_bias"):
            # Equal weight per observed vowel; distance RMS, not sample count.
            r["vowel_"+key] = float(np.sqrt(np.mean([v[key]**2 for v in vv])))
        rows.append(r)
    metrics = {g: {key: summary([r[key] for r in rows if r["group"] == g])
                      for key in rows[0] if key.startswith(("sentence_", "vowel_"))}
               for g in sorted(set(r["group"] for r in rows))}
    # Scale-invariant diagnostic, within/between same utterance and vowel,
    # using the SAME eligible pair set for both sentence and vowel methods.
    ratios = {}
    bygroup = defaultdict(list)
    for p in profiles:
        if p["vowel"] in models:
            bygroup[p["group"]].append(p)
    for group, pp in bygroup.items():
        ratios[group] = {}
        for method in ("sentence", "vowel"):
            within, between = [], []
            for a, b in combinations(pp, 2):
                if a["speaker"] == b["speaker"] or a["utterance"] != b["utterance"] or a["vowel"] != b["vowel"]:
                    continue
                def z(p, mic):
                    q = p["a"] if mic == "mic1" else p["b"]
                    model = models[p["vowel"]]
                    if method == "sentence":
                        q = lookup[p["speaker"], p["utterance"], mic]["values"]
                        model = original
                    return coordinates(q, mic, model, "weighted")
                within.extend([float(np.linalg.norm(z(a,"mic1")-z(a,"mic2"))),float(np.linalg.norm(z(b,"mic1")-z(b,"mic2")))])
                between.append(float(np.linalg.norm(z(a,"mic1")-z(b,"mic1"))))
            ratios[group][method] = dict(pair_comparisons=len(between),
                ratio=float(np.median(within)/np.median(between)) if between else None)
    save_json(OUT / "profile_calibration.json", models)
    save_json(OUT / "profiles.json", profiles)
    save_json(OUT / "profile_metrics.json", dict(metrics=metrics, within_between_diagnostic=ratios))
    csv_write(OUT / "profile_distances.csv", profile_rows)
    csv_write(OUT / "profile_sentence_distances.csv", rows)
    lines = ["# 句内同元音汇总对照", "", "先对同一句、同元音的匹配token取六参数中位数，再重新拟合每元音双麦偏差。",
             "每元音至少4个标定人/标定句profile且至少2人；各句对可用元音等权取距离RMS。缺失不补。",
             "句间比较使用完全相同的句对集合；元音覆盖不完全相同，不能当成完整固定维数音色地图。",
             "", "|条件|句对|整句raw|整句weighted|元音raw（公共尺度）|元音bias（公共尺度）|元音weighted（自身尺度）|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for g, mm in metrics.items():
        vals = [mm[k]["median"] for k in ("sentence_raw", "sentence_weighted", "vowel_common_scale_raw", "vowel_common_scale_bias", "vowel_weighted")]
        lines.append(f"|{g}|{mm['sentence_raw']['n']}|"+"|".join(f"{v:.4f}" for v in vals)+"|")
    lines += ["", "## 尺度不变的辅助诊断", "", "同句同元音、不同人mic1距离作为分母，双麦距离作为分子。使用相同的异人配对集合；含重复句、元音、说话者，不能当独立样本检验。异人分离不是音色相似真值。",
              "", "|条件|整句 within/between|元音 within/between|", "|---|---:|---:|"]
    for g, v in sorted(ratios.items()):
        if v["sentence"]["ratio"] is not None:
            lines.append(f"|{g}|{v['sentence']['ratio']:.4f}|{v['vowel']['ratio']:.4f}|")
    lines += ["", "结论以公共尺度的重复性与留出结果为依据，不能因自身尺度距离较小就宣称改进。",
              "短元音有效样本少、HMM英美口音不匹配、音素边界误差、协同发音、共振峰估计敏感性仍混合存在。",
              "这是对照而非通过测试集调参后的新最佳模型。"]
    (OUT / "profile_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Profile experiment complete", flush=True)


if __name__ == "__main__":
    main()
