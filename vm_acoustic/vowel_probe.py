"""Phone-conditioned paired-microphone diagnostic; preserves original results.

Run after run_probe.py. Automatic US-English alignment is NOT a phonetic label.
"""
import hashlib
import json
import re
from collections import defaultdict
from math import gcd

from acoustics import ROOT, np, read_audio, extract, fit_calibration, coordinates
from scipy.signal import resample_poly
from pocketsphinx import Decoder
from run_probe import CAL_SPEAKERS, CAL_TEXTS, group_name, save_json, csv_write, summary

OUT = ROOT / "results_vowels"
VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())


def align(record):
    cache = OUT / "alignments" / (record["file"] + ".json")
    signature = hashlib.sha256((record["source_sha256"] + record["text"] + "pocketsphinx-5.0.4-v1").encode()).hexdigest()
    if cache.exists():
        result = json.loads(cache.read_text())
        if result["signature"] == signature:
            return result["phones"]
    y, sr = read_audio(ROOT / "data" / record["file"])
    g = gcd(sr, 16000)
    y = resample_poly(y, 16000 // g, sr // g)
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    decoder = Decoder(samprate=16000, loglevel="ERROR")
    decoder.set_align_text(" ".join(re.findall("[a-z']+", record["text"].lower())))
    for subword in (False, True):
        if subword:
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(pcm, full_utt=True)
        decoder.end_utt()
    phones, wi = [], 0
    for word in decoder.get_alignment():
        if word.name.startswith("<"):
            continue
        for pi, phone in enumerate(word):
            label = re.sub(r"\d", "", phone.name)
            if label in VOWELS:
                start, end = phone.start / 100, (phone.start + phone.duration) / 100
                if not 0 <= start < end <= len(y) / 16000 + .02:
                    raise ValueError("Invalid alignment interval")
                phones.append(dict(word_index=wi, phone_index=pi, word=word.name,
                                   vowel=label, start=start, end=end))
        wi += 1
    save_json(cache, dict(signature=signature, phones=phones))
    return phones


def token_values(record, phones):
    frames = record["frames"]
    t, v = np.array(frames["times"]), np.array(frames["values"])
    out = {}
    for phone in phones:
        a, b = phone["start"], phone["end"]
        margin = .2 * (b - a)
        mask = (t >= a + margin) & (t <= b - margin)
        if mask.sum() >= 3:
            key = (phone["word_index"], phone["phone_index"], phone["word"], phone["vowel"])
            out[key] = dict(phone, values=np.median(v[mask], axis=0).tolist(), frames=int(mask.sum()))
    return out


def ceiling_diagnostic(records):
    results = []
    for r in records:
        if r["mic"] != "mic1" or r["speaker"] in CAL_SPEAKERS or r["utterance"] not in ("001", "002"):
            continue
        y, sr = read_audio(ROOT / "data" / r["file"])
        lower = extract(y, sr, formant_ceiling=5000, return_frames=True)
        a = {round(t, 6): v for t, v in zip(r["frames"]["times"], r["frames"]["values"])}
        b = {round(t, 6): v for t, v in zip(lower["frames"]["times"], lower["frames"]["values"])}
        common = sorted(a.keys() & b.keys())
        av, bv = np.array([a[t] for t in common]), np.array([b[t] for t in common])
        # Exact additive decomposition in log-Hz: frame selection, fixed-frame
        # estimator change, then lower-setting frame selection.
        old, new = r["values"][2], lower["values"][2]
        ca, cb = np.median(av[:, 2]), np.median(bv[:, 2])
        jumps = np.abs(np.expm1(bv[:, 2] - av[:, 2])) > .1
        phones = align(r)
        token_changes = []
        for phone in phones:
            margin = .2 * (phone["end"] - phone["start"])
            mask = (np.array(common) >= phone["start"] + margin) & (np.array(common) <= phone["end"] - margin)
            if mask.sum() >= 3:
                token_changes.append(dict(phone, frames=int(mask.sum()),
                    f2_5500_hz=float(np.exp(np.median(av[mask, 2]))),
                    f2_5000_hz=float(np.exp(np.median(bv[mask, 2]))),
                    f2_change_pct=float(100 * np.expm1(np.median(bv[mask, 2]) - np.median(av[mask, 2])))))
        results.append(dict(file=r["file"], old_frames=len(a), new_frames=len(b), common_frames=len(common),
            f2_whole_change_pct=float(100*np.expm1(new-old)),
            f2_common_change_pct=float(100*np.expm1(cb-ca)),
            log_decomposition=[float(ca-old), float(cb-ca), float(new-cb)],
            fraction_common_frames_f2_changes_over_10pct=float(jumps.mean()), tokens=token_changes))
        assert abs(sum(results[-1]["log_decomposition"]) - (new-old)) < 1e-10
    save_json(OUT / "ceiling_diagnostic.json", results)
    return results


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "alignments").mkdir(exist_ok=True)
    records = json.loads((ROOT / "results/features.json").read_text(encoding="utf-8"))
    records = [r for r in records if "error" not in r]
    lookup = {(r["speaker"], r["utterance"], r["mic"]): r for r in records}
    tokens, failures = {}, []
    for i, r in enumerate(records):
        try:
            tokens[r["file"]] = token_values(r, align(r))
        except Exception as e:
            failures.append(dict(file=r["file"], error=str(e)))
            tokens[r["file"]] = {}
        if i % 16 == 0:
            print(f"aligned {i+1}/{len(records)}", flush=True)
    pairs = []
    for (speaker, utt, mic), a in sorted(lookup.items()):
        if mic != "mic1" or (speaker, utt, "mic2") not in lookup:
            continue
        b = lookup[speaker, utt, "mic2"]
        ta, tb = tokens[a["file"]], tokens[b["file"]]
        for key in sorted(ta.keys() & tb.keys()):
            pa, pb = ta[key], tb[key]
            # Independent alignment: reject gross boundary disagreements.
            if max(abs(pa["start"]-pb["start"]), abs(pa["end"]-pb["end"])) > .08:
                continue
            pairs.append(dict(speaker=speaker, utterance=utt, vowel=key[-1], token=list(key),
                              a=pa["values"], b=pb["values"], group=group_name(a)))
    models = {}
    for vowel in sorted(VOWELS):
        cal = [p for p in pairs if p["vowel"] == vowel and p["speaker"] in CAL_SPEAKERS and p["utterance"] in CAL_TEXTS]
        if len(cal) >= 4:
            models[vowel] = fit_calibration([(p["a"], p["b"]) for p in cal])
    save_json(OUT / "calibration.json", models)
    save_json(OUT / "tokens.json", dict(records={f: list(t.values()) for f, t in tokens.items()}, failures=failures, matched_pairs=pairs))
    original_model = json.loads((ROOT / "results/calibration.json").read_text())
    print("original calibration keys", list(original_model), flush=True)
    # Existing file is a direct calibration dictionary.
    if "center" not in original_model:
        original_model = original_model["model"]
    rows = []
    for p in pairs:
        if p["vowel"] not in models:
            continue
        model = models[p["vowel"]]
        a = lookup[p["speaker"], p["utterance"], "mic1"]
        b = lookup[p["speaker"], p["utterance"], "mic2"]
        row = dict(speaker=p["speaker"], utterance=p["utterance"], vowel=p["vowel"], token=str(p["token"]), group=p["group"])
        for mode in ("raw", "bias", "weighted"):
            row["vowel_"+mode] = float(np.linalg.norm(coordinates(p["a"], "mic1", model, mode)-coordinates(p["b"], "mic2", model, mode)))
            row["sentence_"+mode] = float(np.linalg.norm(coordinates(a["values"], "mic1", original_model, mode)-coordinates(b["values"], "mic2", original_model, mode)))
        # Direct measurement repeatability with identical global scale and no correction.
        row["vowel_common_scale_raw"] = float(np.linalg.norm((np.array(p["b"])-p["a"])/original_model["scale"])/np.sqrt(6))
        row["vowel_common_scale_bias"] = float(np.linalg.norm((np.array(p["b"])-p["a"]-model["mic2_bias"])/original_model["scale"])/np.sqrt(6))
        rows.append(row)
    csv_write(OUT / "paired_distances.csv", rows)
    # Aggregate tokens to one value per utterance before reporting, so many
    # phones within a sentence are not counted as independent replications.
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["speaker"], r["utterance"], r["group"]].append(r)
    sentence_rows = []
    measures = [k for k in rows[0] if k.startswith(("vowel_", "sentence_"))]
    for (speaker, utt, group), rr in grouped.items():
        sentence_rows.append(dict(speaker=speaker, utterance=utt, group=group, token_count=len(rr),
            **{m: float(np.median([r[m] for r in rr])) for m in measures}))
    csv_write(OUT / "sentence_aggregated.csv", sentence_rows)
    metrics = {g: {m: summary([r[m] for r in sentence_rows if r["group"]==g]) for m in measures}
               for g in sorted(set(r["group"] for r in sentence_rows))}
    save_json(OUT / "metrics.json", metrics)
    diagnostic = ceiling_diagnostic(records)
    lines = ["# VM-Acoustic 自动元音对齐诊断", "", "PocketSphinx 5.0.4，美国英语默认HMM/词典；已知文本两遍强制对齐。不是人工音素真值。",
             "每token取中央60%、每麦至少3个有效声学帧；两麦独立对齐，词/音素序号/标签匹配且边界差≤80ms。",
             "固定5500Hz正式配置；5000Hz仅作诊断。原始整句结果保留。",
             f"对齐失败 {len(failures)}；匹配token对 {len(pairs)}；可标定元音 {sorted(models)}。",
             "各元音仅用原划分的标定人/标定句拟合，至少4对；未对留出集挑参数。",
             "", "下表先在每句内取token距离中位数，再跨句取中位数；不是独立token样本量。",
             "", "|条件|句对数|整句raw|整句weighted|元音raw（自身尺度）|元音weighted（自身尺度）|元音raw（整句尺度）|", "|---|---:|---:|---:|---:|---:|---:|"]
    for g, mm in metrics.items():
        vals = [mm[m]["median"] for m in ("sentence_raw", "sentence_weighted", "vowel_raw", "vowel_weighted", "vowel_common_scale_raw")]
        lines.append(f"|{g}|{mm['sentence_raw']['n']}|"+"|".join(f"{v:.4f}" for v in vals)+"|")
    lines += ["", "不能直接将自身尺度的元音距离与整句距离比较；公共尺度raw才可判断测量重复性。",
              "短token比整句样本少；三帧不是三个独立观测。不同元音覆盖不等，缺失不填补。",
              "本轮未证明音色感知精度、跨日稳定性或完整24维地图成立。", "", "## F2 上限诊断", "", "|文件|整句变化%|固定共同帧变化%|共同帧F2变化>10%的比例|", "|---|---:|---:|---:|"]
    for d in diagnostic:
        lines.append(f"|{d['file']}|{d['f2_whole_change_pct']:.2f}|{d['f2_common_change_pct']:.2f}|{d['fraction_common_frames_f2_changes_over_10pct']:.3f}|")
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save_json(OUT / "config.json", dict(aligner="pocketsphinx 5.0.4 default en-US HMM", central_fraction=.6,
        minimum_frames=3, boundary_tolerance_s=.08, min_calibration_pairs=4,
        script_sha256=hashlib.sha256((ROOT/"vowel_probe.py").read_bytes()).hexdigest(),
        feature_source_sha256=hashlib.sha256((ROOT/"results/features.json").read_bytes()).hexdigest()))
    print("Finished", OUT, flush=True)


if __name__ == "__main__":
    main()
