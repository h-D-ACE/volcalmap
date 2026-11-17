"""Audit split integrity and sensitivity to 10-ms boundary shifts, no refit."""
import json
from acoustics import ROOT, np
from run_probe import CAL_SPEAKERS, CAL_TEXTS, save_json, summary
from vowel_probe import token_values


def main():
    out = ROOT / "results_vowels"
    models = json.loads((out / "profile_calibration.json").read_text())
    for model in models.values():
        assert all(s in CAL_SPEAKERS and u in CAL_TEXTS for s, u, v in model["fit_keys"])
        assert abs(np.mean(model["weights_mean_one"])-1) < 1e-12
        assert np.all(np.array(model["scale"]) > 0)
    diagnostic = json.loads((out / "ceiling_diagnostic.json").read_text())
    for r in diagnostic:
        assert abs(100*np.expm1(sum(r["log_decomposition"]))-r["f2_whole_change_pct"]) < 1e-8
    records = json.loads((ROOT / "results/features.json").read_text())
    tokens = json.loads((out / "tokens.json").read_text())
    original = json.loads((ROOT / "results/calibration.json").read_text())
    sensitivity = {}
    for shift in (-.01, .01):
        distances, missing, total = [], 0, 0
        for r in records:
            if r["speaker"] in CAL_SPEAKERS or "error" in r:
                continue
            cached = out / "alignments" / (r["file"]+".json")
            if not cached.exists():
                continue
            phones = json.loads(cached.read_text())["phones"]
            shifted = [dict(p, start=p["start"]+shift, end=p["end"]+shift) for p in phones]
            values = token_values(r, shifted)
            for p in tokens["records"][r["file"]]:
                total += 1
                key = (p["word_index"], p["phone_index"], p["word"], p["vowel"])
                if key not in values:
                    missing += 1
                    continue
                delta = (np.array(p["values"])-values[key]["values"])/original["scale"]
                distances.append(float(np.linalg.norm(delta)/np.sqrt(6)))
        sensitivity[str(shift)] = dict(common_scale_parameter_distance=summary(distances),
                                      original_tokens=total, become_invalid=missing)
    result = dict(split_integrity="passed", positive_scales="passed", mean_one_weights="passed",
                  ceiling_decomposition="passed", boundary_shift_diagnostic=sensitivity)
    save_json(out / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
