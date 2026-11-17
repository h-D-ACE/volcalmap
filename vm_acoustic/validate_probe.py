"""Independent numerical checks and explicitly exploratory real-audio sensitivity tests."""
from __future__ import annotations

import json
import sys
from acoustics import ROOT, np, extract, read_audio, fit_calibration, coordinates, FEATURES
from scipy import signal


def source_filter(f0=120.0, snr_db=40.0, seed=17):
    sr = 24000
    t = np.arange(sr*2) / sr
    # Known harmonic source and five stable all-pole resonances; no Praat synthesis.
    excitation = sum(np.sin(2*np.pi*f0*k*t) / k for k in range(1, int(10000/f0)))
    denominator = np.array([1.0])
    for formant, bandwidth in zip([500, 1500, 2500, 3500, 4500], [70, 100, 120, 180, 220]):
        r = np.exp(-np.pi*bandwidth/sr)
        denominator = np.convolve(denominator, [1, -2*r*np.cos(2*np.pi*formant/sr), r*r])
    clean = signal.lfilter([1.0], denominator, excitation)[sr//4:]
    clean = clean / np.max(np.abs(clean)) * 0.4
    noise = np.random.default_rng(seed).normal(size=len(clean))
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2)) * 10**(-snr_db/20)
    return clean + noise, sr


def main():
    results = {}
    clean, sr = source_filter()
    base = extract(clean, sr)
    estimated = np.exp(base["values"][:4])
    target = np.array([120, 500, 1500, 2500])
    relative = np.abs(estimated-target)/target
    # Functional checks have bounds; scientific validity is NOT inferred from passing.
    assert relative[0] < .03, f"Known F0 recovery failed: {estimated[0]}"
    assert np.all(relative[1:] < .20), f"Known resonances poorly recovered: {estimated[1:]}"
    results["known_source_filter"] = {"target_hz": target.tolist(), "estimated_hz": estimated.tolist(),
                                      "relative_errors": relative.tolist(),
                                      "functional_bounds": "F0 <3%, F1-F3 <20%; one synthetic configuration only"}
    gain = extract(clean*.5, sr)
    gain_diff = np.abs(np.array(gain["values"])-base["values"])
    assert np.max(gain_diff) < 1e-5, f"Unexpected gain sensitivity: {gain_diff}"
    results["half_amplitude"] = {"absolute_feature_change": gain_diff.tolist(), "passed": True}
    noisy, _ = source_filter(snr_db=10)
    noisy_features = extract(noisy, sr)
    results["added_noise"] = {"clean_cpp_hnr": base["values"][4:], "noisy_cpp_hnr": noisy_features["values"][4:]}
    assert noisy_features["values"][5] < base["values"][5], "HNR did not respond to known noise"
    higher, _ = source_filter(f0=180)
    higher_features = extract(higher, sr)
    assert abs(np.exp(higher_features["values"][0])-180)/180 < .03
    results["f0_intervention"] = {"target": 180, "estimate": float(np.exp(higher_features["values"][0]))}
    # Calibration algebra, including held-out values NOT changing an existing fit.
    a = np.arange(36, dtype=float).reshape(6, 6)/10
    offset = np.array([.1, .2, .1, -.1, 1., 2.])
    fitted = fit_calibration(list(zip(a, a+offset)))
    assert np.allclose(fitted["mic2_bias"], offset)
    for q in a:
        assert np.allclose(coordinates(q, "mic1", fitted), coordinates(q+offset, "mic2", fitted))
    points = np.array([coordinates(q, "mic1", fitted) for q in a])
    dist = np.linalg.norm(points[:, None]-points[None, :], axis=2)
    assert np.allclose(dist, dist.T) and np.allclose(np.diag(dist), 0)
    assert np.all(dist[:, :, None] <= dist[:, None, :] + dist[None, :, :] + 1e-10)
    try:
        coordinates(a[0], "unknown", fitted)
        raise AssertionError("Unknown microphone incorrectly accepted")
    except ValueError:
        pass
    results["calibration_algebra"] = "passed known-offset removal, Euclidean symmetry/triangle, unknown-channel rejection"
    # Held-out examples only: measure sensitivity, never choose ceiling using these results.
    baseline = json.loads((ROOT / "results/features.json").read_text(encoding="utf-8"))
    sensitivity = []
    for speaker in ("p229", "p230", "p231", "p232"):
        for utt in ("001", "002"):
            name = f"{speaker}_{utt}_mic1.flac"
            rec = next((r for r in baseline if r["file"] == name and "error" not in r), None)
            if rec is None:
                continue
            audio, sr = read_audio(ROOT / "data" / name)
            other = extract(audio, sr, formant_ceiling=5000)
            relative_hz = (np.exp(np.array(other["values"][1:4])-rec["values"][1:4])-1)*100
            sensitivity.append({"file": name, "formant_change_percent_5500_to_5000": relative_hz.tolist()})
    results["formant_ceiling_sensitivity"] = sensitivity
    results["limitations"] = ["One matched synthesis model; not independent anatomical ground truth",
                              "CPP/HNR have implementation-specific definitions",
                              "Whole-utterance voiced-frame medians, not vowel-aligned parameters",
                              "The 5000-Hz run is a sensitivity diagnostic, not an alternative selected on test data"]
    path = ROOT / "results/validation.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
