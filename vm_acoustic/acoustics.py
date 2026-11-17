"""Fixed six-parameter voiced-speech probe, NOT a phoneme-conditioned timbre model."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
import parselmouth as pm
from parselmouth.praat import call
from scipy.signal import resample_poly
import soundfile as sf
from math import gcd

FEATURES = ["log_f0", "log_f1", "log_f2", "log_f3", "cpp_db", "hnr_db"]
CONFIG = {
    "sample_rate": 24000, "step_s": 0.01, "pitch_floor_hz": 60,
    "pitch_ceiling_hz": 600, "formant_ceiling_hz": 5500,
    "formant_window_s": 0.025, "formant_count": 5,
    "cpp_max_frequency_hz": 5000, "cpp_preemphasis_hz": 50,
    "cpp_trend_quefrency_s": [0.001, 0.0],
    "cpp_interpolation": "Parabolic", "cpp_trend": "Straight", "cpp_fit": "Robust",
    "minimum_joint_frames": 20,
}


def read_audio(path):
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("Expected mono source; do not silently mix channels")
    return audio[:, 0], sr


def extract(audio, sr, formant_ceiling=None, return_frames=False):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not np.isfinite(audio).all() or len(audio) < sr * 0.25:
        raise ValueError("Invalid, non-mono or too-short audio")
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    if sr != CONFIG["sample_rate"]:
        g = gcd(int(sr), CONFIG["sample_rate"])
        audio = resample_poly(audio, CONFIG["sample_rate"] // g, int(sr) // g)
    audio = audio - np.mean(audio)
    snd = pm.Sound(audio, sampling_frequency=CONFIG["sample_rate"])
    pitch = snd.to_pitch_ac(time_step=CONFIG["step_s"], pitch_floor=60, pitch_ceiling=600,
                            voicing_threshold=0.45, silence_threshold=0.03)
    formants = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                  maximum_formant=formant_ceiling or 5500,
                                  window_length=0.025, pre_emphasis_from=50)
    harmonicity = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60,
                                       silence_threshold=0.1, periods_per_window=4.5)
    cepstrogram = call(snd, "To PowerCepstrogram", 60, 0.01, 5000, 50)
    rows, times = [], []
    voiced_count, formant_rejects = 0, 0
    for t, f0 in zip(pitch.xs(), pitch.selected_array["frequency"]):
        if not np.isfinite(f0) or f0 <= 0:
            continue
        voiced_count += 1
        fs = [formants.get_value_at_time(i, t) for i in (1, 2, 3)]
        if not (all(np.isfinite(fs)) and 150 < fs[0] < fs[1] < fs[2] < 5000):
            formant_rejects += 1
            continue
        hnr = harmonicity.get_value(t)
        if not np.isfinite(hnr) or hnr <= -100:
            continue
        cepstrum = call(cepstrogram, "To PowerCepstrum (slice)", float(t))
        cpp = call(cepstrum, "Get peak prominence", 60, 600, "Parabolic", 0.001, 0.0, "Straight", "Robust")
        if not np.isfinite(cpp):
            continue
        rows.append([np.log(f0), *np.log(fs), cpp, hnr])
        times.append(float(t))
    if len(rows) < CONFIG["minimum_joint_frames"]:
        raise ValueError(f"Only {len(rows)} jointly valid frames (need 20)")
    values = np.asarray(rows)
    flags = []
    if clipping > 0.001:
        flags.append("clipping")
    if formant_rejects > 0.1 * max(1, voiced_count):
        flags.append("many_formant_rejections")
    result = {
        "values": np.median(values, axis=0).tolist(),
        "iqr": (np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)).tolist(),
        "duration_s": len(audio) / CONFIG["sample_rate"],
        "pitch_voiced_frames": voiced_count, "joint_valid_frames": len(rows),
        "formant_rejections": formant_rejects,
        "valid_fraction_of_pitch_voiced": len(rows) / max(1, voiced_count),
        "clipped_fraction": clipping, "flags": flags,
    }
    if return_frames:
        result["frames"] = {"times": times, "values": values.tolist()}
    return result


def fit_calibration(pairs):
    """Fit ONLY supplied calibration pairs; no labels or held-out samples used."""
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    if len(a) < 4 or a.shape[1] != 6:
        raise ValueError("Insufficient calibration pairs")
    center = np.median(a, axis=0)
    floors = np.array([0.02, 0.03, 0.03, 0.03, 0.5, 1.0])
    scale = np.maximum((np.percentile(a, 75, axis=0) - np.percentile(a, 25, axis=0)) / 1.349, floors)
    bias = np.median(b - a, axis=0)
    delta = (b - a - bias) / scale
    variance = (1.4826 * np.median(np.abs(delta - np.median(delta, axis=0)), axis=0)) ** 2
    weights = 1 / (1 + variance)
    # Preserve mean weight 1: down-weighting cannot trivially shrink every distance.
    weights /= np.mean(weights)
    return {"center": center.tolist(), "scale": scale.tolist(), "scale_floors": floors.tolist(),
            "mic2_bias": bias.tolist(), "residual_variance": variance.tolist(),
            "weights_mean_one": weights.tolist(), "fit_pair_count": len(pairs)}


def coordinates(q, mic, calibration, mode="weighted"):
    q = np.asarray(q, dtype=float)
    if mic not in ("mic1", "mic2"):
        raise ValueError("Unknown microphone: no validated correction")
    if mode not in ("raw", "bias", "weighted"):
        raise ValueError(mode)
    if mode != "raw" and mic == "mic2":
        q = q - np.asarray(calibration["mic2_bias"])
    z = (q - calibration["center"]) / calibration["scale"]
    if mode == "weighted":
        z = z * np.sqrt(calibration["weights_mean_one"])
    return z / np.sqrt(6)
