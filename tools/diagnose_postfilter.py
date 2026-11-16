"""Diagnostics for the round-02 step-6 musical-noise postfilter (evaluation only).

For each file the current default pipeline is run with postfilter=True, psi=0 (identity), and
the gain entering the postfilter is captured. Frames are labelled with the CLEAN signal (used
here for evaluation only, never by the denoiser): a frame is a speech frame when its clean
power is within --speech-db (default 25 dB, our choice) of the file's loudest clean frame.
For each (zeta_thr, psi) it reports how many speech / non-speech frames are smoothed and the
largest gain change in speech frames (dB, over all bins).

Usage: python tools/diagnose_postfilter.py --data-root DATA --split train --out OUT.json
       [--thr 0.4 0.2] [--psi 1 3 7] [--workers 8]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import denoiser.pipeline as pipeline  # noqa: E402
from denoiser import CONFIG  # noqa: E402
from denoiser.gains import musical_noise_postfilter  # noqa: E402
from denoiser.stft import stft  # noqa: E402


def one_file(args):
    row, data_root, thrs, psis, speech_db = args
    noisy, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    clean, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    captured = {}

    def capture(gain, power, zeta_thr, psi):
        captured["gain"], captured["power"] = gain.copy(), power.copy()
        return musical_noise_postfilter(gain, power, zeta_thr, psi)

    original = pipeline.musical_noise_postfilter
    pipeline.musical_noise_postfilter = capture
    try:
        pipeline.denoise_signal(noisy, sr, replace(CONFIG, postfilter=True, postfilter_psi=0))
    finally:
        pipeline.musical_noise_postfilter = original
    gain, power = captured["gain"], captured["power"]
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, _, c = stft(clean, sr, n_fft, hop)
    cp = np.sum(np.abs(c) ** 2, axis=0)
    speech = 10 * np.log10(np.maximum(cp, 1e-30) / max(cp.max(), 1e-30)) >= -speech_db
    _, _, zeta = musical_noise_postfilter(gain, power, 1.0, 0)
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"], "frames": int(len(zeta)),
           "speech_frames": int(speech.sum()),
           "zeta_speech": zeta[speech].tolist(), "zeta_pause": zeta[~speech].tolist()}
    for thr in thrs:
        for psi in psis:
            new, n, _ = musical_noise_postfilter(gain, power, thr, psi)
            key = f"thr{thr}_psi{psi}"
            # gain change in dB with both gains limited to 1 (the clip of LSA gains > 1 in
            # near-empty bins is reported separately as the frame output energy change)
            delta = np.abs(20 * np.log10(np.maximum(np.minimum(new, 1), 1e-12)
                                         / np.maximum(np.minimum(gain, 1), 1e-12)))
            e_old = np.sum(gain ** 2 * power, axis=0)
            e_new = np.sum(new ** 2 * power, axis=0)
            de = np.abs(10 * np.log10(np.maximum(e_new, 1e-30) / np.maximum(e_old, 1e-30)))
            sp = speech & (n > 1)
            out[key] = {"smoothed_speech": int(np.sum((n > 1) & speech)),
                        "smoothed_pause": int(np.sum((n > 1) & ~speech)),
                        "max_delta_db_speech": float(delta[:, speech].max()) if speech.any() else 0.0,
                        "mean_delta_db_speech": float(delta[:, speech].mean()) if speech.any() else 0.0,
                        "max_energy_db_speech": float(de[speech].max()) if speech.any() else 0.0,
                        "energy_db_smoothed_speech": de[sp].tolist(),
                        "energy_db_smoothed_pause": de[(n > 1) & ~speech].tolist()}
    out["gain_above_one"] = float(np.mean(gain > 1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thr", type=float, nargs="+", default=[0.4, 0.2])
    parser.add_argument("--psi", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument("--speech-db", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    rows = [dict(r) for _, r in rows.iterrows()]
    jobs = [(r, str(args.data_root), args.thr, args.psi, args.speech_db) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(one_file, jobs, chunksize=8))

    zs = np.concatenate([r["zeta_speech"] for r in results])
    zp = np.concatenate([r["zeta_pause"] for r in results])
    q = [5, 25, 50, 75, 95]
    summary = {"split": args.split, "files": len(results), "speech_db": args.speech_db,
               "zeta_percentiles": q,
               "zeta_speech": np.percentile(zs, q).round(4).tolist(),
               "zeta_pause": np.percentile(zp, q).round(4).tolist(),
               "speech_frames": int(len(zs)), "pause_frames": int(len(zp)),
               "fraction_below": {str(t): {"speech": float(np.mean(zs < t)),
                                           "pause": float(np.mean(zp < t))}
                                  for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)},
               "settings": {}}
    for thr in args.thr:
        for psi in args.psi:
            key = f"thr{thr}_psi{psi}"
            s = [r[key] for r in results]
            summary["settings"][key] = {
                "max_window": 2 * psi + 1,
                "smoothed_speech_frac": sum(x["smoothed_speech"] for x in s) / len(zs),
                "smoothed_pause_frac": sum(x["smoothed_pause"] for x in s) / len(zp),
                "max_delta_db_speech": max(x["max_delta_db_speech"] for x in s),
                "median_file_max_delta_db_speech": float(np.median([x["max_delta_db_speech"] for x in s])),
                "mean_delta_db_speech": float(np.mean([x["mean_delta_db_speech"] for x in s])),
                "max_energy_db_speech": max(x["max_energy_db_speech"] for x in s),
                "energy_db_smoothed_speech_p50_p95": np.percentile(
                    np.concatenate([x["energy_db_smoothed_speech"] for x in s] + [[0.0]]),
                    [50, 95]).round(3).tolist(),
                "energy_db_smoothed_pause_p50_p95": np.percentile(
                    np.concatenate([x["energy_db_smoothed_pause"] for x in s] + [[0.0]]),
                    [50, 95]).round(3).tolist()}
    summary["gain_above_one_fraction"] = float(np.mean([r["gain_above_one"] for r in results]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    per_file = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("zeta")
                              and not isinstance(v, dict)}
                             | {f"{key}_max_delta_db_speech": r[key]["max_delta_db_speech"]
                                for key in summary["settings"]} for r in results])
    per_file.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1))
    print(json.dumps(summary["settings"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
