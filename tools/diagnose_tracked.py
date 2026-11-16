"""Diagnostics for the round-02 step-11 protected noise tracker (evaluation only).

Per file, the stationary estimate (spp_iter) and the protected tracker are compared with the
true noise periodogram (|STFT(noisy - clean)|^2, 3-frame smoothing) in four bands, frame by
frame. Reported per noise type: mean absolute dB error and mean dB bias, separately in speech
frames (clean frame power within 25 dB of the loudest) and in pause frames (more than 40 dB
below). Speech leakage: the tracker run on the CLEAN signal alone (with the fixed estimate of
the noisy file as its floor) - its mean rise above that floor in speech frames, in dB.

Usage: python tools/diagnose_tracked.py --data-root DATA --split train --out OUT.csv
       [--set KEY=VALUE ...] [--subset CSV]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.pipeline import estimate_noise_psd, protected_noise  # noqa: E402
from denoiser.stft import stft  # noqa: E402
from evaluate import parse_overrides  # noqa: E402

BANDS = (0, 325, 1000, 4000, 8001)


def band_sum(p, sr, n_fft):
    f = np.arange(p.shape[0]) * sr / n_fft
    return np.stack([p[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in zip(BANDS[:-1], BANDS[1:])])


def one_file(args):
    row, data_root, overrides = args
    cfg = parse_overrides(overrides + ["noise_estimator=tracked_protected"])
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = cfg.stft_sizes(sr)
    _, times, z = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    _, _, zn = stft(y - s, sr, n_fft, hop)
    fixed = estimate_noise_psd(z, times, len(y), sr, n_fft, hop, cfg)
    tracked = protected_noise(y, sr, z, fixed, times, n_fft, hop, cfg)
    true = uniform_filter1d(np.abs(zn) ** 2, 3, axis=1, mode="nearest")
    cp = (np.abs(zs) ** 2).sum(axis=0)
    rel = 10 * np.log10(np.maximum(cp, 1e-30) / max(cp.max(), 1e-30))
    speech, pause = rel >= -25, rel < -40
    tb = band_sum(true, sr, n_fft) + 1e-20
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    for name, est in (("fixed", np.broadcast_to(fixed[:, None], z.shape)), ("tracked", tracked)):
        err = 10 * np.log10((band_sum(est, sr, n_fft) + 1e-20) / tb)
        for lab, m in (("speech", speech), ("pause", pause)):
            out[f"{name}_abs_{lab}"] = float(np.abs(err[:, m]).mean()) if m.any() else np.nan
            out[f"{name}_bias_{lab}"] = float(err[:, m].mean()) if m.any() else np.nan
    leak = protected_noise(s, sr, zs, fixed, times, n_fft, hop, cfg)
    rise = 10 * np.log10(band_sum(leak, sr, n_fft).sum(axis=0) / band_sum(np.broadcast_to(
        fixed[:, None], z.shape), sr, n_fft).sum(axis=0))
    out["leak_db_speech"] = float(rise[speech].mean()) if speech.any() else np.nan
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    if args.subset:
        rows = rows[rows.segment_id.isin(set(pd.read_csv(args.subset, comment="#").segment_id))]
    jobs = [(dict(r), str(args.data_root), args.set) for _, r in rows.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_file, jobs, chunksize=4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = [c for c in df.columns if c.startswith(("fixed_", "tracked_", "leak_"))]
    by = df.groupby("noise_category")[cols].mean()
    by.insert(0, "files", df.groupby("noise_category").size())
    by.loc["ALL"] = [len(df)] + [df[c].mean() for c in cols]
    by.round(2).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    pd.set_option("display.width", 250)
    show = ["files", "fixed_abs_speech", "tracked_abs_speech", "fixed_bias_speech", "tracked_bias_speech",
            "fixed_abs_pause", "tracked_abs_pause", "leak_db_speech"]
    print(by[show].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
