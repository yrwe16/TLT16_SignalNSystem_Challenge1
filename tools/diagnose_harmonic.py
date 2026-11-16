"""Diagnostics for the round-02 step-12 harmonic tunnel noise estimate (evaluation only).

Per file (clean used only here):
- in voiced frames (YIN on the first pass of the noisy file): mean dB error and bias of the
  current noise estimate and of the tunnel estimate against the true noise periodogram
  (|STFT(noisy - clean)|^2, 3-frame smoothing), summed in four bands;
- leakage: the tunnel estimate computed on the CLEAN signal alone (its own first pass and f0),
  relative to the true noise level of the noisy file in the same frames (dB): how much speech
  energy the tunnels would add to the noise estimate if no noise were present;
- the share of voiced frames that YIN finds in the noise-only signal (narrowband noise taken
  for speech).

Usage: python tools/diagnose_harmonic.py --data-root DATA --split train --out OUT.csv
       [--set KEY=VALUE ...] [--subset CSV]
"""

import argparse
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
from scipy.ndimage import uniform_filter1d  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.noise_estimation import harmonic_tunnel_noise  # noqa: E402
from denoiser.phase import yin_f0  # noqa: E402
from denoiser.pipeline import estimate_noise_psd, first_pass_f0, protected_noise  # noqa: E402
from denoiser.stft import stft  # noqa: E402
from evaluate import parse_overrides  # noqa: E402

BANDS = (0, 325, 1000, 4000, 8001)


def band_sum(p, sr, n_fft):
    f = np.arange(p.shape[0]) * sr / n_fft
    return np.stack([p[(f >= lo) & (f < hi)].sum(axis=0) for lo, hi in zip(BANDS[:-1], BANDS[1:])])


def one_file(args):
    row, data_root, overrides = args
    cfg = parse_overrides(overrides)
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = cfg.stft_sizes(sr)
    _, times, z = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    _, _, zn = stft(y - s, sr, n_fft, hop)
    f0, voiced = first_pass_f0(y, sr, times, cfg)
    base = estimate_noise_psd(z, times, len(y), sr, n_fft, hop, cfg)
    if cfg.noise_estimator == "tracked_protected":
        base = protected_noise(y, sr, z, base, times, n_fft, hop, cfg, (f0, voiced))
    base = np.broadcast_to(base[:, None] if base.ndim == 1 else base, z.shape)
    _, tunnel = harmonic_tunnel_noise(np.abs(z) ** 2, base, f0, voiced, sr, n_fft, cfg)
    true = uniform_filter1d(np.abs(zn) ** 2, 3, axis=1, mode="nearest")
    tb = band_sum(true, sr, n_fft) + 1e-20
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"], "voiced_share": float(np.mean(voiced))}
    v = ~np.isnan(tunnel).any(axis=0)
    for name, est in (("current", base), ("tunnel", np.nan_to_num(tunnel))):
        err = 10 * np.log10((band_sum(est, sr, n_fft) + 1e-20) / tb)
        out[f"{name}_abs"] = float(np.abs(err[:, v]).mean()) if v.any() else np.nan
        out[f"{name}_bias"] = float(err[:, v].mean()) if v.any() else np.nan
    # leakage: tunnels on clean speech alone
    f0c, vc = first_pass_f0(s, sr, times, replace(cfg, noise_estimator="spp_iter"))
    _, tun_c = harmonic_tunnel_noise(np.abs(zs) ** 2, np.full(z.shape[0], 1e-20), f0c, vc, sr,
                                     n_fft, cfg)
    vcm = ~np.isnan(tun_c).any(axis=0)
    if vcm.any():
        leak = 10 * np.log10((band_sum(np.nan_to_num(tun_c), sr, n_fft).sum(axis=0)[vcm] + 1e-20)
                             / (tb.sum(axis=0)[vcm]))
        out["leak_db_vs_true_noise"] = float(np.mean(leak))
    else:
        out["leak_db_vs_true_noise"] = np.nan
    centers = np.rint(times * sr).astype(int)
    out["yin_voiced_noise_only"] = float(np.mean(yin_f0(y - s, sr, centers, cfg)[1]))
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
    cols = ["voiced_share", "current_abs", "tunnel_abs", "current_bias", "tunnel_bias",
            "leak_db_vs_true_noise", "yin_voiced_noise_only"]
    by = df.groupby("noise_category")[cols].mean()
    by.insert(0, "files", df.groupby("noise_category").size())
    by.loc["ALL"] = [len(df)] + [df[c].mean() for c in cols]
    by.round(2).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    snr = df.groupby("target_snr_db")[cols].mean().round(2)
    pd.set_option("display.width", 250)
    print(by.round(2).to_string())
    print(snr.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
