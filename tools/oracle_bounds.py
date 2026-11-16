"""Upper bounds of gain-based (STFT magnitude) enhancement, per noise type (analysis only).

These use the CLEAN signal and are therefore not denoisers; they show how far the current
framework (noisy STFT x real gain, noisy phase, 32 ms frames) could go at most:
  current       : the current denoiser (defaults)
  oracle_noise  : current pipeline, but the noise PSD is the true noise periodogram per frame
                  (|STFT(noisy - clean)|^2, 3-frame smoothing): perfect noise estimation
  oracle_irm    : ideal ratio mask sqrt(|S|^2 / (|S|^2 + |N|^2)) on the noisy STFT
  oracle_mag    : clean magnitude with noisy phase ("theoretical limit" of Virag 1999)

Usage: python tools/oracle_bounds.py --data-root DATA --split train --out OUT.csv [--subset CSV]
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

import denoiser.pipeline as pipeline  # noqa: E402
from denoiser import CONFIG  # noqa: E402
from denoiser.stft import istft, stft  # noqa: E402
from evaluate import pesq_wb  # noqa: E402


def one_file(args):
    row, data_root = args
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, _, zy = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    _, _, zn = stft(y - s, sr, n_fft, hop)
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    est = {"current": pipeline.denoise_signal(y, sr, CONFIG)}
    true_noise = uniform_filter1d(np.abs(zn) ** 2, 3, axis=1, mode="nearest") + 1e-12
    original = pipeline.estimate_noise_psd
    pipeline.estimate_noise_psd = lambda *a, **k: true_noise
    try:
        est["oracle_noise"] = pipeline.denoise_signal(y, sr, CONFIG)
    finally:
        pipeline.estimate_noise_psd = original
    ps, pn = np.abs(zs) ** 2, np.abs(zn) ** 2
    irm = np.sqrt(ps / np.maximum(ps + pn, 1e-30))
    est["oracle_irm"] = istft(irm * zy, len(y), sr, n_fft, hop)
    est["oracle_mag"] = istft(np.abs(zs) * np.exp(1j * np.angle(zy)), len(y), sr, n_fft, hop)
    out["noisy"] = pesq_wb(s, y, sr)[0]   # pesq_wb returns (score, error message)
    for k, v in est.items():
        out[k] = pesq_wb(s, v, sr)[0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--summarize-only", action="store_true",
                        help="rebuild the summaries from an existing --out CSV")
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    if args.subset:
        rows = rows[rows.segment_id.isin(set(pd.read_csv(args.subset, comment="#").segment_id))]
    if args.summarize_only:
        df = pd.read_csv(args.out)
    else:
        jobs = [(dict(r), str(args.data_root)) for _, r in rows.iterrows()]
        with ProcessPoolExecutor(args.workers) as pool:
            df = pd.DataFrame(pool.map(one_file, jobs, chunksize=4))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    cols = ["noisy", "current", "oracle_noise", "oracle_irm", "oracle_mag"]
    by = df.groupby("noise_category")[cols].mean()
    by.insert(0, "files", df.groupby("noise_category").size())
    by = by.sort_values("current")
    by.loc["ALL (mean of files)"] = [len(df)] + [df[c].mean() for c in cols]
    by.round(3).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    snr = df.groupby("target_snr_db")[cols].mean().round(3)
    snr.to_csv(args.out.with_name(args.out.stem + "_by_snr.csv"))
    pd.set_option("display.width", 200)
    print(by.round(3).to_string())
    print(snr.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
