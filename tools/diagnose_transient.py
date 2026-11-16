"""Transience index per file for the round-02 step-9 two-stage transient estimate.

Index = share of short frames (4 ms) in which at least transient_bin_share of the bins have
S / S_min > delta (noisy input only). Prints the distribution per noise category so that the
two file-gate values of the grid can be chosen. The noise category is used only for reporting.

Usage: python tools/diagnose_transient.py --data-root DATA --split train --out OUT.csv
       [--delta 3 4 6] [--workers N]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoiser import CONFIG  # noqa: E402
from denoiser.noise_estimation import two_stage_transient  # noqa: E402
from denoiser.pipeline import estimate_noise_psd  # noqa: E402
from denoiser.stft import stft  # noqa: E402

TARGET = ("clap", "click", "footsteps", "door_moving", "machine_gun")


def one_file(args):
    row, data_root, deltas = args
    x, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, times, z = stft(x, sr, n_fft, hop)
    stationary = estimate_noise_psd(z, times, len(x), sr, n_fft, hop, CONFIG)
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    for d in deltas:
        cfg = replace(CONFIG, transient_module=True, transient_mode="two_stage",
                      transient_delta=d, transient_file_index=1.0)
        _, index = two_stage_transient(x, sr, (n_fft // 2 + 1, 1), n_fft, hop, cfg, stationary)
        out[f"index_d{d:g}"] = index
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta", type=float, nargs="+", default=[3.0, 4.0, 6.0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    jobs = [(dict(r), str(args.data_root), args.delta) for _, r in rows.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_file, jobs, chunksize=8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    target = df.noise_category.isin(TARGET)
    for d in args.delta:
        col = f"index_d{d:g}"
        print(f"delta {d:g}: target files {target.sum()} index pct 10/50/90 "
              f"{np.percentile(df.loc[target, col], [10, 50, 90]).round(4)}; others pct 50/90/95/99 "
              f"{np.percentile(df.loc[~target, col], [50, 90, 95, 99]).round(4)}")
        for thr in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
            print(f"   gate {thr}: target on {np.mean(df.loc[target, col] >= thr):.2f}, "
                  f"others on {np.mean(df.loc[~target, col] >= thr):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
