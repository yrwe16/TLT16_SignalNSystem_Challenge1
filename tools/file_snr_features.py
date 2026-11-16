"""Per-file features from the noisy signal and the current noise estimate (no clean signal).

Usage:
    python tools/file_snr_features.py --data-root PATH --split train --out FILE.csv [--set K=V]

For every file:
  - snr_lowband_db = 10 log10( sum_{t, f < cutoff} max(|Y|^2 - N, 0) / sum_{t, f < cutoff} N )
  - snr_file_db    = the same over all bins
  - lowband_energy_ratio_db = 10 log10( sum_{f < cutoff} |Y|^2 / sum_all |Y|^2 )
with N the noise PSD estimate of the configured estimator (denoiser.pipeline.estimate_noise_psd),
summed over the complete STFT frames. cutoff = band_cutoff_hz (325 Hz by default).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.noise_estimation import complete_frames, file_snr_features  # noqa: E402
from denoiser.pipeline import estimate_noise_psd  # noqa: E402
from denoiser.stft import stft  # noqa: E402
from evaluate import parse_overrides  # noqa: E402


def features(job):
    root, row, cfg = job
    x, sr = sf.read(str(root / row["noisy_path"]), dtype="float64")
    n_fft, hop = cfg.stft_sizes(sr)
    _, times, z = stft(x, sr, n_fft, hop)
    noise = estimate_noise_psd(z, times, len(x), sr, n_fft, hop, cfg)
    full = complete_frames(times, len(x), sr, n_fft, z.shape[1])
    out = file_snr_features(z[:, full], noise if noise.ndim == 1 else noise[:, full],
                            sr, n_fft, cfg.band_cutoff_hz)
    return {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
            "target_snr_db": row["target_snr_db"], **out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = parse_overrides(args.set)
    root = args.data_root.resolve()
    manifest = pd.read_csv(root / args.split / "samples.csv")
    jobs = [(root, row._asdict(), cfg) for row in manifest.itertuples(index=False)]
    with ProcessPoolExecutor() as pool:
        frame = pd.DataFrame(pool.map(features, jobs, chunksize=16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(frame.describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
