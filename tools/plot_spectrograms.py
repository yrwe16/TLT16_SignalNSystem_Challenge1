"""Spectrograms of Noisy / previous / new / Clean for the fixed listening set.

Usage:
    python tools/plot_spectrograms.py --data-root PATH --split validation \
        --run "Wiener (round-01)=DIR_A" --run "MMSE-LSA=DIR_B" --out FIG_DIR

Each DIR holds denoised WAV files named like the noisy files (evaluate.py --keep-enhanced).
All panels of a figure share the same dB colour range (max of the clean spectrogram - 80 dB).
Requires matplotlib (pip install matplotlib); not needed to run denoise.py.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import stft  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def spec_db(x, sr):
    f, t, z = stft(x, sr, window="hann", nperseg=512, noverlap=384)
    return f, t, 20 * np.log10(np.abs(z) + 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--list", type=Path, default=ROOT / "results" / "listening_set.txt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--range-db", type=float, default=80.0)
    args = parser.parse_args()

    runs = [r.rsplit("=", 1) for r in args.run]  # the label may itself contain "="
    manifest = pd.read_csv(args.data_root / args.split / "samples.csv").set_index("segment_id")
    chosen = pd.read_csv(args.list, comment="#")
    args.out.mkdir(parents=True, exist_ok=True)
    for seg in chosen.segment_id:
        row = manifest.loc[seg]
        clean, sr = sf.read(str(args.data_root / row.clean_path))
        noisy, _ = sf.read(str(args.data_root / row.noisy_path))
        panels = [("Noisy", noisy)]
        for label, folder in runs:
            x, _ = sf.read(str(Path(folder) / Path(row.noisy_path).name))
            panels.append((label, x))
        panels.append(("Clean", clean))
        _, _, ref = spec_db(clean, sr)
        vmax = ref.max()
        fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.0 * len(panels)),
                                 sharex=True, constrained_layout=True)
        for ax, (label, x) in zip(axes, panels):
            f, t, s = spec_db(x, sr)
            mesh = ax.pcolormesh(t, f / 1000, s, shading="auto", cmap="magma",
                                 vmin=vmax - args.range_db, vmax=vmax)
            ax.set_ylabel("kHz")
            ax.set_title(label, fontsize=9, loc="left")
        axes[-1].set_xlabel("time (s)")
        fig.colorbar(mesh, ax=axes, label="dB", shrink=0.6)
        fig.suptitle(f"{seg}  ({row.noise_category}, {row.target_snr_db:g} dB)", fontsize=10)
        fig.savefig(args.out / f"{seg}.png", dpi=80)
        plt.close(fig)
    print(f"Saved {len(chosen)} figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
