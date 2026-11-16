"""Gain curves of the generalized gamma estimators against MMSE-LSA (for comparison with
Fig. 5 of Erkelens et al., IEEE TASLP 2007: gain versus a posteriori SNR at xi = -5, +5, +15 dB).

Usage: python tools/plot_gg_gain.py --out FIGURE.png   (requires matplotlib)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoiser.gains import lsa_gain  # noqa: E402
from denoiser.gg_gain import gain_gamma1, gain_gamma1_high, gain_gamma1_low, gain_gamma2  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    zeta_db = np.linspace(-10, 14, 200)
    zeta = 10 ** (zeta_db / 10)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, constrained_layout=True)
    for col, xi_db in enumerate((-5, 5, 15)):
        xi = np.full_like(zeta, 10 ** (xi_db / 10))
        ax = axes[0, col]
        ax.plot(zeta_db, gain_gamma1(xi, zeta, 0.6), "k-", label="gamma=1, nu=0.6 combined")
        ax.plot(zeta_db, gain_gamma1_low(xi, zeta, 0.6), "b:", label="eq. (16), K=5")
        ax.plot(zeta_db, gain_gamma1_high(xi, zeta, 0.6), "r--", label="eq. (18)")
        ax.plot(zeta_db, lsa_gain(xi, zeta), color="0.6", label="MMSE-LSA")
        ax.set_title(f"xi = {xi_db:+d} dB", fontsize=9)
        ax = axes[1, col]
        for nu in (0.1, 0.2, 0.3, 1.0):
            ax.plot(zeta_db, gain_gamma2(xi, zeta, nu), label=f"gamma=2, nu={nu}"
                    + (" (STSA)" if nu == 1.0 else ""))
        ax.plot(zeta_db, lsa_gain(xi, zeta), color="0.6", label="MMSE-LSA")
        ax.set_xlabel("a posteriori SNR (dB)")
    for ax in axes.ravel():
        ax.set_ylim(0, 1.3)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("gain")
    axes[1, 0].set_ylabel("gain")
    axes[0, 2].legend(fontsize=7)
    axes[1, 2].legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=80)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
