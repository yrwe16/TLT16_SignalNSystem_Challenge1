"""Regenerate the reference outputs used by tests/test_regression.py.

Usage (from the repository root):
    python tests/make_reference.py

Run this only when an output change is intended (e.g. a stage is adopted and the
defaults change), and commit the new tests/data/*.npy together with that change.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, DATA, SAMPLE_RATE, reference_path, synthetic_signal  # noqa: E402

from denoiser.pipeline import denoise_signal  # noqa: E402


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    x = synthetic_signal()
    for name, cfg in CONFIGS.items():
        y = denoise_signal(x, SAMPLE_RATE, cfg)
        np.save(reference_path(name), y)
        print(f"{reference_path(name).name}: {len(y)} samples, rms {np.sqrt(np.mean(y ** 2)):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
