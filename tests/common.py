"""Shared fixtures for the regression tests (standard library + the runtime packages only)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from denoiser import CONFIG  # noqa: E402

SAMPLE_RATE = 16000
SECONDS = 3.0

# Stage-0 baseline Wiener (notebook algorithm), independent of the current defaults.
WIENER_BASELINE = replace(CONFIG, gain_rule="wiener", prior_snr="dd", wind_module=False,
                          phase_reconstruction=False, noise_tracker="fixed", omlsa=False,
                          tonal_module=False, rpca_module=False, transient_module=False)
CONFIGS = {"default": CONFIG, "wiener": WIENER_BASELINE}


def synthetic_signal(seed=0):
    """3 s at 16 kHz: a harmonic tone with a gliding f0 (on 0.5-2.5 s) plus white noise
    and a 60 Hz hum, fixed seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * SECONDS)) / SAMPLE_RATE
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    envelope = np.clip((t - 0.5) / 0.05, 0, 1) * np.clip((2.5 - t) / 0.05, 0, 1)
    speech = sum(np.cos(h * phase) / h for h in range(1, 25)) * envelope * 0.1
    noise = 0.02 * rng.standard_normal(t.size) + 0.02 * np.sin(2 * np.pi * 60 * t)
    return speech + noise


def reference_path(name):
    return DATA / f"synthetic_{name}.npy"
