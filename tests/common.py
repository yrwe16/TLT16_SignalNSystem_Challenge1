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
                          band_floor=False, noise_estimator="lowenergy", noise_beta_low=1.0,
                          noise_beta=1.0, noise_beta_high=1.0,
                          phase_reconstruction=False, noise_tracker="fixed", omlsa=False,
                          tonal_module=False, rpca_module=False, transient_module=False,
                          postfilter=False, masking=False, mdkf=False)
# r03-base-s2 (before step R3-E8): no modulation-domain Kalman a priori SNR.
R03_S2 = replace(CONFIG, mdkf=False)
# r03-base-s1 (before step R3-2): low-band cut-off 325 Hz.
R03_S1 = replace(R03_S2, band_cutoff_hz=325.0)
# r02-base-s6 = round-02 submission (before step R3-1): DD alpha 0.95, high-band factor 1.25.
R02_S6 = replace(R03_S1, lsa_smoothing=0.95, noise_beta_high=1.25)
# r02-base-s5 (before step 15): one -15 dB LSA floor above 325 Hz, no noise factor.
R02_S5 = replace(R02_S6, band_mid_floor_db=-15.0, band_midhigh_floor_db=-15.0,
                 band_high_floor_db=-15.0, noise_beta_high=1.0)
# r02-base-s4 (before step 14): tonal tracks without the envelope test, observed power.
R02_S4 = replace(R02_S5, tonal_level="observed", tonal_env_max_db=1000.0)
# Round-02 base before step 7 (r02-base-s2): tonal module off, stationary noise estimate.
R02_S2 = replace(R02_S4, tonal_module=False, noise_estimator="spp_iter")
# r02-base-s3 (before step 11): stationary spp_iter noise estimate.
R02_S3 = replace(R02_S4, noise_estimator="spp_iter")
CONFIGS = {"default": CONFIG, "wiener": WIENER_BASELINE, "r02_s2": R02_S2, "r02_s3": R02_S3,
           "r02_s4": R02_S4, "r02_s5": R02_S5, "r02_s6": R02_S6,
           "r03_s1": R03_S1, "r03_s2": R03_S2}


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
