"""Noise power estimation from the noisy input only (no clean reference)."""

import numpy as np
from scipy.ndimage import uniform_filter1d


def lowest_energy_frames(z, times, length, sample_rate, n_fft, noise_fraction):
    """Mean power of the lowest-energy complete frames, smoothed over 3 bins.

    One estimate per file, as in the baseline notebook. Boundary frames padded with
    zeros are excluded so that artificially quiet frames are not selected.
    """
    power = np.abs(z) ** 2
    centers = np.rint(times * sample_rate).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    if not len(full):
        full = np.arange(power.shape[1])
    energy = power[:, full].mean(axis=0)
    count = max(1, int(np.ceil(noise_fraction * len(full))))
    selected = full[np.argsort(energy, kind="stable")[:count]]
    estimate = power[:, selected].mean(axis=1)
    return np.maximum(uniform_filter1d(estimate, size=3, mode="nearest"), 1e-18)
