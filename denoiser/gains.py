"""Spectral gain rules."""

import numpy as np


def wiener_decision_directed(z, noise, smoothing, gain_floor):
    """Wiener gain with a decision-directed a priori SNR (Ephraim & Malah, 1984).

    xi_t = a * |S_{t-1}|^2 / N + (1 - a) * max(gamma_t - 1, 0),  G_t = clip(xi/(1+xi), g_min, 1).
    The first frame uses max(gamma_0 - 1, 0).
    """
    power = np.abs(z) ** 2
    posterior_snr = power / noise[:, None]
    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros_like(noise)
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (smoothing * previous_clean_power / noise
                         + (1 - smoothing) * instantaneous)
        gain = np.clip(prior_snr / (1 + prior_snr), gain_floor, 1)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2
    return enhanced
