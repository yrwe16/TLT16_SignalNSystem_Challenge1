"""Spectral gain rules."""

import numpy as np
from scipy.special import exp1


def _per_frame(noise, shape):
    """Noise PSD as a (bins, frames) view: one vector per file or one vector per frame."""
    noise = np.asarray(noise, dtype=np.float64)
    return np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, shape)


def wiener_decision_directed(z, noise, smoothing, gain_floor):
    """Wiener gain with a decision-directed a priori SNR (Ephraim & Malah, 1984).

    xi_t = a * |S_{t-1}|^2 / N + (1 - a) * max(gamma_t - 1, 0),  G_t = clip(xi/(1+xi), g_min, 1).
    The first frame uses max(gamma_0 - 1, 0).
    """
    power = np.abs(z) ** 2
    noise_tf = _per_frame(noise, z.shape)
    posterior_snr = power / noise_tf
    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros(z.shape[0])
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (smoothing * previous_clean_power / noise_tf[:, frame]
                         + (1 - smoothing) * instantaneous)
        gain = np.clip(prior_snr / (1 + prior_snr), gain_floor, 1)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2
    return enhanced


def lsa_gain(prior_snr, posterior_snr):
    """MMSE log-spectral amplitude gain (Ephraim & Malah, IEEE TASSP 33(2), 1985, eq. 20).

    G = xi/(1+xi) * exp(0.5 * E1(v)),  v = xi*gamma/(1+xi).
    v is floored at 1e-10 so that E1 stays finite; as v -> 0 the gain tends to
    sqrt(xi/gamma) -> 0, so the floor only matters in bins that are floored anyway.
    """
    ratio = prior_snr / (1 + prior_snr)
    v = np.maximum(ratio * posterior_snr, 1e-10)
    return ratio * np.exp(0.5 * exp1(v))


def lsa_decision_directed(z, noise, smoothing, gain_floor):
    """MMSE-LSA gain with the same decision-directed a priori SNR as the Wiener rule.

    xi_t = a * |S_{t-1}|^2 / N + (1 - a) * max(gamma_t - 1, 0);  G_t = max(G_LSA, g_min).
    The first frame uses max(gamma_0 - 1, 0). No upper clip, as in the paper.
    """
    power = np.abs(z) ** 2
    noise_tf = _per_frame(noise, z.shape)
    posterior_snr = power / noise_tf
    enhanced = np.empty_like(z)
    previous_clean_power = np.zeros(z.shape[0])
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (smoothing * previous_clean_power / noise_tf[:, frame]
                         + (1 - smoothing) * instantaneous)
        gain = np.maximum(lsa_gain(prior_snr, posterior_snr[:, frame]), gain_floor)
        enhanced[:, frame] = gain * z[:, frame]
        previous_clean_power = np.abs(enhanced[:, frame]) ** 2
    return enhanced
