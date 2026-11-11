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


def _hann_kernel(half_width):
    """Normalised Hann window of length 2w+1 without zero end points (MATLAB hanning)."""
    n = 2 * half_width + 1
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1))
    return w / w.sum()


class _CohenAbsence:
    """A priori speech absence probability q (Cohen & Berdugo, 2001, Sec. 3).

    xi is smoothed in time (beta = 0.7), then in frequency with local (w = 1) and global
    (w = 15) Hann windows; each is mapped to P in [P_min, 1] linearly in dB between -10 and
    -5 dB; a frame term P_frame follows the band-mean xi (50 Hz - 10 kHz) and its peak
    xi_m in [0, 10] dB. q = 1 - P_local * P_global * P_frame, capped at 0.998.
    Constants: BSD-3 reference chenwj1989/python-speech-enhancement (re-implemented);
    not checked against the paper's full text.
    """

    def __init__(self, n_bins, sample_rate, n_fft):
        freqs = np.arange(n_bins) * sample_rate / n_fft
        self.band = (freqs >= 50) & (freqs <= 10000)
        self.local_band = (freqs >= 500) & (freqs <= 3500)
        self.k_local, self.k_global = _hann_kernel(1), _hann_kernel(15)
        self.xi = np.zeros(n_bins)
        self.first = True
        self.previous_frame_db = None
        self.peak_db = 0.0
        self.p_min = 0.005

    def _map(self, value_db, low=-10.0, high=-5.0):
        p = self.p_min + (value_db - low) / (high - low) * (1 - self.p_min)
        return np.clip(p, self.p_min, 1.0)

    def update(self, prior_snr):
        self.xi = prior_snr.copy() if self.first else 0.7 * self.xi + 0.3 * prior_snr
        self.first = False
        to_db = lambda v: 10 * np.log10(np.maximum(v, 1e-10))
        p_local = self._map(to_db(np.convolve(self.xi, self.k_local, mode="same")))
        p_global = self._map(to_db(np.convolve(self.xi, self.k_global, mode="same")))
        if p_local[self.local_band].mean() < 0.25:
            p_local[self.local_band] = self.p_min
        frame_db = float(to_db(self.xi[self.band].mean()))
        rising = self.previous_frame_db is None or frame_db >= self.previous_frame_db
        if frame_db <= -10:
            p_frame = self.p_min
        elif rising:
            self.peak_db = min(max(frame_db, 0.0), 10.0)
            p_frame = 1.0
        else:
            p_frame = float(self._map(frame_db, self.peak_db - 10, self.peak_db - 5))
        self.previous_frame_db = frame_db
        return np.minimum(1 - p_local * p_global * p_frame, 0.998)


def omlsa_decision_directed(z, noise, sample_rate, n_fft, cfg):
    """Optimally-modified LSA (Cohen & Berdugo, Signal Processing 81(11), 2001).

    xi_t = max(a * G_H1(t-1)^2 * gamma(t-1) + (1-a) * max(gamma_t - 1, 0), G_min^2)
    G_H1 = MMSE-LSA gain;  p = speech presence probability;  G = G_H1^p * G_min^(1-p).
    With omlsa_spp="cohen", p = 1/(1 + q/(1-q)*(1+xi)*exp(-v)) and p = 0 where q >= 0.9
    (as in the BSD-3 reference); with "posterior", p is the stage-2 fixed-xi_H1 SPP.
    """
    power = np.abs(z) ** 2
    noise_tf = _per_frame(noise, z.shape)
    posterior_snr = power / noise_tf
    g_min = cfg.omlsa_gain_floor
    xi_min = g_min ** 2
    a = cfg.lsa_smoothing
    absence = _CohenAbsence(z.shape[0], sample_rate, n_fft) if cfg.omlsa_spp == "cohen" else None
    xi_h1 = 10 ** (cfg.spp_snr_db / 10)
    pfac = (1 / cfg.spp_prior - 1) * (1 + xi_h1)
    enhanced = np.empty_like(z)
    previous = None  # G_H1(t-1)^2 * gamma(t-1)
    for frame in range(z.shape[1]):
        gamma = posterior_snr[:, frame]
        instantaneous = np.maximum(gamma - 1.0, 0.0)
        prior_snr = instantaneous if previous is None else a * previous + (1 - a) * instantaneous
        if cfg.omlsa_dd == "h1":
            prior_snr = np.maximum(prior_snr, xi_min)
        gain_h1 = lsa_gain(prior_snr, gamma)
        if absence is not None:
            q = absence.update(prior_snr)
            v = prior_snr / (1 + prior_snr) * gamma
            with np.errstate(over="ignore", divide="ignore"):
                p = 1 / (1 + q / (1 - q) * (1 + prior_snr) * np.exp(-v))
            p = np.where(q < 0.9, p, 0.0)
        else:
            p = 1 / (1 + pfac * np.exp(-gamma * xi_h1 / (1 + xi_h1)))
        gain = gain_h1 ** p * g_min ** (1 - p)
        enhanced[:, frame] = gain * z[:, frame]
        if cfg.omlsa_dd == "h1":
            previous = gain_h1 ** 2 * gamma
        else:  # stage-1 DD: previous output power, no xi floor
            previous = np.abs(enhanced[:, frame]) ** 2 / noise_tf[:, frame]
    return enhanced
