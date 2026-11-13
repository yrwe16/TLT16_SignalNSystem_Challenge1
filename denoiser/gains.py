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


EULER_GAMMA = 0.5772156649015329


def _per_frame_factor(alpha, hop_seconds, reference_hop):
    """Convert a per-frame smoothing factor given at reference_hop to the actual hop
    (same time constant); reference_hop=None keeps the value per frame."""
    return alpha if reference_hop is None else alpha ** (hop_seconds / reference_hop)


def cepstral_prior_snr(posterior_snr, noise_tf, sample_rate, hop_seconds, cfg):
    """A priori SNR by selective cepstro-temporal smoothing (Breithaupt, Gerkmann & Martin,
    ICASSP 2008, pp. 4897-4900, eqs. (3)-(13), Table 1).

    lambda_ml = N * max(gamma - 1, xi_ml_min)                                   (3)
    c_ml = IDFT(log lambda_ml); c = alpha(q,l) c(l-1) + (1 - alpha(q,l)) c_ml   (4, 5)
    alpha = alpha_pitch on Q_pitch, else abar; abar = beta abar(l-1) + (1-beta) a_const (6, 7)
    Q_pitch = q_pitch +- dq if the Hamming-smoothed cepstral peak in fs/300..fs/70 is
    >= thr and c_ml(1) > 0                                                     (8)-(10)
    lambda_s = exp(kappa + DFT(c));  xi = max(lambda_s / N, xi_min)            (12, 13)
    The cepstrum is taken of log(lambda_ml) including the noise power, so that the pitch
    peak and the spectral-tilt test c_ml(1) > 0 see the speech power estimate (eq. 10).
    """
    bins, frames = posterior_snr.shape
    m = 2 * (bins - 1)
    ref = None if cfg.ceps_per_frame_constants else 0.016  # paper: 16 kHz, M=512, shift M/2
    scale = lambda a: _per_frame_factor(a, hop_seconds, ref)
    a_const = np.full(m // 2 + 1, scale(0.97))
    a_const[:3] = scale(0.5)
    a_const[3:20] = scale(0.7)
    a_pitch, beta = scale(0.2), scale(0.96)
    q_low = int(np.floor(sample_rate / 300.0))
    q_high = min(int(np.floor(sample_rate / 70.0)), m // 2)
    window = np.hamming(8)
    if cfg.ceps_window_normalized:
        window = window / window.sum()
    xi_ml_min = 10 ** (-27 / 10)
    xi_min = 10 ** (cfg.ceps_xi_min_db / 10)
    kappa = cfg.ceps_kappa * EULER_GAMMA
    log_ml = np.log(np.maximum(posterior_snr - 1.0, xi_ml_min) * noise_tf)
    c_ml_all = np.fft.irfft(log_ml, n=m, axis=0)[: m // 2 + 1]  # (q, frames), symmetric
    smoothed = c_ml_all[:, 0].copy()
    abar = a_const.copy()
    xi = np.empty_like(posterior_snr)
    voiced = 0
    for t in range(frames):
        c_ml = c_ml_all[:, t]
        abar = beta * abar + (1 - beta) * a_const
        alpha = abar.copy()
        lowpass = np.convolve(c_ml, window, mode="same")
        q_pitch = q_low + int(np.argmax(lowpass[q_low:q_high + 1]))
        if lowpass[q_pitch] >= cfg.ceps_threshold and c_ml[1] > 0:
            span = slice(max(q_pitch - 2, 0), q_pitch + 3)
            alpha[span] = a_pitch
            abar[span] = a_pitch  # abar(q,l) feeds the next frame's recursion (7)
            voiced += 1
        if t:
            smoothed = alpha * smoothed + (1 - alpha) * c_ml
        full = np.concatenate([smoothed, smoothed[-2:0:-1]])
        log_s = np.fft.rfft(full).real
        xi[:, t] = np.maximum(np.exp(kappa + log_s) / noise_tf[:, t], xi_min)
    cepstral_prior_snr.voiced_share = voiced / max(frames, 1)
    return xi


def ml_2d_prior_snr(posterior_snr, cfg):
    """Offline variant: centred 2-D (frequency x time) moving average of max(gamma-1, 0)."""
    from scipy.ndimage import uniform_filter
    ml = np.maximum(posterior_snr - 1.0, 0.0)
    xi = uniform_filter(ml, size=(cfg.ml2d_bins, cfg.ml2d_frames), mode="nearest")
    return np.maximum(xi, 10 ** (cfg.ceps_xi_min_db / 10))


def _dd_prior_snr(posterior_snr, noise_tf, z, smoothing, gain_floor):
    """The stage-1 DD-LSA loop, returning xi for every frame."""
    xi = np.empty_like(posterior_snr)
    previous_clean_power = np.zeros(z.shape[0])
    for frame in range(z.shape[1]):
        instantaneous = np.maximum(posterior_snr[:, frame] - 1.0, 0.0)
        if frame == 0:
            prior_snr = instantaneous
        else:
            prior_snr = (smoothing * previous_clean_power / noise_tf[:, frame]
                         + (1 - smoothing) * instantaneous)
        xi[:, frame] = prior_snr
        gain = np.maximum(lsa_gain(prior_snr, posterior_snr[:, frame]), gain_floor)
        previous_clean_power = np.abs(gain * z[:, frame]) ** 2
    return xi


def lsa_with_prior(z, noise, sample_rate, hop_seconds, cfg, gain_floor=None):
    """MMSE-LSA gain (stage 1 floor) with an alternative a priori SNR estimator.

    gain_floor: scalar or one value per frequency bin (round-02 band_floor); default
    cfg.lsa_gain_floor."""
    floor = cfg.lsa_gain_floor if gain_floor is None else gain_floor
    power = np.abs(z) ** 2
    noise_tf = _per_frame(noise, z.shape)
    gamma = power / noise_tf
    if cfg.prior_snr == "cepstral":
        xi = cepstral_prior_snr(gamma, noise_tf, sample_rate, hop_seconds, cfg)
    elif cfg.prior_snr == "ml_2d":
        xi = ml_2d_prior_snr(gamma, cfg)
    else:  # "dd_bidir": mean of forward and time-reversed DD estimates
        fwd = _dd_prior_snr(gamma, noise_tf, z, cfg.lsa_smoothing, floor)
        bwd = _dd_prior_snr(gamma[:, ::-1], noise_tf[:, ::-1], z[:, ::-1],
                            cfg.lsa_smoothing, floor)[:, ::-1]
        xi = 0.5 * (fwd + bwd)
    floor_tf = np.asarray(floor)[:, None] if np.ndim(floor) else floor
    gain = np.maximum(lsa_gain(xi, gamma), floor_tf)
    return gain * z


def band_floor_profile(n_bins, sample_rate, n_fft, cfg):
    """Round-02 step 1 (band_floor): per-bin LSA gain floor and low-band cap.

    Bands [0, f_c), [f_c, 1000), [1000, 4000), [4000, fs/2] Hz. Below f_c the output is
    capped at band_low_depth_db relative to the noisy input (the round-01 wind_module rule,
    which used the LSA floor, -15 dB, at f_c = 325 Hz); the LSA floor there stays
    lsa_gain_floor. The other bands get their own LSA floor. With band_smooth_bins = n > 0
    the floors (in dB) are interpolated linearly over n bins after each boundary and the cap
    is relaxed linearly in dB to 0 dB over n bins above f_c. All values: our choice
    (multiband idea after Kamath & Loizou, ICASSP 2002; original not checked).
    Returns (floor per bin, cap per bin with inf meaning no cap).
    """
    freqs = np.arange(n_bins) * sample_rate / n_fft
    edges = [cfg.band_cutoff_hz, 1000.0, 4000.0]
    floor_db = np.select(
        [freqs < edges[0], freqs < edges[1], freqs < edges[2]],
        [20 * np.log10(cfg.lsa_gain_floor), cfg.band_mid_floor_db, cfg.band_midhigh_floor_db],
        cfg.band_high_floor_db)
    cap_db = np.where(freqs < edges[0], cfg.band_low_depth_db, np.inf)
    n = cfg.band_smooth_bins
    if n > 0:
        for edge in edges:
            k0 = int(np.searchsorted(freqs, edge))  # first bin of the upper band
            if not 0 < k0 < n_bins:
                continue
            before, after = floor_db[k0 - 1], floor_db[min(k0 + n, n_bins - 1)]
            for j in range(n):
                if k0 + j < n_bins:
                    floor_db[k0 + j] = before + (after - before) * (j + 1) / (n + 1)
        k0 = int(np.searchsorted(freqs, edges[0]))
        for j in range(n):
            if k0 + j < n_bins:
                cap_db[k0 + j] = cfg.band_low_depth_db * (1 - (j + 1) / (n + 1))
    floor = 10 ** (floor_db / 20)
    if not n and np.all(floor_db == 20 * np.log10(cfg.lsa_gain_floor)):
        floor = np.full(n_bins, cfg.lsa_gain_floor)  # exactly the scalar floor's value
    cap = np.where(np.isfinite(cap_db), 10 ** (cap_db / 20), np.inf)
    return floor, cap
