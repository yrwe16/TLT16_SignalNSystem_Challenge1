"""Noise power estimation from the noisy input only (no clean reference)."""

from functools import lru_cache

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter, uniform_filter1d


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


def complete_frames(times, length, sample_rate, n_fft, n_frames):
    """Indices of frames that are not zero-padded at the signal boundaries."""
    centers = np.rint(times * sample_rate).astype(int)
    full = np.flatnonzero((centers >= n_fft // 2) & (centers + n_fft // 2 <= length))
    return full if len(full) else np.arange(n_frames)


def percentile_noise(power, percentile):
    """Per-bin low percentile of the periodogram over the clip, smoothed over 3 bins.

    Bias-corrected for an exponential (2-dof chi-square) periodogram, whose p-quantile is
    -ln(1-p) times the mean. Offline variant: uses the whole clip, not only the past.
    """
    q = np.percentile(power, percentile, axis=1) / -np.log1p(-percentile / 100)
    return np.maximum(uniform_filter1d(q, size=3, mode="nearest"), 1e-18)


def spp_track(power, initial, hop_seconds, cfg):
    """Unbiased MMSE noise PSD tracking with a speech presence probability (SPP).

    Gerkmann & Hendriks, IEEE TASLP 20(4):1383-1393, 2012. For each frame and bin:
      P  = 1 / (1 + (1-q)/q * (1+xi_H1) * exp(-gamma * xi_H1/(1+xi_H1))),  gamma = |Y|^2 / N
      Ps = ap*Ps + (1-ap)*P;  where Ps > threshold: P = min(P, 1 - safety)   (stagnation)
      E  = (1-P)*|Y|^2 + P*N                                 (expected noise periodogram)
      N  = ax*N + (1-ax)*E
    Constants and update order follow VOICEBOX v_estnoiseg (LGPL-3; read and re-implemented,
    not copied). Returns the noise PSD used for each frame, shape (bins, frames).
    """
    xi_h1 = 10 ** (cfg.spp_snr_db / 10)
    pfac = (1 / cfg.spp_prior - 1) * (1 + xi_h1)
    slope = -xi_h1 / (1 + xi_h1)
    ax = np.exp(-hop_seconds / cfg.spp_noise_time)
    ap = np.exp(-hop_seconds / cfg.spp_prob_time)
    noise = np.maximum(np.asarray(initial, dtype=np.float64), 1e-18)
    smoothed = np.full_like(noise, cfg.spp_init_scale)
    out = np.empty_like(power)
    for t in range(power.shape[1]):
        y = power[:, t]
        prob = 1 / (1 + pfac * np.exp(slope * y / noise))
        smoothed = ap * smoothed + (1 - ap) * prob
        prob = np.where(smoothed > cfg.spp_prob_threshold,
                        np.minimum(prob, 1 - cfg.spp_prob_safety), prob)
        expected = (1 - prob) * y + prob * noise
        noise = np.maximum(ax * noise + (1 - ax) * expected, 1e-18)
        out[:, t] = noise
    return out


def tracked_noise(z, times, length, sample_rate, n_fft, hop, cfg):
    """Noise PSD per frame, shape (bins, frames), for noise_tracker == "spp"."""
    power = np.abs(z) ** 2
    full = complete_frames(times, length, sample_rate, n_fft, power.shape[1])
    hop_seconds = hop / sample_rate

    def initial(reverse):
        if cfg.spp_init == "lowenergy":
            return lowest_energy_frames(z, times, length, sample_rate, n_fft,
                                        cfg.noise_fraction)
        if cfg.spp_init == "percentile":
            return percentile_noise(power[:, full], cfg.spp_init_percentile)
        count = max(1, min(len(full), int(round(1 + cfg.spp_init_time / hop_seconds))))
        frames = (full[::-1] if reverse else full)[:count]
        return cfg.spp_init_scale * power[:, frames].mean(axis=1)

    forward = spp_track(power, initial(False), hop_seconds, cfg)
    if cfg.spp_direction == "forward":
        return forward
    backward = spp_track(power[:, ::-1], initial(True), hop_seconds, cfg)[:, ::-1]
    return 0.5 * (forward + backward)


def tonal_tracks(power, sample_rate, n_fft, hop_seconds, cfg):
    """Long-lasting narrowband peak tracks (sirens, alarms, bells, hum); boolean (bins, frames).

    Peak picking and frame-to-frame matching follow the idea of McAulay & Quatieri's
    sinusoidal tracking (IEEE TASSP 34(4):744-754, 1986): a peak continues the track whose
    last frequency is nearest within a matching interval, otherwise it starts a new track.
    Here a peak is a local maximum at least tonal_peak_db above the median of the spectrum
    over tonal_median_bins bins; a track may skip up to tonal_max_gap_frames frames; only
    tracks lasting at least tonal_min_seconds are kept, since the whole clip is available
    (offline) and speech harmonics rarely stay on one partial that long. All constants are
    our own choices (the paper's values were not checked).
    """
    from scipy.ndimage import median_filter
    bins, frames = power.shape
    freqs = np.arange(bins) * sample_rate / n_fft
    band = (freqs >= cfg.tonal_fmin) & (freqs <= cfg.tonal_fmax)
    local = median_filter(power, size=(cfg.tonal_median_bins, 1), mode="nearest")
    ratio = 10 ** (cfg.tonal_peak_db / 10)
    is_peak = np.zeros_like(power, dtype=bool)
    is_peak[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    is_peak &= (power > ratio * local) & band[:, None]

    active = []    # [last_bin, last_frame, [(frame, bin), ...]]
    finished = []
    for t in range(frames):
        peaks = list(np.flatnonzero(is_peak[:, t]))
        still = []
        for track in sorted(active, key=lambda tr: -len(tr[2])):  # longer tracks choose first
            if t - track[1] > cfg.tonal_max_gap_frames + 1:
                finished.append(track)
                continue
            if peaks:
                distances = np.abs(np.array(peaks) - track[0])
                j = int(np.argmin(distances))
                if distances[j] <= cfg.tonal_max_jump_bins:
                    k = peaks.pop(j)
                    track[0], track[1] = k, t
                    track[2].append((t, k))
            still.append(track)
        active = still + [[k, t, [(t, k)]] for k in peaks]
    finished.extend(active)

    mask = np.zeros_like(power, dtype=bool)
    level = np.zeros_like(power)  # median track power at its peak bins (track_median mode)
    min_frames = cfg.tonal_min_seconds / hop_seconds
    w = cfg.tonal_halfwidth_bins
    for _, _, points in finished:
        if points[-1][0] - points[0][0] + 1 < min_frames:
            continue
        median_peak = np.median([power[k, t] for t, k in points])
        for t, k in points:
            span = slice(max(k - w, 0), k + w + 1)
            mask[span, t] = True
            # spread the peak level over the main lobe with the shape seen in this frame
            shape = power[span, t] / max(power[k, t], 1e-30)
            level[span, t] = np.maximum(level[span, t], median_peak * np.minimum(shape, 1.0))
    tonal_tracks.level = level
    return mask


def tonal_tracks_r02(power, f0, voiced, sample_rate, n_fft, hop_seconds, cfg):
    """Round-02 step 7: revised narrowband tracker; boolean mask (bins, frames).

    Differences from tonal_tracks (6a), all our choices (background: Gul, Ariyaeeinia &
    Dewhirst, Eurospeech 2003, read in the original; alarms as sinusoids that continue
    independently of speech; their LP-based filter is not used):
    1. in voiced frames, peaks within tonal_f0_guard_bins of a multiple of the YIN f0 are
       dropped (speech harmonics);
    2. a peak continues a track if it lies within tonal_slope_hz_s x elapsed time (at least
       one bin) of the track's last bin, i.e. the track follows sweeps frame by frame;
    3. tracks are kept if they last tonal_r02_min_seconds and at least tonal_unvoiced_min of
       their frames are unvoiced or silent (YIN), i.e. they continue outside speech;
    0. peaks are searched from tonal_r02_fmin (400 Hz) instead of tonal_fmin;
    4. the mask covers +-tonal_halfwidth_bins around each track point and around confirmed
       peaks (+-1 bin) at the integer multiples of its frequency, up to tonal_fmax.
    Returns (mask, list of kept tracks as [(frame, bin), ...])."""
    from scipy.ndimage import median_filter
    bins, frames = power.shape
    df = sample_rate / n_fft
    freqs = np.arange(bins) * df
    band = (freqs >= cfg.tonal_r02_fmin) & (freqs <= cfg.tonal_fmax)
    local = median_filter(power, size=(cfg.tonal_median_bins, 1), mode="nearest")
    ratio = 10 ** (cfg.tonal_peak_db / 10)
    is_peak = np.zeros_like(power, dtype=bool)
    is_peak[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    is_peak &= (power > ratio * local) & band[:, None]
    g = cfg.tonal_f0_guard_bins
    for t in np.flatnonzero(voiced):
        if f0[t] <= 0:
            continue
        centres = np.rint(np.arange(f0[t], cfg.tonal_fmax + f0[t], f0[t]) / df).astype(int)
        for c in centres:
            is_peak[max(c - g, 0):c + g + 1, t] = False

    step = cfg.tonal_slope_hz_s * hop_seconds / df    # bins per frame
    active, finished = [], []
    for t in range(frames):
        peaks = list(np.flatnonzero(is_peak[:, t]))
        still = []
        for track in sorted(active, key=lambda tr: -len(tr[2])):
            elapsed = t - track[1]
            if elapsed > cfg.tonal_max_gap_frames + 1:
                finished.append(track)
                continue
            if peaks:
                distances = np.abs(np.array(peaks) - track[0])
                j = int(np.argmin(distances))
                if distances[j] <= max(1.0, step * elapsed):
                    k = peaks.pop(j)
                    track[0], track[1] = k, t
                    track[2].append((t, k))
            still.append(track)
        active = still + [[k, t, [(t, k)]] for k in peaks]
    finished.extend(active)

    mask = np.zeros_like(power, dtype=bool)
    min_frames = cfg.tonal_r02_min_seconds / hop_seconds
    w = cfg.tonal_halfwidth_bins
    top = min(int(cfg.tonal_fmax / df), bins - 1)
    kept = []
    for _, _, points in finished:
        if points[-1][0] - points[0][0] + 1 < min_frames:
            continue
        unvoiced = np.mean([not voiced[t] for t, _ in points])
        if unvoiced < cfg.tonal_unvoiced_min:
            continue
        kept.append(points)
        for t, k in points:
            mask[max(k - w, 0):k + w + 1, t] = True
            m = 2
            while m * k <= top:
                c = int(round(m * k))
                near = np.flatnonzero(is_peak[max(c - 1, 0):c + 2, t])
                if len(near):
                    p = max(c - 1, 0) + near[0]
                    mask[max(p - w, 0):p + w + 1, t] = True
                m += 1
    return mask, kept


def add_tonal_noise(noise, z, sample_rate, n_fft, hop, cfg, f0_voiced=None):
    """Noise PSD per frame: the background estimate plus the observed power on tonal tracks."""
    power = np.abs(z) ** 2
    base = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, power.shape)
    if cfg.tonal_mode == "r02":
        mask, kept = tonal_tracks_r02(power, *f0_voiced, sample_rate, n_fft, hop / sample_rate,
                                      cfg)
        add_tonal_noise.tracks = kept
    else:
        mask = tonal_tracks(power, sample_rate, n_fft, hop / sample_rate, cfg)
    add_tonal_noise.share = float(mask.mean())
    add_tonal_noise.mask = mask
    if cfg.tonal_level == "track_median":
        tonal = np.minimum(tonal_tracks.level, power)
    else:  # "observed": all power on the track bins counts as noise
        tonal = np.where(mask, power, 0.0)
    return base + cfg.tonal_weight * tonal


def rpca(d, lam, max_iter, tol=1e-7, rho=1.5):
    """Principal component pursuit (Candes, Li, Ma & Wright, J. ACM 58(3), 2011):
    min ||L||_* + lam ||S||_1 s.t. D = L + S. Solved by the inexact augmented Lagrangian
    method with a growing penalty mu (Lin, Chen & Ma, 2010; mu0 = 1.25/||D||_2, rho = 1.5),
    which reaches the same solution as the paper's fixed-mu Algorithm 1 in far fewer
    iterations. Re-implemented; dganguli/robust-pca (MIT) was read for comparison only."""
    norm2 = np.linalg.norm(d, 2) + 1e-30
    mu = 1.25 / norm2
    mu_max = mu * 1e7
    y = d / max(norm2, np.abs(d).max() / lam)
    s = np.zeros_like(d)
    norm = np.linalg.norm(d) + 1e-30
    for _ in range(max_iter):
        u, sig, vt = np.linalg.svd(d - s + y / mu, full_matrices=False)
        low = (u * np.maximum(sig - 1 / mu, 0)) @ vt
        r = d - low + y / mu
        s = np.sign(r) * np.maximum(np.abs(r) - lam / mu, 0)
        residual = d - low - s
        y += mu * residual
        mu = min(mu * rho, mu_max)
        if np.linalg.norm(residual) <= tol * norm:
            break
    return low, s


def rpca_noise(noise, z, cfg):
    """Stage 6 module for machine / engine / cockpit noise: the low-rank part of the
    magnitude spectrogram (Huang et al., ICASSP 2012 use RPCA on magnitudes) as a noise
    PSD per frame, combined with the background estimate ("replace" or "max")."""
    mag = np.abs(z)
    scale = mag.max() + 1e-30
    lam = cfg.rpca_lambda_scale / np.sqrt(max(mag.shape))
    low, _ = rpca(mag / scale, lam, cfg.rpca_max_iter)
    low_psd = (np.maximum(low, 0) * scale) ** 2
    base = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, mag.shape)
    if cfg.rpca_combine == "max":
        return np.maximum(base, low_psd)
    return np.maximum(low_psd, 1e-18)


def transient_noise(noise, z, cfg):
    """Stage 6 module for clapping / clicks / door / footsteps: percussive part by
    median-filtering HPSS (FitzGerald, DAFx 2010; 17-bin/17-frame medians and a margin as
    in later HPSS work; constants not checked against the paper) added to the noise PSD,
    only in frames where the percussive mask covers a large share of the band (broadband
    transients, cf. Talmon, Cohen & Gannot, IEEE TASLP 19(6), 2011)."""
    from scipy.ndimage import median_filter
    mag = np.abs(z)
    harmonic = median_filter(mag, size=(1, 17), mode="nearest")
    percussive = median_filter(mag, size=(17, 1), mode="nearest")
    mask = percussive > cfg.transient_margin * harmonic
    frames = mask.mean(axis=0) >= cfg.transient_frame_share
    mask &= frames[None, :]
    transient_noise.share = float(frames.mean())
    base = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, mag.shape)
    return base + np.where(mask, mag ** 2, 0.0)


NOISE_BAND_EDGES = (0.0, 325.0, 1000.0, 2000.0, 4000.0)


def band_frame_noise(z, times, length, sample_rate, n_fft, fraction):
    """Round-02 step 2 (b): choose the quietest complete frames separately per band
    (0/325/1k/2k/4k/fs/2 Hz) by the band energy, average their power, smooth over 3 bins.
    Band edges and the per-band selection are our choice."""
    power = np.abs(z) ** 2
    full = complete_frames(times, length, sample_rate, n_fft, power.shape[1])
    freqs = np.arange(power.shape[0]) * sample_rate / n_fft
    edges = list(NOISE_BAND_EDGES) + [np.inf]
    estimate = np.empty(power.shape[0])
    count = max(1, int(np.ceil(fraction * len(full))))
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        if not band.any():
            continue
        energy = power[np.ix_(band, full)].mean(axis=0)
        chosen = full[np.argsort(energy, kind="stable")[:count]]
        estimate[band] = power[np.ix_(band, chosen)].mean(axis=1)
    return np.maximum(uniform_filter1d(estimate, size=3, mode="nearest"), 1e-18)


def spp_weighted_noise(z, times, length, sample_rate, n_fft, initial, snr_db, iterations,
                       start_db):
    """Round-02 step 2 (c): noise PSD as an SPP-weighted average over the whole clip.

    P(H1|Y) = 1 / (1 + (1+xi_H1) exp(-gamma xi_H1/(1+xi_H1))) with P(H1) = 0.5
    (a posteriori SPP of Gerkmann & Hendriks, IEEE TASLP 20(4), 2012, as in VOICEBOX
    v_estnoiseg), gamma = |Y|^2 / sigma^2. Then sigma^2(k) = sum_t P(H0)|Y|^2 / sum_t P(H0)
    over the complete frames, repeated `iterations` times from initial * 10^(start_db/10),
    and smoothed over 3 bins. The batch fixed-point use over the clip is our choice.
    """
    power = np.abs(z) ** 2
    full = complete_frames(times, length, sample_rate, n_fft, power.shape[1])
    y = power[:, full]
    xi_h1 = 10 ** (snr_db / 10)
    sigma = np.maximum(np.asarray(initial, float) * 10 ** (start_db / 10), 1e-18)
    for _ in range(iterations):
        gamma = y / sigma[:, None]
        p_absent = 1 - 1 / (1 + (1 + xi_h1) * np.exp(-gamma * xi_h1 / (1 + xi_h1)))
        weight = p_absent.sum(axis=1)
        new = (p_absent * y).sum(axis=1) / np.maximum(weight, 1e-12)
        sigma = np.where(weight > 1e-6, np.maximum(new, 1e-18), sigma)
    return np.maximum(uniform_filter1d(sigma, size=3, mode="nearest"), 1e-18)


def scale_noise_bands(noise, sample_rate, n_fft, beta_low, beta, beta_high):
    """Round-02 step 2 (a): multiply the estimate by beta_low below 325 Hz, beta in
    [325, 4000) Hz and beta_high from 4 kHz. Applied after the 3-bin smoothing, per bin, so a
    band's factor does not leak into the neighbouring band. Values: our choice."""
    freqs = np.arange(noise.shape[0]) * sample_rate / n_fft
    factor = np.where(freqs < 325.0, beta_low, np.where(freqs < 4000.0, beta, beta_high))
    return noise * (factor if noise.ndim == 1 else factor[:, None])


VARYING_BANDS = (0.0, 325.0, 1000.0, 4000.0)


@lru_cache(maxsize=32)
def quantile_bias(quantile, window_frames, n_fft, hop):
    """Mean of the sliding-window quantile / true power for white noise through the same STFT
    (Hann, n_fft, hop, centred percentile filter over window_frames). Computed numerically from
    60 s of fixed-seed white noise instead of the iid exponential formula, because overlapping
    frames are correlated (round-02 step 3 rule; our choice of signal length and seed)."""
    from .stft import stft
    rng = np.random.default_rng(12345)
    x = rng.standard_normal(16000 * 60)
    _, _, z = stft(x, 16000, n_fft, hop)
    power = np.abs(z[:, 4:-4]) ** 2
    q = percentile_filter(power, 100 * quantile, size=(1, window_frames), mode="nearest")
    return float(q[:, window_frames:-window_frames].mean() / power.mean())


def quantile_window_noise(z, times, length, sample_rate, n_fft, hop, fixed, cfg):
    """Round-02 step 3: time-varying noise PSD from a centred (non-causal) sliding-window
    quantile per frequency bin, combined with the fixed step-2 estimate `fixed`.

    Quantile-based noise estimation after Stahl, Fischer & Bippus, ICASSP 2000, vol. 3,
    pp. 1875-1878, DOI 10.1109/ICASSP.2000.862122 (bibliography checked; full text not read,
    "원문 미확인": the sliding window, bias table, smoothing and combination are our choice).
    1. per bin, quantile q of |Y|^2 over a window of noise_window_s centred on each frame
       (zero-padded edge frames replaced by the nearest complete frame);
    2. divided by the numerical white-noise bias of that quantile and window;
    3. smoothed over noise_smooth_frames frames x 3 bins;
    4. combined with the fixed estimate S(k): "max" = max(S, Q); "mix" = w Q + (1-w) S;
       "level" = S(k) g_b(t), g_b = band ratio sum Q / sum S in 4 bands (0/325/1k/4k Hz),
       moving-averaged over 200 ms and limited to [-6, +12] dB.
    """
    power = np.abs(z) ** 2
    full = complete_frames(times, length, sample_rate, n_fft, power.shape[1])
    if len(full) < power.shape[1]:
        idx = np.clip(np.arange(power.shape[1]), full[0], full[-1])
        power = power[:, idx]
    hop_s = hop / sample_rate
    window = max(3, int(round(cfg.noise_window_s / hop_s)) | 1)
    q = percentile_filter(power, 100 * cfg.noise_quantile, size=(1, window), mode="nearest")
    q = q / quantile_bias(cfg.noise_quantile, window, n_fft, hop)
    q = uniform_filter(q, size=(3, cfg.noise_smooth_frames), mode="nearest")
    q = np.maximum(q, 1e-18)
    fixed_tf = np.broadcast_to(fixed[:, None], q.shape)
    if cfg.noise_combine == "max":
        return np.maximum(fixed_tf, q)
    if cfg.noise_combine == "mix":
        return cfg.noise_mix_weight * q + (1 - cfg.noise_mix_weight) * fixed_tf
    freqs = np.arange(q.shape[0]) * sample_rate / n_fft
    edges = list(VARYING_BANDS) + [np.inf]
    level = np.empty_like(q)
    span = max(1, int(round(0.2 / hop_s)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        ratio = q[band].sum(axis=0) / max(fixed[band].sum(), 1e-30)
        ratio_db = uniform_filter1d(10 * np.log10(np.maximum(ratio, 1e-30)), size=span,
                                    mode="nearest")
        level[band] = 10 ** (np.clip(ratio_db, -6.0, 12.0) / 10)
    return fixed_tf * level


def file_snr_features(z, noise, sample_rate, n_fft, cutoff_hz):
    """Round-02 step 4: file-level SNR features from the noisy STFT and the noise estimate only.

    snr_lowband_db = 10 log10( sum_{t, f<cutoff} max(|Y|^2 - N, 0) / sum_{t, f<cutoff} N ),
    snr_file_db the same over all bins, lowband_energy_ratio_db = low-band / total noisy
    energy. N may be one vector or one vector per frame."""
    power = np.abs(z) ** 2
    noise_tf = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, power.shape)
    low = np.arange(power.shape[0]) * sample_rate / n_fft < cutoff_hz
    clean_est = np.maximum(power - noise_tf, 0.0)
    db = lambda a, b: float(10 * np.log10(max(a, 1e-30) / max(b, 1e-30)))
    return {"snr_lowband_db": db(clean_est[low].sum(), noise_tf[low].sum()),
            "snr_file_db": db(clean_est.sum(), noise_tf.sum()),
            "lowband_energy_ratio_db": db(power[low].sum(), power.sum())}


def _window_min(values, length, causal):
    """Minimum over the last (causal) or next (anti-causal) `length` frames, edges excluded."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = [(0, 0), (length - 1, 0)] if causal else [(0, 0), (0, length - 1)]
    padded = np.pad(values, pad, constant_values=np.inf)
    return sliding_window_view(padded, length, axis=1).min(axis=-1)


def two_stage_transient(x, sample_rate, long_shape, n_fft, hop, cfg, stationary=None):
    """Round-02 step 9: transient PSD from a short-frame analysis (transient_mode="two_stage").

    A. Hirszhorn, D. Dov, R. Talmon, I. Cohen, "Transient interference suppression in speech
    signals based on the OM-LSA algorithm", IWAENC 2012 (read in the original), Sec. 3.1:
    short STFT frames (64 samples = 4 ms at 16 kHz, 75 % overlap), smoothed periodogram
    S = a_s S(l-1) + (1 - a_s)|Y|^2 with a_s = 0.7 (eq. 5), causal minimum over L frames
    (eq. 6), anti-causal minimum over T = 40 ms (eq. 12), S_min = max of the two (eq. 13),
    transient where S / S_min > delta (eqs. 7-8). The paper gives no values for L and delta.
    Simplified (as the team asked): no first-stage OM-LSA; the transient power is the excess
    S - S_min on flagged cells (our choice), mapped to the long STFT by averaging the short
    frames under the long window (Hann^2 weights) and interpolating over frequency, scaled
    by n_short / n_long (scipy's STFT normalisation; our choice). The caller adds it to the
    stationary estimate for the LSA (the paper's OM-LSA is not used). File gate (our choice):
    a short frame is transient when at least transient_bin_share of its bins are flagged; the
    module acts only if the share of such frames is >= transient_file_index.
    Team decision 11-16 after the first measurement (S / S_min > delta flagged 8-34 % of the
    frames of ordinary files: 4 ms periodograms smoothed with a_s 0.7 fluctuate strongly): the
    reference level is max(S_min, stationary noise PSD mapped to the short grid) (our choice),
    so fluctuations of stationary noise are not taken as transients; the excess is measured
    from that reference.
    Returns (transient PSD on the long grid or None when gated off, transience index)."""
    from scipy.signal import lfilter
    from .stft import stft
    n_s = cfg.transient_short_n
    hop_s = max(1, n_s // 4)
    _, t_s, zs = stft(x, sample_rate, n_s, hop_s)
    p = np.abs(zs) ** 2
    a = cfg.transient_alpha_s
    s, _ = lfilter([1 - a], [1, -a], p, axis=1, zi=a * p[:, :1])
    frame_s = hop_s / sample_rate
    causal = _window_min(s, max(1, int(round(cfg.transient_causal_s / frame_s))), True)
    ahead = _window_min(s, max(1, int(round(cfg.transient_anticausal_s / frame_s))), False)
    s_min = np.maximum(causal, ahead)
    f_s = np.arange(p.shape[0]) * sample_rate / n_s
    if stationary is not None:
        f_long = np.arange(len(stationary)) * sample_rate / n_fft
        floor = np.interp(f_s, f_long, stationary) * (n_fft / n_s)   # long -> short scaling
        s_min = np.maximum(s_min, floor[:, None])
    flag = s > cfg.transient_delta * np.maximum(s_min, 1e-30)
    frames = flag.mean(axis=0) >= cfg.transient_bin_share
    index = float(frames.mean())
    if index < cfg.transient_file_index:
        return None, index
    excess = np.where(flag & frames[None, :], np.maximum(s - s_min, 0.0), 0.0)
    # time: weighted mean of the short frames under each long analysis window
    n_bins_l, n_frames_l = long_shape
    t_l = np.arange(n_frames_l) * hop / sample_rate
    half = n_fft / (2 * sample_rate)
    d = (t_s[None, :] - t_l[:, None]) / half                          # in (-1, 1) inside
    w = np.where(np.abs(d) < 1, np.cos(0.5 * np.pi * d) ** 4, 0.0)    # Hann^2 of the long window
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-30)
    per_frame = excess @ w.T                                          # (short bins, long frames)
    f_l = np.arange(n_bins_l) * sample_rate / n_fft
    out = np.empty((n_bins_l, n_frames_l))
    for j in range(n_frames_l):
        out[:, j] = np.interp(f_l, f_s, per_frame[:, j])
    return out * (n_s / n_fft), index


def tracked_protected_noise(power, fixed, f0, voiced, voiced_ok, hop_seconds, sample_rate, n_fft,
                            cfg):
    """Round-02 step 11 (version A): time-varying noise PSD with speech protection.

    Starting point: the stationary SPP-weighted estimate `fixed` (spp_iter). In every
    unprotected cell the noise PSD follows the SPP-weighted expected noise periodogram,
        P = 1 / (1 + (1 + xi_H1) exp(-gamma xi_H1 / (1 + xi_H1))),  gamma = |Y|^2 / N
        N <- a N + (1 - a) [(1 - P)|Y|^2 + P N],  a = exp(-hop / tau)
    (update of Gerkmann & Hendriks, IEEE TASLP 20(4), 2012 as in VOICEBOX v_estnoiseg;
    original not checked). Protected cells keep the previous value (no update while speech is
    present, the idea of IMCRA, Cohen, IEEE TSAP 11(5), 2003; original not checked):
      - in voiced frames (YIN on a first pass), bins within +-1 of multiples of f0;
      - whole frames whose posterior SNR sum|Y|^2 / sum fixed >= tracked_frame_snr_db.
    Run forward and time-reversed from `fixed`, combined by the cell-wise minimum (or mean,
    tracked_combine) and kept above fixed x 10^(tracked_floor_db/10) (None = no floor); our choices. tau, the frame SNR threshold and xi_H1 (noise_spp_snr_db) are
    our choices. Returns the noise PSD per frame (bins x frames)."""
    bins, frames = power.shape
    df = sample_rate / n_fft
    protect = np.zeros_like(power, dtype=bool)
    frame_snr = 10 * np.log10(np.maximum(power.sum(axis=0), 1e-30)
                              / max(float(fixed.sum()), 1e-30))
    protect[:, frame_snr >= cfg.tracked_frame_snr_db] = True
    for t in np.flatnonzero(np.asarray(voiced) & np.asarray(voiced_ok)):
        if f0[t] <= 0:
            continue
        centres = np.rint(np.arange(f0[t], sample_rate / 2, f0[t]) / df).astype(int)
        for c in centres:
            protect[max(c - 1, 0):c + 2, t] = True
    xi = 10 ** (cfg.noise_spp_snr_db / 10)
    a = np.exp(-hop_seconds / cfg.tracked_tau_s)

    def run(p, guard):
        n = fixed.astype(float).copy()
        out = np.empty_like(p)
        for t in range(p.shape[1]):
            y = p[:, t]
            prob = 1 / (1 + (1 + xi) * np.exp(-(y / np.maximum(n, 1e-30)) * xi / (1 + xi)))
            update = a * n + (1 - a) * ((1 - prob) * y + prob * n)
            n = np.where(guard[:, t], n, update)
            out[:, t] = n
        return out

    forward = run(power, protect)
    backward = run(power[:, ::-1], protect[:, ::-1])[:, ::-1]
    both = np.minimum(forward, backward) if cfg.tracked_combine == "min" else 0.5 * (forward + backward)
    if cfg.tracked_floor_db is None:
        return np.maximum(both, 1e-18)
    return np.maximum(both, fixed[:, None] * 10 ** (cfg.tracked_floor_db / 10))


def harmonic_tunnel_noise(power, base, f0, voiced, sample_rate, n_fft, cfg):
    """Round-02 step 12 (version B): noise sampled between speech harmonics in voiced frames.

    D. Ealey, H. Kelleher, D. Pearce, "Harmonic tunnelling: tracking non-stationary noises
    during speech", Eurospeech 2001, pp. 437-440, DOI 10.21437/Eurospeech.2001-117 (read in
    the original, ISCA open archive). From the paper: the noise at a harmonic peak is
    interpolated linearly between end points that are the averages of up to N <= 4 tunnel
    values on each side (Sec. 2.4.1), and the frame estimates enter a running average in which
    the current frame contributes 25 % (Sec. 3). Simplified (team prompt, our choice): the
    harmonic regions are the bins within +-harmonic_halfwidth_bins of multiples of the YIN f0
    instead of the paper's structural peak classification (Sec. 2.3) and its tunnel-edge rule
    (Fig. 5); the paper warns that pure pitch-multiple ("comb") sampling can be distorted.
    The tunnel estimate is scaled by harmonic_bias_db and combined with the existing estimate
    in voiced frames only, by the mean (weight 0.5) or the maximum (harmonic_combine).
    Returns (noise PSD per frame, tunnel estimate per frame with NaN outside voiced frames)."""
    bins, frames = power.shape
    df = sample_rate / n_fft
    w = cfg.harmonic_halfwidth_bins
    base = np.broadcast_to(base[:, None] if base.ndim == 1 else base, power.shape)
    out = base.copy()
    tunnel_est = np.full(power.shape, np.nan)
    running = None
    scale = 10 ** (cfg.harmonic_bias_db / 10)
    idx = np.arange(bins)
    for t in range(frames):
        if not voiced[t] or f0[t] <= 0:
            running = None
            continue
        harm = np.zeros(bins, dtype=bool)
        for c in np.rint(np.arange(f0[t], sample_rate / 2, f0[t]) / df).astype(int):
            harm[max(c - w, 0):c + w + 1] = True
        tunnel = ~harm
        if tunnel.sum() < 2:
            running = None
            continue
        est = np.where(tunnel, power[:, t], 0.0)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], harm.astype(int), [0]])))
        for start, stop in zip(edges[0::2], edges[1::2]):
            left = idx[(idx < start) & tunnel][-4:]
            right = idx[(idx >= stop) & tunnel][:4]
            if not len(left) and not len(right):
                continue
            lv = power[left, t].mean() if len(left) else power[right, t].mean()
            rv = power[right, t].mean() if len(right) else lv
            frac = (np.arange(start, stop) - start + 1) / (stop - start + 1)
            est[start:stop] = lv + (rv - lv) * frac
        est = est * scale
        running = est if running is None else 0.75 * running + 0.25 * est
        tunnel_est[:, t] = running
        if cfg.harmonic_combine == "max":
            out[:, t] = np.maximum(base[:, t], running)
        else:
            out[:, t] = 0.5 * base[:, t] + 0.5 * running
    return out, tunnel_est
