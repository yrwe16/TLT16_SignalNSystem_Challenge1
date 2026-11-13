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


def add_tonal_noise(noise, z, sample_rate, n_fft, hop, cfg):
    """Noise PSD per frame: the background estimate plus the observed power on tonal tracks."""
    power = np.abs(z) ** 2
    base = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, power.shape)
    mask = tonal_tracks(power, sample_rate, n_fft, hop / sample_rate, cfg)
    add_tonal_noise.share = float(mask.mean())
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
