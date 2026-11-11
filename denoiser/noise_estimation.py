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
