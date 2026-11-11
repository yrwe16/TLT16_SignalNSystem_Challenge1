"""Roadmap stage 5: STFT phase reconstruction in voiced speech.

Krawczyk & Gerkmann, "STFT Phase Reconstruction in Voiced Speech for an Improved
Single-Channel Speech Enhancement", IEEE/ACM TASLP 22(12):1931-1940, 2014 (read in the
original, authors' PDF). For voiced frames with fundamental frequency f0:

  along frequency (eq. 13): for a band k dominated by harmonic h (nearest harmonic, eq. 8),
      phi(k) = princ(phi(k'_h) - phiW(k'_h - kappa_h) + phiW(k - kappa_h)),
      kappa_h = N h f0 / fs, k'_h = round(kappa_h), phiW = phase of the analysis window's
      DTFT (zero-padded DFT, Sec. IV-A3);
  along time (eq. 11, "combi", Sec. IV-B): in bands k'_h of consecutive voiced frames,
      phi(k'_h, l) = princ(phi(k'_h, l-1) + 2 pi h f0 L / fs); at a voiced onset the noisy
      phase of k'_h is used.
Harmonics up to 4 kHz (H = floor(4000/f0)) as in the paper. The paper combines the
reconstructed phase with the noisy amplitude (eq. 17); here it is combined with the stage-4
LSA amplitude. The paper's f0 estimator PEFAC uses a trained voicing model, so YIN
(de Cheveigne & Kawahara, JASA 111(4), 2002) is used instead (no training).
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window


def princ(x):
    return np.angle(np.exp(1j * x))


def yin_f0(x, sample_rate, centers, cfg):
    """YIN f0 per frame centre (sample index). Returns (f0 Hz, voiced bool)."""
    w = int(round(0.040 * sample_rate))
    tau_min = int(np.floor(sample_rate / cfg.phase_f0_max))
    tau_max = int(np.ceil(sample_rate / cfg.phase_f0_min))
    span = w + tau_max + 1
    padded = np.pad(x, (span, span))
    starts = np.asarray(centers) + span - w // 2
    seg = padded[starts[:, None] + np.arange(span)[None, :]]          # (frames, span)
    nfft = 1 << int(np.ceil(np.log2(2 * span)))
    a = np.zeros_like(seg)
    a[:, :w] = seg[:, :w]
    r = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(seg, nfft), nfft)[:, :tau_max + 1]
    cs = np.concatenate([np.zeros((len(seg), 1)), np.cumsum(seg ** 2, axis=1)], axis=1)
    e1 = cs[:, w][:, None]
    taus = np.arange(tau_max + 1)
    e2 = cs[:, taus + w] - cs[:, taus]
    d = np.maximum(e1 + e2 - 2 * r, 0.0)
    d[:, 0] = 0.0
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-20)

    f0 = np.zeros(len(seg))
    voiced = np.zeros(len(seg), dtype=bool)
    energy = e1[:, 0]
    loud = energy > 10 ** (cfg.phase_energy_db / 10) * max(energy.max(), 1e-20)
    for i in range(len(seg)):
        c = cmnd[i]
        below = np.flatnonzero(c[tau_min:tau_max] < cfg.phase_yin_threshold)
        if not len(below) or not loud[i]:
            continue
        t = tau_min + below[0]
        while t + 1 < tau_max and c[t + 1] < c[t]:
            t += 1
        if 1 <= t < tau_max:  # parabolic interpolation
            den = c[t - 1] - 2 * c[t] + c[t + 1]
            shift = 0.5 * (c[t - 1] - c[t + 1]) / den if den > 0 else 0.0
        else:
            shift = 0.0
        f0[i] = sample_rate / (t + np.clip(shift, -0.5, 0.5))
        voiced[i] = True
    return f0, voiced


def smooth_track(f0, voiced, cfg):
    """Offline variant: fill short gaps, drop short voiced runs, median-filter f0."""
    v = voiced.copy()
    gap = cfg.phase_fill_gap_frames
    idx = np.flatnonzero(v)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            v[a:b] = True
            f0[a:b] = np.linspace(f0[a], f0[b], b - a, endpoint=False)
    out = np.zeros_like(v)
    t = 0
    while t < len(v):
        if v[t]:
            e = t
            while e < len(v) and v[e]:
                e += 1
            if e - t >= cfg.phase_min_run_frames:
                out[t:e] = True
                f0[t:e] = median_filter(f0[t:e], size=5, mode="nearest")
            t = e
        else:
            t += 1
    return f0, out


@lru_cache(maxsize=8)
def _window_dtft(n_fft, oversample=64, max_offset=64):
    """Complex DTFT of the periodic Hann analysis window at bin offsets on a fine grid."""
    w = get_window("hann", n_fft)
    grid = np.arange(-max_offset * oversample, max_offset * oversample + 1) / oversample
    n = np.arange(n_fft)
    values = (w[None, :] * np.exp(-2j * np.pi * grid[:, None] * n[None, :] / n_fft)).sum(axis=1)
    return grid, values


def window_phase(offset, n_fft):
    grid, values = _window_dtft(n_fft)
    re = np.interp(offset, grid, values.real)
    im = np.interp(offset, grid, values.imag)
    return np.angle(re + 1j * im)


def reconstruct_phase(spec, f0, voiced, sample_rate, n_fft, hop, cfg):
    """Replace the phase of voiced frames below phase_max_hz by the harmonic-model phase."""
    bins, frames = spec.shape
    noisy_phase = np.angle(spec)
    phase = noisy_phase.copy()
    weight = np.zeros(frames)  # 1 in voiced frames, ramps at the edges when fading
    previous = None
    for t in range(frames):
        if not voiced[t]:
            previous = None
            continue
        f = f0[t]
        h = np.arange(1, int(np.floor(min(cfg.phase_max_hz, sample_rate / 2) / f)) + 1)
        if not len(h):
            previous = None
            continue
        kappa = n_fft * h * f / sample_rate
        kprime = np.clip(np.rint(kappa).astype(int), 0, bins - 1)
        if cfg.phase_mode == "combi" and previous is not None:
            anchor = princ(previous[kprime] + 2 * np.pi * h * f * hop / sample_rate)
        else:
            anchor = noisy_phase[kprime, t]
        spacing = n_fft * f / sample_rate
        top = min(bins, int(np.ceil((h[-1] + 0.5) * spacing)))
        k = np.arange(int(np.ceil(0.5 * spacing)), top)
        which = np.clip(np.rint(k / spacing).astype(int), 1, h[-1]) - 1
        phase[k, t] = princ(anchor[which] - window_phase(kprime[which] - kappa[which], n_fft)
                            + window_phase(k - kappa[which], n_fft))
        phase[kprime, t] = anchor
        previous = phase[:, t].copy()
        weight[t] = 1.0
    if cfg.phase_fade_frames > 0:
        weight = _fade(weight > 0, cfg.phase_fade_frames)
    delta = princ(phase - noisy_phase) * weight[None, :]
    return np.abs(spec) * np.exp(1j * (noisy_phase + delta))


def _fade(mask, n):
    """Linear ramps of n frames inside each voiced run (crossfade at voicing boundaries)."""
    w = mask.astype(float)
    t = 0
    while t < len(mask):
        if mask[t]:
            e = t
            while e < len(mask) and mask[e]:
                e += 1
            run = e - t
            ramp = np.minimum(np.minimum(np.arange(1, run + 1), np.arange(run, 0, -1)) / (n + 1), 1.0)
            w[t:e] = ramp
            t = e
        else:
            t += 1
    return w
