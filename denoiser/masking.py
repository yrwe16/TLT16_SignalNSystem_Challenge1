"""Round-02 step 8: auditory masking threshold (simultaneous masking) for the gain floor.

N. Virag, "Single channel speech enhancement based on masking properties of the human auditory
system", IEEE Trans. Speech Audio Process. 7(2):126-137, 1999, DOI 10.1109/89.748118 (read in
the original). Sec. III-A: the noise masking threshold T is obtained by
  1) summing the power spectrum in critical (Bark) bands,
  2) convolving with the spreading function of Schroeder, Atal & Hall (JASA 66, 1979; Fig. 3a),
  3) subtracting a relative threshold offset O(k) (Sinha & Tewfik 1993; Fig. 3b),
  4) renormalizing and comparing with the absolute threshold of hearing (Johnston 1988).
The threshold is computed from a first estimate of the clean speech (here: the output of the
current pipeline, as the team asked; the paper uses power spectral subtraction).

Details not given numerically in the paper, all marked:
- band edges: Zwicker critical bands (Zwicker & Fastl; original not checked), which reproduce
  Table I (8 kHz) and give K = 22 bands at 16 kHz as stated in Sec. III-A;
- spreading function: the closed form of Schroeder et al. as used by Johnston,
  SF(dz) = 15.81 + 7.5 (dz + 0.474) - 17.5 sqrt(1 + (dz + 0.474)^2) dB, dz = maskee - masker
  band (original not checked; it matches Fig. 3a: 0 dB at 0, about -100 dB at -5);
- O(k): values read from Fig. 3(b) (approximate, +-0.5 dB);
- renormalization: division by the spreading gain of a flat unit spectrum, and the band
  threshold is shared equally among the band's bins (Johnston; original not checked);
- the absolute threshold of hearing is not used because the playback level is unknown (our choice).
"""

import numpy as np

BARK_EDGES_HZ = (0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320,
                 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500)
# Fig. 3(b), relative threshold offset O(k) in dB for k = 1..25 (read from the figure)
OFFSET_DB = (-17, -18, -19, -20, -21, -22, -23, -24, -25, -25, -25, -25, -25, -25, -24, -23,
             -22, -19, -18, -18, -18, -18, -18, -18, -18)


def spreading_db(dz):
    dz = np.asarray(dz, dtype=float) + 0.474
    return 15.81 + 7.5 * dz - 17.5 * np.sqrt(1.0 + dz ** 2)


def band_index(n_bins, sample_rate, n_fft):
    """Critical band number (0-based) of every FFT bin."""
    freqs = np.arange(n_bins) * sample_rate / n_fft
    edges = np.asarray(BARK_EDGES_HZ, dtype=float)
    return np.clip(np.searchsorted(edges, freqs, side="right") - 1, 0, len(edges) - 2)


def masking_threshold(power, sample_rate, n_fft, offset_db=0.0):
    """Per-bin masking threshold (same shape as power: bins x frames).

    offset_db shifts the whole threshold (our grid parameter, 0 = paper values)."""
    n_bins = power.shape[0]
    band = band_index(n_bins, sample_rate, n_fft)
    k = band.max() + 1
    onehot = np.zeros((k, n_bins))
    onehot[band, np.arange(n_bins)] = 1.0
    energy = onehot @ power                                    # (k, frames)
    dz = np.arange(k)[:, None] - np.arange(k)[None, :]         # maskee - masker
    spread = 10 ** (spreading_db(dz) / 10)
    spread_gain = spread @ np.ones(k)
    offset = np.asarray(OFFSET_DB[:k], dtype=float) + offset_db
    band_threshold = (spread @ energy) / spread_gain[:, None] * 10 ** (offset[:, None] / 10)
    per_bin = band_threshold / onehot.sum(axis=1)[:, None]
    return per_bin[band]


def speech_frames(noisy_power, noise_tf, snr_db):
    """Speech-frame decision from the noisy input only (our choice): frame posterior SNR
    10 log10(sum|Y|^2 / sum N) >= snr_db. Checked against clean labels on Train (diagnostic):
    10 dB covers 43 % of speech frames and marks 0.3 % of pause frames."""
    ratio = noisy_power.sum(axis=0) / np.maximum(noise_tf.sum(axis=0), 1e-30)
    return 10 * np.log10(np.maximum(ratio, 1e-30)) >= snr_db


def masking_floor(first_power, noise_tf, speech, base_floor, sample_rate, n_fft, cfg):
    """Gain floor per bin and frame: base_floor everywhere, raised in speech frames only.

    In a speech frame the floor rises to the largest gain at which the residual noise stays at
    the masking threshold, sqrt(T / N) (Virag Sec. III-B: residual noise below the threshold is
    inaudible), but at most cfg.masking_range_db above the base floor; it is never lowered.
    (Virag's own adaptation, eqs. 8-9, interpolates alpha / beta between the frame's minimum
    and maximum of T; it is not used because a threshold offset would cancel in it.)
    Returns the floor (bins x frames)."""
    base = np.broadcast_to(np.asarray(base_floor, dtype=float).reshape(-1, 1)
                           if np.ndim(base_floor) else np.full((first_power.shape[0], 1), base_floor),
                           first_power.shape)
    floor = base.copy()
    if np.any(speech):
        t = masking_threshold(first_power[:, speech], sample_rate, n_fft, cfg.masking_offset_db)
        g_mask = np.sqrt(t / np.maximum(noise_tf[:, speech], 1e-30))
        cap = base[:, speech] * 10 ** (cfg.masking_range_db / 20)
        floor[:, speech] = np.clip(g_mask, base[:, speech], cap)
    return floor
