"""Signal-, file- and folder-level noise reduction."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import CONFIG
from .gains import (band_floor_profile, lsa_decision_directed, lsa_with_prior,
                    musical_noise_postfilter, omlsa_decision_directed, wiener_decision_directed)
from .noise_estimation import (add_tonal_noise, band_frame_noise, complete_frames,
                               gated_transient_noise, shape_gain_noise,
                               file_snr_features, lowest_energy_frames,
                               quantile_window_noise, rpca_noise, scale_noise_bands, spp_weighted_noise,
                               harmonic_tunnel_noise, tracked_noise, tracked_protected_noise,
                               transient_noise, two_stage_transient)
from .masking import masking_floor, speech_frames
from .phase import reconstruct_phase, smooth_track, yin_f0
from .stft import as_waveform, istft, stft


def snr_adaptive_offset_db(snr_hat_db, cfg):
    """Round-02 step 4 part 2: raise the LSA floor (above the low band) by an offset that grows
    linearly in dB from 0 at snr_adapt_start_db to (snr_adapt_floor15_db - band floor) at
    15 dB, with the file SNR estimate clipped to [0, 15] dB. Values: our choice (SNR-dependent
    suppression after Berouti et al., ICASSP 1979; original not checked)."""
    snr = float(np.clip(snr_hat_db, 0.0, 15.0))
    if snr <= cfg.snr_adapt_start_db:
        return 0.0
    full_offset = cfg.snr_adapt_floor15_db - cfg.band_mid_floor_db
    return full_offset * (snr - cfg.snr_adapt_start_db) / (15.0 - cfg.snr_adapt_start_db)


def tonal_f0(x, sample_rate, times, cfg):
    """Round-02 step 7: YIN f0 per STFT frame for the tonal module, independent of the phase
    stage. YIN runs on a first pass of the same pipeline without the tonal module (and
    without phase reconstruction), then the offline smooth_track. Our choice."""
    first = denoise_signal(x, sample_rate, replace(cfg, tonal_module=False,
                                                   phase_reconstruction=False))
    centers = np.rint(times * sample_rate).astype(int)
    f0, voiced = yin_f0(first, sample_rate, centers, cfg)
    return smooth_track(f0, voiced, cfg)


def apply_masking_floor(x, sample_rate, z, enhanced, noise, band, times, n_fft, hop, cfg):
    """Round-02 step 8: raise the gain floor in speech frames from the masking threshold of the
    first-pass output (the same pipeline with masking off, re-analysed with the same STFT); see
    masking.py. Gains in other frames, and bins whose gain is already above the new floor, are
    left exactly as they are."""
    first = denoise_signal(x, sample_rate, replace(cfg, masking=False))
    _, _, s1 = stft(first, sample_rate, n_fft, hop)
    power = np.abs(z) ** 2
    noise_tf = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, z.shape)
    speech = speech_frames(power, noise_tf, cfg.masking_frame_snr_db)
    base = cfg.lsa_gain_floor if band is None else band[0]
    floor = masking_floor(np.abs(s1) ** 2, noise_tf, speech, base, sample_rate, n_fft, cfg)
    gain = np.abs(enhanced) / np.maximum(np.abs(z), 1e-30)
    # speech frames only; the 1e-9 margin ignores the rounding of |G Y| / |Y| at the base floor
    raise_ = speech[None, :] & (floor > gain * (1 + 1e-9))
    apply_masking_floor.speech = speech
    apply_masking_floor.raised = raise_
    return np.where(raise_, floor * z, enhanced)


def first_pass_f0(x, sample_rate, times, cfg):
    """YIN f0 (+ the offline smooth_track) on a first pass of the same pipeline with the
    stationary spp_iter estimate and without the harmonic step (steps 11-12; no recursion)."""
    first = denoise_signal(x, sample_rate, replace(cfg, noise_estimator="spp_iter",
                                                   harmonic_noise=False, transient_module=False,
                                                   phase_reconstruction=False))
    centers = np.rint(times * sample_rate).astype(int)
    return smooth_track(*yin_f0(first, sample_rate, centers, cfg), cfg)


def protected_noise(x, sample_rate, z, fixed, times, n_fft, hop, cfg, f0_voiced=None):
    """Round-02 step 11 (version A): speech-protected time-varying noise PSD. YIN (+ the
    offline smooth_track) runs on a first pass of the same pipeline with the stationary
    spp_iter estimate (no recursion), as in step 7."""
    f0, voiced = f0_voiced if f0_voiced is not None else first_pass_f0(x, sample_rate, times, cfg)
    ok = np.ones_like(voiced)
    return tracked_protected_noise(np.abs(z) ** 2, fixed, f0, voiced, ok, hop / sample_rate,
                                   sample_rate, n_fft, cfg)


def estimate_noise_psd(z, times, length, sample_rate, n_fft, hop, cfg=CONFIG):
    """Background noise PSD from the noisy STFT only (one vector, or one per frame)."""
    if cfg.noise_tracker == "spp":
        return tracked_noise(z, times, length, sample_rate, n_fft, hop, cfg)
    if cfg.noise_estimator == "band_frames":
        noise = band_frame_noise(z, times, length, sample_rate, n_fft, cfg.noise_band_fraction)
    else:
        noise = lowest_energy_frames(z, times, length, sample_rate, n_fft, cfg.noise_fraction)
        if cfg.noise_estimator in ("spp_iter", "quantile_window", "tracked_protected"):
            noise = spp_weighted_noise(z, times, length, sample_rate, n_fft, noise,
                                       cfg.noise_spp_snr_db, cfg.noise_spp_iterations,
                                       cfg.noise_spp_start_db)
        if cfg.noise_estimator == "quantile_window":
            noise = quantile_window_noise(z, times, length, sample_rate, n_fft, hop, noise, cfg)
    if (cfg.noise_beta_low, cfg.noise_beta, cfg.noise_beta_high) != (1.0, 1.0, 1.0):
        noise = scale_noise_bands(noise, sample_rate, n_fft, cfg.noise_beta_low,
                                  cfg.noise_beta, cfg.noise_beta_high)
    return noise


def denoise_signal(x, sample_rate, cfg=CONFIG):
    """Denoise one mono channel. The output has exactly the input length."""
    x = as_waveform(x)
    if not np.any(x):
        return x.copy()
    n_fft, hop = cfg.stft_sizes(sample_rate)
    _, times, z = stft(x, sample_rate, n_fft, hop)
    noise = estimate_noise_psd(z, times, len(x), sample_rate, n_fft, hop, cfg)
    f0_voiced = None
    if cfg.harmonic_noise or cfg.noise_estimator == "tracked_protected":
        f0_voiced = first_pass_f0(x, sample_rate, times, cfg)
    fixed = noise
    if cfg.noise_estimator == "tracked_protected":
        noise = protected_noise(x, sample_rate, z, noise, times, n_fft, hop, cfg, f0_voiced)
    if cfg.shape_gain_noise and fixed.ndim == 1:
        noise = shape_gain_noise(np.abs(z) ** 2, noise, fixed, sample_rate, n_fft, cfg)
    if cfg.harmonic_noise:
        noise, _ = harmonic_tunnel_noise(np.abs(z) ** 2, noise, *f0_voiced, sample_rate, n_fft,
                                         cfg)
    stationary = noise
    if cfg.tonal_module:
        f0_voiced = None
        if cfg.tonal_mode == "r02":
            f0_voiced = tonal_f0(x, sample_rate, times, cfg)
        noise = add_tonal_noise(noise, z, sample_rate, n_fft, hop, cfg, f0_voiced)
    if cfg.rpca_module:
        noise = rpca_noise(noise, z, cfg)
    if cfg.transient_module and cfg.transient_mode == "two_stage":
        extra, index = two_stage_transient(x, sample_rate, z.shape, n_fft, hop, cfg,
                                           stationary if stationary.ndim == 1 else None)
        denoise_signal.transient_index = index
        if extra is not None:
            base = np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, z.shape)
            noise = base + extra
    elif cfg.transient_module and cfg.transient_mode == "gated":
        if f0_voiced is None:
            f0_voiced = first_pass_f0(x, sample_rate, times, cfg)
        noise = gated_transient_noise(noise, z, f0_voiced[1], sample_rate, n_fft, cfg)
    elif cfg.transient_module:
        noise = transient_noise(noise, z, cfg)
    band = band_floor_profile(z.shape[0], sample_rate, n_fft, cfg) if cfg.band_floor else None
    if cfg.snr_adaptive and band is not None:
        full = complete_frames(times, len(x), sample_rate, n_fft, z.shape[1])
        snr_hat = file_snr_features(z[:, full], noise if noise.ndim == 1 else noise[:, full],
                                    sample_rate, n_fft, cfg.band_cutoff_hz)["snr_file_db"]
        offset_db = snr_adaptive_offset_db(snr_hat, cfg)
        if offset_db:
            above = np.arange(z.shape[0]) * sample_rate / n_fft >= cfg.band_cutoff_hz
            band = (np.where(above, band[0] * 10 ** (offset_db / 20), band[0]), band[1])
    if cfg.gain_rule in ("lsa", "gg") and cfg.prior_snr != "dd":
        enhanced = lsa_with_prior(z, noise, sample_rate, hop / sample_rate, cfg,
                                  gain_floor=None if band is None else band[0])
    elif cfg.gain_rule == "lsa" and cfg.omlsa:
        enhanced = omlsa_decision_directed(z, noise, sample_rate, n_fft, cfg)
    elif cfg.gain_rule == "lsa":
        enhanced = lsa_decision_directed(z, noise, cfg.lsa_smoothing, cfg.lsa_gain_floor)
    else:
        enhanced = wiener_decision_directed(z, noise, cfg.wiener_smoothing, cfg.wiener_gain_floor)
    if cfg.masking:
        enhanced = apply_masking_floor(x, sample_rate, z, enhanced, noise, band, times, n_fft, hop,
                                       cfg)
    if cfg.postfilter:
        magnitude = np.abs(z)
        gain = np.abs(enhanced) / np.maximum(magnitude, 1e-30)
        gain, _, _ = musical_noise_postfilter(gain, magnitude ** 2, cfg.postfilter_zeta_thr,
                                              cfg.postfilter_psi)
        enhanced = gain * z
    if cfg.wind_module:
        low = np.arange(enhanced.shape[0]) * sample_rate / n_fft < cfg.wind_cutoff_hz
        floor = cfg.wiener_gain_floor if cfg.gain_rule == "wiener" else cfg.lsa_gain_floor
        cap = floor * np.abs(z[low])
        current = np.abs(enhanced[low])
        enhanced[low] *= np.minimum(1.0, cap / np.maximum(current, 1e-30))
    if band is not None:
        cap = band[1][:, None] * np.abs(z)
        current = np.abs(enhanced)
        limit = np.isfinite(band[1])
        enhanced[limit] *= np.minimum(1.0, cap[limit] / np.maximum(current[limit], 1e-30))
    if cfg.phase_reconstruction:
        first_pass = istft(enhanced, len(x), sample_rate, n_fft, hop)
        centers = np.rint(times * sample_rate).astype(int)
        f0, voiced = yin_f0(first_pass, sample_rate, centers, cfg)
        # zero-padded boundary frames have a different effective window: leave them as is
        voiced &= (centers >= n_fft // 2) & (centers + n_fft // 2 <= len(x))
        if cfg.phase_smooth:
            f0, voiced = smooth_track(f0, voiced, cfg)
        enhanced = reconstruct_phase(enhanced, f0, voiced, sample_rate, n_fft, hop, cfg)
    return istft(enhanced, len(x), sample_rate, n_fft, hop)


def denoise_file(source, target, cfg=CONFIG):
    """Denoise a WAV file, keeping sample rate, channel count, length and subtype."""
    info = sf.info(str(source))
    audio, sample_rate = sf.read(str(source), dtype="float64", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError("Empty audio file.")
    out = np.column_stack([denoise_signal(audio[:, ch], sample_rate, cfg)
                           for ch in range(audio.shape[1])])

    subtype = info.subtype
    peak = float(np.abs(out).max()) if out.size else 0.0
    if subtype.startswith("PCM") and peak > cfg.pcm_peak_limit:
        # Integer formats would clip; scale the whole file uniformly instead.
        out *= cfg.pcm_peak_limit / peak

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    sf.write(str(partial), out, sample_rate, subtype=subtype, format="WAV")
    partial.replace(target)
    return {"sample_rate": sample_rate, "channels": audio.shape[1],
            "frames": audio.shape[0], "subtype": subtype}


def find_wavs(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".wav")


def denoise_folder(input_dir, output_dir, cfg=CONFIG, log=print):
    """Process every WAV under input_dir into output_dir with the same relative path.

    If a file cannot be processed, the input is copied unchanged so that every input
    still has an output, and the failure is reported.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("The output folder must be outside the input folder.")
    files = find_wavs(input_dir)
    failures = []
    for index, source in enumerate(files, start=1):
        target = output_dir / source.relative_to(input_dir)
        try:
            denoise_file(source, target, cfg)
        except Exception as error:  # keep going; never leave an input without output
            failures.append((source, error))
            log(f"[WARN] {source.name}: {type(error).__name__}: {error} -> copied unchanged")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            except OSError as copy_error:
                log(f"[ERROR] {source.name}: could not write output: {copy_error}")
        if index % 50 == 0 or index == len(files):
            log(f"{index}/{len(files)} files")
    return files, failures
