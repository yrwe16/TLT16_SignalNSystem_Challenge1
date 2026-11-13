"""Signal-, file- and folder-level noise reduction."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import CONFIG
from .gains import (band_floor_profile, lsa_decision_directed, lsa_with_prior, omlsa_decision_directed,
                    wiener_decision_directed)
from .noise_estimation import (add_tonal_noise, band_frame_noise, lowest_energy_frames,
                               rpca_noise, scale_noise_bands, spp_weighted_noise,
                               tracked_noise, transient_noise)
from .phase import reconstruct_phase, smooth_track, yin_f0
from .stft import as_waveform, istft, stft


def estimate_noise_psd(z, times, length, sample_rate, n_fft, hop, cfg=CONFIG):
    """Background noise PSD from the noisy STFT only (one vector, or one per frame)."""
    if cfg.noise_tracker == "spp":
        return tracked_noise(z, times, length, sample_rate, n_fft, hop, cfg)
    if cfg.noise_estimator == "band_frames":
        noise = band_frame_noise(z, times, length, sample_rate, n_fft, cfg.noise_band_fraction)
    else:
        noise = lowest_energy_frames(z, times, length, sample_rate, n_fft, cfg.noise_fraction)
        if cfg.noise_estimator == "spp_iter":
            noise = spp_weighted_noise(z, times, length, sample_rate, n_fft, noise,
                                       cfg.noise_spp_snr_db, cfg.noise_spp_iterations,
                                       cfg.noise_spp_start_db)
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
    if cfg.tonal_module:
        noise = add_tonal_noise(noise, z, sample_rate, n_fft, hop, cfg)
    if cfg.rpca_module:
        noise = rpca_noise(noise, z, cfg)
    if cfg.transient_module:
        noise = transient_noise(noise, z, cfg)
    band = band_floor_profile(z.shape[0], sample_rate, n_fft, cfg) if cfg.band_floor else None
    if cfg.gain_rule == "lsa" and cfg.prior_snr != "dd":
        enhanced = lsa_with_prior(z, noise, sample_rate, hop / sample_rate, cfg,
                                  gain_floor=None if band is None else band[0])
    elif cfg.gain_rule == "lsa" and cfg.omlsa:
        enhanced = omlsa_decision_directed(z, noise, sample_rate, n_fft, cfg)
    elif cfg.gain_rule == "lsa":
        enhanced = lsa_decision_directed(z, noise, cfg.lsa_smoothing, cfg.lsa_gain_floor)
    else:
        enhanced = wiener_decision_directed(z, noise, cfg.wiener_smoothing, cfg.wiener_gain_floor)
    if cfg.wind_module:
        low = np.arange(enhanced.shape[0]) * sample_rate / n_fft < cfg.wind_cutoff_hz
        floor = cfg.lsa_gain_floor if cfg.gain_rule == "lsa" else cfg.wiener_gain_floor
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
