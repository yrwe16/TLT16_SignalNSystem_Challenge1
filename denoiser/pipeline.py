"""Signal-, file- and folder-level noise reduction."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import CONFIG
from .gains import lsa_decision_directed, wiener_decision_directed
from .noise_estimation import lowest_energy_frames, tracked_noise
from .stft import as_waveform, istft, stft


def denoise_signal(x, sample_rate, cfg=CONFIG):
    """Denoise one mono channel. The output has exactly the input length."""
    x = as_waveform(x)
    if not np.any(x):
        return x.copy()
    n_fft, hop = cfg.stft_sizes(sample_rate)
    _, times, z = stft(x, sample_rate, n_fft, hop)
    if cfg.noise_tracker == "spp":
        noise = tracked_noise(z, times, len(x), sample_rate, n_fft, hop, cfg)
    else:
        noise = lowest_energy_frames(z, times, len(x), sample_rate, n_fft, cfg.noise_fraction)
    if cfg.gain_rule == "lsa":
        enhanced = lsa_decision_directed(z, noise, cfg.lsa_smoothing, cfg.lsa_gain_floor)
    else:
        enhanced = wiener_decision_directed(z, noise, cfg.wiener_smoothing, cfg.wiener_gain_floor)
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
