"""All algorithm parameters live here so the submitted code is self-contained."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # STFT: a 32 ms Hann window with a 75 % overlap.
    # At 16 kHz this is n_fft=512 and hop=128, identical to the baseline notebook.
    frame_seconds: float = 0.032
    overlap_ratio: float = 0.75

    # Noise PSD: mean power of the lowest-energy complete frames, smoothed over 3 bins.
    noise_fraction: float = 0.20

    # Wiener filter with decision-directed a priori SNR (Ephraim & Malah, 1984).
    wiener_smoothing: float = 0.98
    wiener_gain_floor: float = 0.05

    # Output safety for integer PCM files (PESQ aligns levels, so a uniform gain is harmless).
    pcm_peak_limit: float = 0.999

    def __post_init__(self):
        if not 0 < self.frame_seconds <= 0.25:
            raise ValueError("frame_seconds must be in (0, 0.25].")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1).")
        if not 0 < self.noise_fraction <= 1:
            raise ValueError("noise_fraction must be in (0, 1].")
        if not 0 <= self.wiener_smoothing < 1:
            raise ValueError("wiener_smoothing must be in [0, 1).")
        if not 0 <= self.wiener_gain_floor <= 1:
            raise ValueError("wiener_gain_floor must be in [0, 1].")
        if not 0 < self.pcm_peak_limit <= 1:
            raise ValueError("pcm_peak_limit must be in (0, 1].")

    def stft_sizes(self, sample_rate):
        """Window and hop in samples for a sample rate (512/128 at 16 kHz)."""
        n_fft = max(4, int(round(self.frame_seconds * sample_rate)))
        n_fft += n_fft % 2
        hop = max(1, int(round(n_fft * (1 - self.overlap_ratio))))
        return n_fft, hop


CONFIG = Config()
