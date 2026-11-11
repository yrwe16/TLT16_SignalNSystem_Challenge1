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

    # Gain rule (roadmap stage 1): "wiener" = round-01 submission, "lsa" = MMSE log-spectral
    # amplitude estimator (Ephraim & Malah, IEEE TASSP 33(2):443-445, 1985).
    # Both use the decision-directed a priori SNR; each rule keeps its own parameters so
    # gain_rule="wiener" reproduces the round-01 submission byte for byte.
    # round-01 submission uses "wiener"; "lsa" was adopted on 2026-11-10 for a later round
    # (results/experiments.csv).
    gain_rule: str = "wiener"

    # Wiener filter with decision-directed a priori SNR (Ephraim & Malah, 1984).
    wiener_smoothing: float = 0.98
    wiener_gain_floor: float = 0.05

    # MMSE-LSA: G = xi/(1+xi) * exp(E1(v)/2), v = xi*gamma/(1+xi), then G = max(G, floor).
    # No upper clip (as in the paper, logmmse and VOICEBOX v_ssubmmse).
    # Chosen on Train by PESQ-WB: floor {-10,-12,-15,-20,-25} dB x a {0.92,0.95,0.98}.
    lsa_smoothing: float = 0.95
    lsa_gain_floor: float = 0.1778279410038923  # 10**(-15/20), -15 dB

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
        if self.gain_rule not in ("wiener", "lsa"):
            raise ValueError("gain_rule must be 'wiener' or 'lsa'.")
        if not 0 <= self.lsa_smoothing < 1:
            raise ValueError("lsa_smoothing must be in [0, 1).")
        if not 0 <= self.lsa_gain_floor <= 1:
            raise ValueError("lsa_gain_floor must be in [0, 1].")
        if not 0 < self.pcm_peak_limit <= 1:
            raise ValueError("pcm_peak_limit must be in (0, 1].")

    def stft_sizes(self, sample_rate):
        """Window and hop in samples for a sample rate (512/128 at 16 kHz)."""
        n_fft = max(4, int(round(self.frame_seconds * sample_rate)))
        n_fft += n_fft % 2
        hop = max(1, int(round(n_fft * (1 - self.overlap_ratio))))
        return n_fft, hop


CONFIG = Config()
