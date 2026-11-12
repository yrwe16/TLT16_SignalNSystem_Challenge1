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

    # Defaults = round-01 submission: the stack adopted over roadmap stages 1-6
    # (results/experiments.csv). Setting gain_rule="wiener", prior_snr="dd",
    # phase_reconstruction=False and wind_module=False gives the stage-0 baseline Wiener.
    # Gain rule (roadmap stage 1): "wiener" = baseline (stage 0), "lsa" = MMSE log-spectral
    # amplitude estimator (Ephraim & Malah, IEEE TASSP 33(2):443-445, 1985).
    # Both use the decision-directed a priori SNR; each rule keeps its own parameters.
    gain_rule: str = "lsa"

    # Wiener filter with decision-directed a priori SNR (Ephraim & Malah, 1984).
    wiener_smoothing: float = 0.98
    wiener_gain_floor: float = 0.05

    # MMSE-LSA: G = xi/(1+xi) * exp(E1(v)/2), v = xi*gamma/(1+xi), then G = max(G, floor).
    # No upper clip (as in the paper, logmmse and VOICEBOX v_ssubmmse).
    # Chosen on Train by PESQ-WB: floor {-10,-12,-15,-20,-25} dB x a {0.92,0.95,0.98}.
    lsa_smoothing: float = 0.95
    lsa_gain_floor: float = 0.1778279410038923  # 10**(-15/20), -15 dB

    # Noise PSD (roadmap stage 2): "fixed" = one estimate per file (lowest-energy frames,
    # round-01), "spp" = unbiased MMSE speech-presence-probability tracking per frame
    # (Gerkmann & Hendriks, IEEE TASLP 20(4):1383-1393, 2012). Constants follow VOICEBOX
    # v_estnoiseg defaults (read only, LGPL; not copied). Time constants are converted to
    # per-frame factors with the actual hop (0.8 / 0.9 at a 16 ms hop, as in the paper).
    noise_tracker: str = "fixed"
    spp_snr_db: float = 15.0             # xi_H1, speech-present a priori SNR (asnr)
    spp_prior: float = 0.5               # prior speech presence probability (pspri)
    spp_noise_time: float = 0.0717       # noise PSD smoothing time constant, s (tax)
    spp_prob_time: float = 0.152         # SPP smoothing time constant, s (tap)
    spp_prob_threshold: float = 0.99     # stagnation threshold on the smoothed SPP (psthr)
    spp_prob_safety: float = 0.01        # SPP capped at 1 - safety above threshold (pnsaf)
    # Initial noise PSD: "first" = VOICEBOX (init_scale x mean periodogram of the first
    # 1 + init_time/hop frames), "lowenergy" = the round-01 fixed estimate,
    # "percentile" = per-bin low percentile of the whole clip / -ln(1-p) (offline variant).
    spp_init: str = "first"
    spp_init_time: float = 0.064         # tavini
    spp_init_scale: float = 0.5          # psini (also the initial smoothed SPP)
    spp_init_percentile: float = 10.0
    # "forward" = causal tracking as published; "bidirectional" = mean of a forward and a
    # time-reversed pass (offline variant, no literature support).
    spp_direction: str = "forward"

    # OM-LSA (roadmap stage 3, applies with gain_rule="lsa"): G = G_H1^p * G_min^(1-p)
    # (Cohen & Berdugo, Signal Processing 81(11):2403-2418, 2001). The DD a priori SNR uses
    # the previous G_H1^2*gamma and is floored at G_min^2, as in the BSD-3 reference
    # chenwj1989/python-speech-enhancement pns/suppression_gain.py (f38fde7).
    # omlsa_spp: "cohen" = a priori speech absence q from smoothed xi (local / global /
    # frame terms, Cohen 2001), p = 1/(1 + q/(1-q)*(1+xi)*exp(-v));
    # "posterior" = stage-2 SPP with a fixed xi_H1 (spp_snr_db, spp_prior).
    omlsa: bool = False
    omlsa_spp: str = "cohen"
    omlsa_gain_floor: float = 0.1778279410038923  # G_min; stage-1 floor (-15 dB)
    # omlsa_dd: "h1" = Cohen (previous G_H1^2*gamma, xi >= G_min^2);
    # "output" = stage-1 DD (previous output power, no xi floor), variant.
    omlsa_dd: str = "h1"

    # A priori SNR (roadmap stage 4, applies with gain_rule="lsa" and omlsa=False):
    # "dd" = decision-directed (stage 1); "cepstral" = selective cepstro-temporal smoothing
    # (Breithaupt, Gerkmann & Martin, ICASSP 2008, pp. 4897-4900, constants from Table 1,
    # read in the original); offline variants: "dd_bidir" = mean of forward and backward DD,
    # "ml_2d" = centred 2-D moving average of the ML SNR.
    prior_snr: str = "dd_bidir"  # stage 4 (adopted variant)
    # Table 1 is for a 16 ms frame shift (M = 512, shift M/2 at 16 kHz); False converts the
    # per-frame constants (alpha_const, alpha_pitch, beta) to our hop with equal time constants.
    ceps_per_frame_constants: bool = False
    ceps_kappa: float = 0.5               # bias correction in units of Euler's constant (Table: 0.5)
    ceps_xi_min_db: float = -25.0         # xi_min (Table 1)
    ceps_threshold: float = 0.2           # Lambda_thr (Table 1)
    # Normalisation of the 8-tap Hamming window in eq. (8) is not stated in the paper.
    # Normalised (sum 1) marks ~27 % of noisy Validation frames voiced; unnormalised ~92 %
    # (more than the 78 % on clean speech), so the normalised window is the default.
    ceps_window_normalized: bool = True
    ml2d_bins: int = 3
    ml2d_frames: int = 5

    # Noise-type module (roadmap stage 6, weakest type alarm_siren; also bells, hum):
    # long narrowband peak tracks are added to the noise PSD (tonal_weight x observed power
    # on the track bins). Tracking after McAulay & Quatieri, IEEE TASSP 34(4), 1986; all
    # constants are our own choices.
    tonal_module: bool = False
    tonal_peak_db: float = 10.0          # peak must exceed the local spectral median by this
    tonal_median_bins: int = 15
    tonal_max_jump_bins: int = 2         # matching interval per frame (8 ms)
    tonal_max_gap_frames: int = 2
    tonal_min_seconds: float = 0.4
    tonal_halfwidth_bins: int = 2        # Hann main lobe half width at n_fft
    tonal_weight: float = 1.0
    # "observed" = all power on track bins is noise; "track_median" = the median peak power
    # along the track (so speech overlapping the tone in some frames is not removed), variant.
    tonal_level: str = "observed"
    tonal_fmin: float = 100.0
    tonal_fmax: float = 7800.0

    # Phase reconstruction in voiced speech (roadmap stage 5; Krawczyk & Gerkmann, IEEE/ACM
    # TASLP 22(12):1931-1940, 2014). "combi" = along time and frequency (Sec. IV-B),
    # "frequency" = across frequency only (eq. 13). Harmonics up to 4 kHz as in the paper.
    # f0 by YIN on the enhanced signal (the paper's PEFAC uses a trained voicing model);
    # YIN threshold 0.1 and the -30 dB energy gate are our choices (not from the paper).
    phase_reconstruction: bool = True  # stage 5 (adopted variant below)
    phase_mode: str = "frequency"
    phase_max_hz: float = 600.0
    phase_f0_min: float = 60.0
    phase_f0_max: float = 400.0
    phase_yin_threshold: float = 0.1
    phase_energy_db: float = -30.0
    # Offline variants: smooth the f0 track non-causally (fill gaps <= fill_gap frames, drop
    # voiced runs shorter than min_run frames, 5-frame median), and crossfade the phase
    # change over fade_frames at voicing boundaries.
    phase_smooth: bool = True
    phase_fill_gap_frames: int = 2
    phase_min_run_frames: int = 5
    phase_fade_frames: int = 3

    # Further stage-6 noise-type modules (each its own switch, all off by default):
    # RPCA low-rank noise PSD for machine / engine / cockpit noise.
    rpca_module: bool = False
    rpca_lambda_scale: float = 1.0       # lambda = scale / sqrt(max(n_bins, n_frames))
    rpca_combine: str = "replace"        # "replace" or "max" with the background estimate
    rpca_max_iter: int = 100
    # Wind: extra attenuation below wind_cutoff_hz (high-pass by capping the gain at the
    # LSA floor), as the skill's first step for wind noise.
    wind_module: bool = True  # stage 6b
    wind_cutoff_hz: float = 325.0
    # Transients (clap, click, door, footsteps): HPSS percussive power added to the noise PSD.
    transient_module: bool = False
    transient_margin: float = 2.0
    transient_frame_share: float = 0.3

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
        if self.noise_tracker not in ("fixed", "spp"):
            raise ValueError("noise_tracker must be 'fixed' or 'spp'.")
        if self.spp_init not in ("first", "lowenergy", "percentile"):
            raise ValueError("spp_init must be 'first', 'lowenergy' or 'percentile'.")
        if self.spp_direction not in ("forward", "bidirectional"):
            raise ValueError("spp_direction must be 'forward' or 'bidirectional'.")
        if not 0 < self.spp_prior < 1 or not 0 < self.spp_init_scale <= 1:
            raise ValueError("spp_prior must be in (0, 1) and spp_init_scale in (0, 1].")
        if min(self.spp_noise_time, self.spp_prob_time) <= 0 or self.spp_init_time < 0:
            raise ValueError("SPP time constants must be positive.")
        if not 0 < self.spp_prob_threshold <= 1 or not 0 <= self.spp_prob_safety < 1:
            raise ValueError("spp_prob_threshold must be in (0, 1], spp_prob_safety in [0, 1).")
        if not 0 < self.spp_init_percentile < 100:
            raise ValueError("spp_init_percentile must be in (0, 100).")
        if self.omlsa_spp not in ("cohen", "posterior"):
            raise ValueError("omlsa_spp must be 'cohen' or 'posterior'.")
        if self.omlsa_dd not in ("h1", "output"):
            raise ValueError("omlsa_dd must be 'h1' or 'output'.")
        if not 0 < self.omlsa_gain_floor <= 1:
            raise ValueError("omlsa_gain_floor must be in (0, 1].")
        if self.omlsa and self.gain_rule != "lsa":
            raise ValueError("omlsa requires gain_rule='lsa'.")
        if self.prior_snr not in ("dd", "cepstral", "dd_bidir", "ml_2d"):
            raise ValueError("prior_snr must be 'dd', 'cepstral', 'dd_bidir' or 'ml_2d'.")
        if self.prior_snr != "dd" and (self.gain_rule != "lsa" or self.omlsa):
            raise ValueError("prior_snr other than 'dd' requires gain_rule='lsa', omlsa=False.")
        if self.ceps_kappa < 0 or self.ml2d_bins < 1 or self.ml2d_frames < 1:
            raise ValueError("ceps_kappa must be >= 0 and ml2d sizes >= 1.")
        if (self.tonal_min_seconds <= 0 or self.tonal_weight < 0 or self.tonal_median_bins < 3
                or self.tonal_max_jump_bins < 0 or self.tonal_max_gap_frames < 0
                or self.tonal_halfwidth_bins < 0 or not 0 <= self.tonal_fmin < self.tonal_fmax):
            raise ValueError("Invalid tonal module parameters.")
        if self.tonal_level not in ("observed", "track_median"):
            raise ValueError("tonal_level must be 'observed' or 'track_median'.")
        if self.phase_mode not in ("combi", "frequency"):
            raise ValueError("phase_mode must be 'combi' or 'frequency'.")
        if not 0 < self.phase_f0_min < self.phase_f0_max or self.phase_max_hz <= 0:
            raise ValueError("Invalid phase f0 range or phase_max_hz.")
        if (self.phase_yin_threshold <= 0 or self.phase_fade_frames < 0
                or self.phase_fill_gap_frames < 0 or self.phase_min_run_frames < 1):
            raise ValueError("Invalid phase reconstruction parameters.")
        if (self.rpca_combine not in ("replace", "max") or self.rpca_lambda_scale <= 0
                or self.rpca_max_iter < 1 or self.wind_cutoff_hz <= 0
                or self.transient_margin <= 0 or not 0 <= self.transient_frame_share <= 1):
            raise ValueError("Invalid stage-6 module parameters.")
        if not 0 < self.pcm_peak_limit <= 1:
            raise ValueError("pcm_peak_limit must be in (0, 1].")

    def stft_sizes(self, sample_rate):
        """Window and hop in samples for a sample rate (512/128 at 16 kHz)."""
        n_fft = max(4, int(round(self.frame_seconds * sample_rate)))
        n_fft += n_fft % 2
        hop = max(1, int(round(n_fft * (1 - self.overlap_ratio))))
        return n_fft, hop


CONFIG = Config()
