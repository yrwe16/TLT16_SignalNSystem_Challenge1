# R2-12 (version B): noise sampled between f0 harmonics (harmonic_noise) on r02-base-s4

Decision: **not adopted** (stopped before the Train grid; team moved on to version C). Defaults unchanged.

## Literature
Ealey, Kelleher & Pearce, "Harmonic tunnelling: tracking non-stationary noises during speech", Eurospeech 2001,
pp. 437-440, DOI 10.21437/Eurospeech.2001-117 (read in the original, ISCA open archive). Taken from it: interpolation
across a harmonic peak between the means of <= 4 tunnel values per side (Sec. 2.4.1); running average with 25 % new
frame (Sec. 3). Not taken: the structural peak classification (Sec. 2.3, E1-E5) and the tunnel-edge rule (Fig. 5; N %
not given). The prompt's f0-multiple regions are close to the "comb" sampling the paper says gives distorted
estimates (Sec. 5); marked as our simplification.

## Check before PESQ (Train, voiced frames of the first pass; clean used only here; width +-2 bins, no bias)
| | current estimate | tunnel estimate |
| --- | ---: | ---: |
| mean abs error (4 bands) | 6.0 dB | 14.1 dB |
| mean bias | +3.5 dB | +9.1 dB |
Worse for every type with >= 10 files (e.g. machine 4.3 -> 13.1, wind 4.0 -> 15.1, traffic 4.3 -> 13.4, cockpit
1.6 -> 6.5 dB). Leakage (tunnels on clean speech alone, relative to the true noise): -5.3 dB on average, +0.6 dB at
15 dB SNR. YIN marks only 12 % of frames voiced; on noise alone it marks phone_dialing 94 %, click 48 %, bell 47 %,
alarm_siren 44 % voiced. Likely causes: the 32 ms window main lobe (+-2 bins) is about the harmonic spacing of male
voices (100 Hz = 3.2 bins), plus aperiodic speech energy and f0 errors. A bias grid of {0, -3 dB} cannot remove +9 dB.
