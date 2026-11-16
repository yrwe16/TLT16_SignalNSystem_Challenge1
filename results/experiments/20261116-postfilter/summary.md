# R2-6: musical-noise postfilter (Esch & Vary, ICASSP 2009) on r02-base-s2

Decision: **not adopted** (no Train candidate reaches +0.01; Validation not run).

## Implementation (`postfilter`, off by default)
- Paper (read in the original, PDF supplied by the team, not committed): eq. (5) zeta = sum|G Y|^2 / sum|Y|^2 per frame,
  eq. (6) threshold, eq. (7) N = 2 round((1 - zeta/zeta_thr) psi) + 1, eqs. (8)-(9) moving average of |G| over frequency.
  Paper settings: zeta_thr 0.4, psi 10, 8 kHz, Wiener / spectral subtraction / MMSE / MAP gains with floor -15 dB.
- Applied after the LSA gain (with its floor) and before the low-band cap.
- Our choices: one-sided bins, reflected edges, round half up; gains limited to 1 before smoothing in smoothed frames
  (5.4 % of LSA gains exceed 1, up to ~27, in bins with tiny |Y|; averaging spread them to neighbours by ~+30 dB);
  neither the frame's mean squared gain nor its output energy sum |G Y|^2 may rise (frame scaled back).
  With the mean-squared-gain limit alone the output energy rose in 52 % of smoothed frames (95th pct +2.6 to +3.1 dB),
  so the energy limit was added, following the stated purpose (no extra residual energy in speech pauses).
- Not used: cepstral gain smoothing (Breithaupt et al. 2007, US 8271271).

## Train grid (base 1.9485)
| zeta_thr | N max 3 | 7 | 15 |
| --- | ---: | ---: | ---: |
| 0.4 (paper) | 1.9489 | 1.9430 | 1.9315 |
| 0.2 (our choice) | 1.9489 | 1.9457 | 1.9371 |

Best (zeta_thr 0.2, psi 1): +0.0004 [-0.0006, +0.0012], improved share 47 %. Checks: train +0.01 FAIL, CI > 0 FAIL,
noise types PASS, SNR bins PASS. SNR bins 0/5/10/15 dB: -0.0029 / -0.0001 / +0.0006 / +0.0015.
Stationary noise: machine +0.0011 (264), aircraft_cockpit +0.0004 (111), broadband_noise -0.0004 (73).
Largest changes: phone_dialing -0.014 (43), destroyer_engine -0.0085 (36), vehicle_interior +0.0078 (19).
Worst files: destroyer_engine 15 dB -0.35, phone_dialing 5 dB -0.12 / -0.09 / -0.07, phone_dialing 0 dB -0.07.
Time 0.096 s/file (base 0.070 s).

## Speech frames (diagnose-train.json; clean used only for labelling, frame within 25 dB of the file's loudest)
zeta median: speech 0.80, pauses 0.10. Best candidate smooths 4.7 % of speech frames and 49 % of pause frames;
largest per-bin gain change in speech frames 12.4 dB (gains limited to 1), mean 0.09 dB; frame output energy change
in speech frames at most -1.95 dB. With zeta_thr 0.4 / N 15: 17.8 % of speech frames, max 22.9 dB, energy up to -5.8 dB.

## Not done
Validation, low-cut-off parallel score and the listening set (no candidate passed Train).
