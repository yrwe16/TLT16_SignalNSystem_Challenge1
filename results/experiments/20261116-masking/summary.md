# R2-8: auditory-masking gain floor (masking) on r02-base-s3

Decision: **on hold (team decision needed)**. Train above the base (+0.0005), Validation 0.00003 below it.
Defaults unchanged (masking=False).

## Implementation (masking.py; read in the original: Virag, IEEE TSAP 7(2):126-137, 1999, DOI 10.1109/89.748118)
- Masking threshold T from the first-pass output (current pipeline, re-analysed with the same STFT): Zwicker critical
  bands (22 at 16 kHz, Sec. III-A; edges not checked in Zwicker & Fastl), Schroeder spreading function (closed form as
  used by Johnston, matches Fig. 3a; original not checked), offset O(k) read from Fig. 3b (approximate), renormalised
  by the spreading gain, shared equally among the band's bins; no absolute threshold (playback level unknown).
- Floor rule (our choice): in speech frames the LSA floor rises to sqrt(T / N) (largest gain with residual noise at the
  threshold; Sec. III-B principle), at most `masking_range_db` above the base floor. Virag's eqs. 8-9 (alpha/beta
  interpolated between the frame's min and max of T) are not used: a threshold offset would cancel in them.
- Speech frames (our choice): frame posterior SNR sum|Y|^2 / sum N >= 10 dB. First tried YIN-voiced & zeta >= 0.4:
  only 6.4 % of clean-labelled speech frames, 2.4 % of pause frames (tonal noise called voiced).
- Bug found and fixed before the grid: |G Y| / |Y| rounding made the base floor look "raised" in every frame.

## Speech-frame check (Train, clean used only for labels; speech = within 25 dB of the loudest frame, pause = 40 dB below)
Flagged: 42 % of speech frames, 1.1 % of pause frames. Gains actually raised: 36 % of speech frames and 179 pause
frames (0.93 %, 38 files; clap, animal, door, machine_gun: transients above the stationary noise estimate).
The rule "never raise the gain in frames without speech" is therefore not met exactly.

## Train grid (base 1.95372)
| offset | range 3 dB | 6 dB |
| --- | ---: | ---: |
| 0 dB | 1.95401 | **1.95421** |
| -6 dB | 1.95380 | 1.95386 |

Best: +0.0005 [0.0004, 0.0006], improved 69 %. README conditions: 1 FAIL, CI PASS, types PASS, SNR bins PASS.
SNR bins 0/5/10/15 dB: +0.0000 / +0.0000 / +0.0004 / +0.0010 (the target, 0-10 dB, barely moves).
Stationary noise: aircraft_cockpit +0.0000 (111), broadband +0.0002 (73), machine +0.0004 (264).
Worst files: wind 10 dB -0.015, destroyer_engine 15 dB -0.015, military_vehicle 10 dB -0.013,
destroyer_operations 15 dB -0.013, aircraft_cockpit 10 dB -0.012.

## Validation (one candidate)
1.89457 vs 1.89460: -0.00003 [-0.0010, +0.0004], improved 66 %. Worst: wind 15 dB -0.041, destroyer_engine 10 dB -0.014.

## Band floor off (parallel): 1.7820 (base-s3 off 1.7835). Submission threshold 1.884: candidate 1.9542 passes.
Time 0.67 s/file Train (base 0.28 s; one extra full pipeline pass).
