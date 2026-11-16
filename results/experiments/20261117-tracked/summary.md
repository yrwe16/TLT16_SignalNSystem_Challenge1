# R2-11 (version A): speech-protected time-varying noise tracking on r02-base-s3

Decision: **adopted** under the team rule of 2026-11-16 (Train and Validation means both above the base); evidence
is weak (Train +0.0002, CI includes 0). New base `20261117-r02-base-s4`. Listening check pending
(`listening/r02-step11_base_vs_tracked`, fixed list, 21 files).

## Implementation (noise_estimator="tracked_protected"; all values our choice)
spp_iter estimate updated frame by frame with the SPP-weighted expected periodogram (Gerkmann & Hendriks 2012 update as
in VOICEBOX v_estnoiseg; original not checked), not updated in protected cells (YIN f0 harmonics +-1 bin in voiced
frames of a first pass with spp_iter; frames with posterior SNR >= threshold; the IMCRA idea of Cohen 2003, original
not checked); forward and backward runs, cell-wise minimum. The prompt asked for a floor at the spp_iter estimate:
with it the tracker changed almost nothing (frame error 6.41 -> 6.43 dB; the fixed estimate is mostly too HIGH frame
by frame, e.g. +4.5 dB in speech frames, animal +12 dB), so "floor on/off" was added as the third grid parameter.

## Noise estimate check (subset, clean used only here; mean abs dB error in 4 bands, frame by frame)
| variant | speech frames | pause frames | leakage on clean speech |
| --- | ---: | ---: | ---: |
| fixed spp_iter | 7.63 | 7.23 | - |
| tracked, min, floor = spp_iter | 7.66 | 7.25 | +0.01 dB |
| tracked, min, no floor | 7.04 | 6.56 | -8.3 dB (stays below the fixed level) |

## Train grid (subset 350; base 1.97455)
No floor: tau 0.1 / 0.3 / 1.0 s with frame SNR 10 dB: 1.99402 / 1.99046 / 1.98318 (6 dB: 1.99193 / 1.98963 / 1.98342);
with floor: 1.97488-1.97786.
Full Train (base 1.95372): tau 0.3 s / 10 dB 1.95389 (+0.0002 [-0.0032, +0.0035]), tau 0.1 s / 10 dB 1.94431,
tau 0.1 s / 6 dB 1.94375. The subset (equal weight per type x SNR cell) over-weights rare time-varying noise types.

## Chosen tau 0.3 s, frame SNR 10 dB, no floor
- Train: +0.0002, README conditions 1 FAIL, CI FAIL, types PASS, SNR bins PASS. SNR 2.658 -> 2.630 dB.
- SNR bins 0/5/10/15 dB: +0.0092 / +0.0005 / -0.0001 / -0.0021.
- Gains: vehicle_interior +0.134 (19), phone_dialing +0.071 (43), military_vehicle +0.039 (37), traffic +0.021 (76),
  destroyer_engine +0.020 (36), animal +0.018 (39), wind +0.010 (202).
- Losses: alarm_siren -0.028 (187), destroyer_operations -0.018 (36), water -0.018 (57), aircraft_cockpit -0.014 (111),
  broadband -0.010 (73), machine -0.009 (264).
- Worst files: alarm_siren 5 dB -0.49, aircraft_cockpit 5 dB -0.40, alarm_siren 5 dB -0.38, alarm_siren 0 dB -0.30,
  alarm_siren 15 dB -0.28.
- Validation: 1.89798 vs 1.89460, +0.0034 [-0.0039, +0.0114]; SNR bins +0.001/+0.004/+0.002/+0.006; worst machine
  10 dB -0.14, alarm_siren 15 dB -0.13.
- Band floor off: 1.7821 (base-s3 off 1.7835). Submission threshold 1.884: passes (1.9539).
- Time 0.77 s/file (base 0.28 s; extra first pass for YIN).
- Recovered share of the perfect-noise-estimate gap (1.954 -> 2.378): about 0 % overall; positive only for
  vehicle_interior, phone_dialing, military_vehicle, traffic.
