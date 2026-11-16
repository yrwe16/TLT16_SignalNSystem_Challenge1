# R2-7: narrowband (tonal) module revised (tonal_mode="r02") on r02-base-s2

Decision: **adopted** under the team rule of 2026-11-16 (Train and Validation means both above the base).
New base: `results/experiments/20261116-r02-base-s3/`. Listening check pending (`listening/r02-step7_base_vs_tonal`, 10 files).

## Implementation (tonal_module=True, tonal_mode="r02"; all values our choice)
- YIN f0 per frame from a first pass of the pipeline without the module (phase settings, offline smooth_track).
- In voiced frames peaks within +-1 bin of multiples of f0 are dropped; tracks follow peaks frame by frame within
  +-2 kHz/s (at least 1 bin per frame), last >= 300 ms, and lie in unvoiced/silent frames for >= tonal_unvoiced_min.
- The observed power on the track and on confirmed peaks at its integer multiples (+-2 bins) is added to the noise PSD
  (no notch, no frequency smoothing of this component).
- Lower search frequency 400 Hz (team decision after the first check, see below).
- Background: Gul, Ariyaeeinia & Dewhirst, Eurospeech 2003 (read in the original; alarm types, alarms as sinusoids
  independent of speech; their LP mean-vector filter, eqs. 1-4, is not used).

## First check: false detection on clean Train speech (share of time-frequency bins)
| tracker | share | files with any detection |
| --- | ---: | ---: |
| 6a | 0.59 % | 64 % |
| r02, lower limit 100 Hz, peak 10 / 8 dB | 0.60 % / 1.46 % | 46 % / 58 % |
| r02, lower limit 400 Hz, peak 10 / 8 dB | 0.01 % / 0.05 % | 3 % / 10 % |

Cause at 100 Hz (150 clean files): false tracks at 125-250 Hz (median 156 Hz, speech f0) in loud frames that YIN called
unvoiced (98.7 %), and 63 % of the false bins came from their integer multiples. On noise-only signals YIN calls 44 % of
alarm_siren frames and 94 % of phone_dialing frames voiced. Siren tracks: median 2.3 kHz, 4 % below 400 Hz (bell 0 %,
phone_dialing 56 %, at 250 Hz). Precision on alarm_siren (masked bins where noise > speech): 56 % -> 73 %.

## Train grid (base 1.9485)
| peak dB | unvoiced >= 0.3 | >= 0.5 |
| --- | ---: | ---: |
| 8 | 1.95372 | **1.95372** |
| 10 | 1.95348 | 1.95348 |

Chosen p8 / u0.5 (highest PESQ; u0.3 lower by 2e-6). Train +0.0053 [0.0028, 0.0078], improved 21 %, Wilcoxon p 4e-9.
README conditions: 1 FAIL (+0.01), CI > 0 PASS, types PASS, SNR bins PASS.
SNR bins 0/5/10/15 dB: -0.0021 / +0.0020 / +0.0069 / +0.0077.
Types (n): alarm_siren +0.023 (187), bell +0.078 (15), phone_dialing -0.009 (43), traffic_vehicle +0.032 (76),
destroyer_engine +0.030 (36), machine -0.002 (264), aircraft_cockpit -0.001 (111), broadband -0.003 (73), water -0.005 (57),
wind -0.001 (202).
Module on (any kept track): target types (siren, bell, phone) 68 %, other types 24 % (mean +0.007 on those 249 files;
machine 88 files -0.005, traffic 48 +0.051, destroyer_engine 36 +0.030, wind 18 -0.015, military_vehicle 9 -0.027).
Worst files: animal 15 dB -0.33, machine 10 dB -0.28, alarm_siren 15 dB -0.28, machine 15 dB -0.22, military_vehicle 15 dB -0.21.

## Validation (one candidate)
1.8946 vs 1.8868: +0.0078 [0.0032, 0.0141], Wilcoxon p 0.006, 52 of 164 files changed. alarm_siren -0.005 (23),
traffic_vehicle +0.079 (9), machine +0.006 (34). The 5 worst files are all alarm_siren (-0.09 to -0.05).

## Band floor off (parallel report)
Train with band_floor=False: 1.7835 (base-s2 with band_floor off: 1.7772). band_floor gain 0.1713 -> 0.1702.
Submission threshold 1.884: passed (1.9537).

Time 0.28 s/file Train (base 0.07 s; the YIN first pass doubles the pipeline).
