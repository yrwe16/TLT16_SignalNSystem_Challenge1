# R2-15: further exploration on r02-base-s5 (per-band floors, noise factors, shape x level noise model)

Decision: **per-band floors + high-band noise factor adopted** (team rule 2026-11-16: Train and Validation both above
the base); shape x level noise model not adopted. New base `20261118-r02-base-s6`. Listening check pending
(`listening/r02-step15_base_vs_bandfloor`, fixed list, 21 files). All values our choice (Train grid only).

## Subset grids (350 stratified Train files; base 1.99252)
- Per-band LSA floors (dB) 325 Hz-1 kHz {-9, -12, -15} x 1-4 kHz {-12, -15, -18, -21} x 4-8 kHz {-12, -15, -20, -25}
  (two rounds; the first optimum lay on the grid edge): best -12 / -21 / -20 1.99907, -12 / -18 / -20 1.99886,
  -12 / -18 / -15 1.99868 (plateau around mid -12, mid-high -18..-21, high -20).
- Noise factors (applied to the stationary estimate) 325 Hz-4 kHz {0.8, 1, 1.25} x above 4 kHz {0.8, 1, 1.25}:
  best 1.0 / 1.25 1.99498.
- Shape x level noise model (shape_gain_noise; water idea from the report): band from {2, 4} kHz x quantile {0.25, 0.5}
  x {max, replace}: 1.89283-1.91191, about -0.08 (overestimation in all files). Not adopted.

## Full Train (base s5 1.95568)
| candidate | Train |
| --- | ---: |
| floors -12/-18/-20 + noise x1.25 above 4 kHz | **1.96310** |
| floors -12/-21/-20 | 1.96139 |
| floors -12/-18/-20 | 1.96044 |
| floors -12/-18/-15 | 1.96012 |
| noise x1.25 above 4 kHz | 1.95832 |

## Chosen: floors -12/-18/-20 dB + noise_beta_high 1.25
- Train +0.0074 [0.0053, 0.0096], improved 59 %; README conditions 1 FAIL (+0.01), CI PASS, types PASS, SNR bins PASS.
- SNR bins 0/5/10/15 dB: +0.006 / +0.007 / +0.007 / +0.009.
- Gains: broadband +0.045 (73), destroyer_engine +0.042 (36), aircraft_cockpit +0.032 (111), destroyer_operations
  +0.025 (36), hf_channel +0.021 (37), machine +0.010 (264). Losses: wind -0.019 (202), vehicle_interior -0.014 (19),
  phone_dialing -0.006 (43), factory -0.006 (38).
- Worst files: factory 15 dB -0.16, alarm_siren 0 dB -0.15, wind 15 dB -0.15, alarm_siren 15 dB -0.13, wind 15 dB -0.13.
- Validation: 1.9011 vs 1.8989, +0.0022 [-0.0034, +0.0085]; SNR bins 0/5/10/15 dB +0.008/-0.004/+0.011/-0.003.
- Time unchanged (0.77 s/file).
