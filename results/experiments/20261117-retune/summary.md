# R2-10: joint re-tuning with all adopted steps on (r02-base-s3) and submission candidates

No TA answer on whether parameter search counts as ML: coordinate search only (2-3 parameters at a time, the rest fixed).
Screening on a stratified Train subset (`train_subset.csv`: up to 6 files per noise_category x target_snr_db cell,
seed 0, 350 files, 69 cells; `tools/make_subset.py`), top 5 re-evaluated on full Train; a change is kept only if it
beats the current value by at least the Train MDE (0.012). Result: **no parameter changes**; final = r02-base-s3.

## Coordinate 1: LSA floor (all bands) x DD coefficient a (subset, current -15 dB / 0.95 = 1.97455)
| floor | a 0.92 | 0.95 | 0.98 |
| --- | ---: | ---: | ---: |
| -12 dB | 1.96654 | 1.97153 | 1.97128 |
| -15 dB | 1.96615 | **1.97455** | 1.97151 |
| -18 dB | 1.95763 | 1.96755 | 1.95859 |
| -22 dB | 1.94841 | 1.95666 | 1.94106 |
Full Train (current 1.95372): -15/0.98 1.95696 (+0.0032 < MDE), -18/0.95 1.94912, -12/0.98 1.94741, -12/0.95 1.94374.

## Coordinate 2: low band (step 4 adaptive not adopted -> fixed options), full Train / Validation
| option | Train | Validation | Train >= 1.884 (round-01 Train 1.872 + 0.012) |
| --- | ---: | ---: | --- |
| band_floor 325 Hz -25 dB (current) | 1.9537 | 1.8946 | yes |
| fixed low cut 325 Hz -15 dB (round-01 style) | 1.8868 | 1.8203 | yes (barely) |
| off | 1.7835 | 1.7491 | no |

## Coordinate 3: SPP xi_H1 x frame length (subset, current 10 dB / 32 ms = 1.97455)
| xi_H1 | 32 ms | 64 ms |
| --- | ---: | ---: |
| 6 dB | 1.97693 | 1.89087 |
| 10 dB | **1.97455** | 1.89257 |
| 15 dB | 1.96296 | 1.89147 |
Full Train: 6/32 1.95163, 15/32 1.95035, 10/64 1.88813, 15/64 1.89059 (all below the current value).

Coordinate 1 optimum unchanged -> no second pass. Final candidate = current defaults; its Validation (1.8946) was
measured once in step 7 and is not re-run.

Regression: with every switch off the output is byte-identical to the stage-0 Wiener baseline (tests/test_regression).
Listening set: `listening/r02-final_round01_vs_r02` (round-01 submission vs final, fixed listening list, 21 files).
