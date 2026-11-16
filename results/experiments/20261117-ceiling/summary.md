# Upper bounds of the current framework per noise type (Train, analysis only; clean used)

`tools/oracle_bounds.py`; per-type means in `oracle_train_by_type.csv`, per SNR in `oracle_train_by_snr.csv`.

| estimate | Train mean | types >= 2.5 (of 23) |
| --- | ---: | ---: |
| current denoiser (r02-base-s3) | 1.954 | 3 |
| current pipeline with the TRUE noise periodogram per frame (perfect noise estimation) | 2.378 | 10 |
| ideal ratio mask on the noisy STFT (perfect per-cell SNR) | 3.724 | 23 |
| clean magnitude + noisy phase | 3.881 | 23 |

With perfect noise estimation, 13 types stay below 2.5 (broadband 1.77, hf_channel 1.78, aircraft_cockpit 1.83,
clap 2.06, destroyer_engine 2.09, destroyer_operations 2.13, shower 2.15, tooth_brushing 2.18, door_moving 2.27,
alarm_siren 2.28, water 2.37, wind 2.42, machine 2.48); 0 dB files 1.59. The remaining gap to the ideal mask is the
per-cell speech/noise decision (a priori SNR) of the gain rule, which is what learned models estimate.
