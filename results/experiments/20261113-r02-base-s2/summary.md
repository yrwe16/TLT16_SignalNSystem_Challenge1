# Baseline statistics: 20261113-r02-base-s2

## Train (1296 files)

- Mean PESQ-WB 1.9485, mean SNR 2.66 dB (noisy: PESQ 1.4704, SNR 9.69 dB)
- SNR-equally-weighted mean: PESQ-WB 1.8295, SNR 2.52 dB
- Paired difference base - R2-1 base (noise_estimator=lowenergy): mean +0.0151, SD_d 0.1243, n 1296, MDE = 2.8 x SD_d / sqrt(n) = 0.0097
- Processing time per file: mean 0.070 s, median 0.072 s, max 0.137 s (16 parallel workers, BLAS threads {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})

### Mean per target SNR

| target_snr_db | files | pesq_noisy | pesq | snr_db |
| --- | --- | --- | --- | --- |
| 0.0 | 117 | 1.1678 | 1.4332 | 1.9962 |
| 5.0 | 298 | 1.2548 | 1.679 | 2.4773 |
| 10.0 | 429 | 1.4042 | 1.9019 | 2.7331 |
| 15.0 | 452 | 1.7538 | 2.3037 | 2.875 |

### Files per noise type x target SNR

| noise_category | 0 dB | 5 dB | 10 dB | 15 dB | total |
| --- | --- | --- | --- | --- | --- |
| aircraft_cockpit | 12 | 37 | 34 | 28 | 111 |
| alarm_siren | 23 | 44 | 58 | 62 | 187 |
| animal | 0 | 0 | 17 | 22 | 39 |
| bell | 0 | 0 | 9 | 6 | 15 |
| broadband_noise | 15 | 12 | 24 | 22 | 73 |
| clap | 0 | 0 | 2 | 1 | 3 |
| click | 0 | 0 | 1 | 2 | 3 |
| destroyer_engine | 7 | 7 | 8 | 14 | 36 |
| destroyer_operations | 4 | 13 | 7 | 12 | 36 |
| door_moving | 0 | 0 | 2 | 1 | 3 |
| factory | 0 | 6 | 14 | 18 | 38 |
| footsteps | 0 | 0 | 1 | 7 | 8 |
| hf_channel | 5 | 12 | 10 | 10 | 37 |
| machine | 7 | 62 | 100 | 95 | 264 |
| machine_gun | 0 | 0 | 1 | 2 | 3 |
| military_vehicle | 0 | 11 | 15 | 11 | 37 |
| phone_dialing | 8 | 12 | 11 | 12 | 43 |
| shower | 0 | 0 | 1 | 2 | 3 |
| tooth_brushing | 0 | 0 | 2 | 1 | 3 |
| traffic_vehicle | 5 | 22 | 22 | 27 | 76 |
| vehicle_interior | 0 | 5 | 6 | 8 | 19 |
| water | 0 | 0 | 19 | 38 | 57 |
| wind | 31 | 55 | 65 | 51 | 202 |
| total | 117 | 298 | 429 | 452 | 1296 |

## Validation (164 files)

- Mean PESQ-WB 1.8868, mean SNR 2.48 dB (noisy: PESQ 1.3947, SNR 9.63 dB)
- SNR-equally-weighted mean: PESQ-WB 1.7608, SNR 2.32 dB
- Paired difference base - R2-1 base (noise_estimator=lowenergy): mean +0.0121, SD_d 0.1089, n 164, MDE = 2.8 x SD_d / sqrt(n) = 0.0238
- Processing time per file: mean 0.067 s, median 0.068 s, max 0.092 s (16 parallel workers, BLAS threads {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})

### Mean per target SNR

| target_snr_db | files | pesq_noisy | pesq | snr_db |
| --- | --- | --- | --- | --- |
| 0.0 | 15 | 1.0796 | 1.3285 | 1.6988 |
| 5.0 | 38 | 1.2181 | 1.5919 | 2.2819 |
| 10.0 | 55 | 1.3575 | 1.8677 | 2.6303 |
| 15.0 | 56 | 1.6354 | 2.2551 | 2.6622 |

### Files per noise type x target SNR

| noise_category | 0 dB | 5 dB | 10 dB | 15 dB | total |
| --- | --- | --- | --- | --- | --- |
| aircraft_cockpit | 1 | 4 | 4 | 5 | 14 |
| alarm_siren | 3 | 9 | 5 | 6 | 23 |
| animal | 0 | 0 | 1 | 1 | 2 |
| bell | 0 | 0 | 1 | 2 | 3 |
| broadband_noise | 2 | 3 | 3 | 2 | 10 |
| clap | 0 | 0 | 0 | 1 | 1 |
| click | 0 | 0 | 0 | 1 | 1 |
| destroyer_engine | 1 | 0 | 4 | 0 | 5 |
| destroyer_operations | 1 | 0 | 3 | 1 | 5 |
| door_moving | 0 | 0 | 0 | 1 | 1 |
| factory | 0 | 1 | 1 | 3 | 5 |
| footsteps | 0 | 0 | 1 | 0 | 1 |
| hf_channel | 1 | 1 | 2 | 0 | 4 |
| machine | 1 | 6 | 17 | 10 | 34 |
| machine_gun | 0 | 0 | 1 | 0 | 1 |
| military_vehicle | 0 | 1 | 3 | 1 | 5 |
| phone_dialing | 0 | 0 | 2 | 2 | 4 |
| shower | 0 | 0 | 0 | 1 | 1 |
| tooth_brushing | 0 | 0 | 0 | 1 | 1 |
| traffic_vehicle | 0 | 2 | 2 | 5 | 9 |
| vehicle_interior | 0 | 3 | 0 | 0 | 3 |
| water | 0 | 0 | 0 | 5 | 5 |
| wind | 5 | 8 | 5 | 8 | 26 |
| total | 15 | 38 | 55 | 56 | 164 |
