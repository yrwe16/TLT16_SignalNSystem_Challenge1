# R2-9: transient noise (clap, click, footsteps, door_moving, machine_gun) on r02-base-s3

Decision: **not adopted** (stopped before the Train grid; team decision 2026-11-16). Defaults unchanged.
Target types are 23 of 1296 Train files (1.8 %).

## 1. Stage-6d HPSS module re-evaluated on the current base (Train)
-0.042 [-0.048, -0.036] overall. Targets: clap +0.49 (3), door_moving +0.43 (3), footsteps +0.27 (8), click +0.23 (3),
machine_gun +0.01 (3). Others: traffic -0.066 (76), wind -0.065 (202), machine -0.046 (264), animal -0.040 (39),
alarm_siren -0.029 (187). Usable only with a file gate.

## 2. Two-stage transient estimate (transient_mode="two_stage", off by default)
Hirszhorn, Dov, Talmon & Cohen, IWAENC 2012 (read in the original, PDF supplied by the team, not committed): 64-sample
frames, 75 % overlap, alpha_s 0.7 (eq. 5), causal and 40 ms anti-causal minima, S_min = max (eqs. 6, 12, 13),
S / S_min > delta (eq. 7). Causal window 0.2 s, delta grid, bin share 0.5 and the file gate are our choices (the paper
gives no L or delta); the first-stage OM-LSA is replaced by the excess S - S_min, added to the stationary estimate
for the LSA (OM-LSA not used).

Transience index (share of 4 ms frames with >= 50 % of bins flagged), Train medians:
| delta | target files (20) | other files (median / 95th pct) |
| ---: | ---: | ---: |
| 3 | 47 % | 34 % / 55 % |
| 4 | 34 % | 18 % / 40 % |
| 6 | 22 % | 8.1 % / 25 % |
With the reference raised to max(S_min, stationary PSD) (team-approved fix, our choice), delta 6: targets 12.5 %, others
3.6 % / 13 %; a gate that keeps 95 % of the targets still switches the module on in 41-78 % of the other files.

Cause (100 files, clean used only for the diagnosis), delta 6 median index: other files noisy 3.9 %, speech only 4.3 %,
noise only 0 %; target files noisy 12.5 %, speech only 8.4 %, noise only 3.7 %. The stationary floor removes the
noise fluctuations, but speech itself (glottal pulses and syllable changes at 4 ms resolution) passes the 40 ms
anti-causal test, so the flags in ordinary files are speech. Adding them to the noise PSD would suppress speech.
