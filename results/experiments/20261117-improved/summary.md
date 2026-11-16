# R2-14: improvements from the weak-noise research report on r02-base-s4

Source of the candidates: `reports/약점 잡음 PESQ 개선 알고리즘.md` (priorities 1-3). Decision: **tonal module upgrade
adopted** (team rule 2026-11-16: Train and Validation both above the base); transient gate and steady floor not adopted.
New base `20261117-r02-base-s5`. Listening check pending (`listening/r02-step14_base_vs_tonalenv`, fixed list, 21 files).

## 1. Tonal module upgrade (adopted): tonal_env_max_db = 6, tonal_level = "excess"
- Envelope test (our choice): a kept r02 track's peak level (dB) minus its linear trend must have a standard deviation
  <= 6 dB (alarm/siren tones are steady, speech harmonics follow the syllable envelope).
- Excess power (our choice): only the power above the local spectral median (15 bins) on track bins is added to the noise
  PSD, so speech under a track is not removed with it (idea related to US 8,489,396, patent not checked).
- Subset grid (base 1.99046): envelope {3, 6} dB x lower frequency {400, 150} Hz x {observed, excess}: best 400 Hz,
  3 dB, excess 1.99331; 150 Hz always worse (1.977-1.988).
- Full Train (base 1.95389): 6 dB excess 1.95568, 3 dB excess 1.95409, 3 dB observed 1.95385.
- Chosen 6 dB / excess: Train +0.0018 [0.0008, 0.0030]; Validation 1.8989 vs 1.8980, +0.0009 [0.0001, 0.0028].
  SNR bins 0/5/10/15 dB: +0.0016 / +0.0018 / +0.0002 / +0.0033. Gains: phone_dialing +0.014 (43), vehicle_interior
  +0.007 (19), alarm_siren +0.006 (187), destroyer_engine +0.003 (36). Loss: bell -0.041 (15; two 10-15 dB files -0.24).
  README conditions: 1 FAIL (+0.01), CI PASS, types PASS, SNR bins PASS.

## 2. Safe transient gate (not adopted): transient_mode = "gated"
6d HPSS percussive power only in frames with a two-sided band-energy minimum jump (Talmon et al. 2013 idea) in >= 3 of 4
bands incl. 4-8 kHz, not voiced, no voiced frame within 12 frames. Gate opens in 0.9 % of frames of other types at
delta 10 (targets 6-20 %), but the subset grid delta {6, 10} x veto {0, 12} was below the base: -0.007 to -0.019.

## 3. Steady residual floor (not adopted): floor_mode = "noise"
Floored cells output floor x sqrt(noise PSD) instead of floor x |Y|. Subset (with the tonal upgrade): floor -12/-15/-18
dB: noisy-floor 1.98342/1.99252/1.99129, noise-floor 1.98820/1.99139/1.99029.

Time 0.77 s/file Train (unchanged).
