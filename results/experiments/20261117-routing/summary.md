# R2-13 (version C): file-feature routing of modules - oracle bounds only

Decision: **not implemented** (every rule's type-oracle bound is below +0.01; prompt: rules under +0.01 are reported,
not implemented). Defaults unchanged (r02-base-s4). Whether hand-set routing rules with grid-tuned thresholds count
as ML would need TA confirmation; moot here.

Each module was run on the full Train set on top of r02-base-s4 (`modules-train/`); `tools/oracle_routing.py`
computes the per-file change. Type oracle = module on exactly for the noise types whose mean change is positive
(perfect routing by label, the best any file rule could do at type level); file oracle = on exactly for improving
files (looser).

| module (rule) | all files on | type oracle | file oracle | types switched on by the oracle (files, mean change) |
| --- | ---: | ---: | ---: | --- |
| 6d transient (transience high) | -0.0485 | **+0.0050** | +0.0115 | clap (3, +0.49), door_moving (3, +0.44), footsteps (8, +0.27), click (3, +0.23), tooth_brushing (3, +0.02), water (57, +0.01) |
| 8 masking (stationary, low SNR) | +0.0002 | +0.0003 | +0.0005 | many types, each <= +0.002 |
| tonal peak 6 dB (narrowband high) | -0.0073 | +0.0008 | +0.0045 | traffic (76, +0.010), destroyer_engine (36, +0.009) |
| 6 postfilter zeta 0.4, N <= 7 | -0.0068 | +0.0009 | +0.0046 | machine_gun (3, +0.03), vehicle_interior (19, +0.015), destroyer_operations (36, +0.006) |

Sum of the type-oracle bounds: about +0.007. The transient module helps a lot on four rare types (20 of 1296 files,
1.5 %), so even perfect routing moves the mean by only +0.005; it costs -0.2 on phone_dialing and vehicle_interior
when switched on wrongly. Per-type table: `oracle_train_by_type.csv`.
