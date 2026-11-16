"""Share of frames and files that open the round-02 step-14 transient gate, per noise type
(noisy input only; the noise type is used only for reporting).

Usage: python tools/diagnose_gate.py --data-root DATA --split train --out OUT.csv
       [--set KEY=VALUE ...] [--workers N]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.noise_estimation import gated_transient_noise  # noqa: E402
from denoiser.pipeline import first_pass_f0  # noqa: E402
from denoiser.stft import stft  # noqa: E402
from evaluate import parse_overrides  # noqa: E402

TARGET = ("clap", "click", "footsteps", "door_moving", "machine_gun", "tooth_brushing")


def one_file(args):
    row, data_root, overrides = args
    cfg = parse_overrides(["transient_module=true", "transient_mode=gated"] + overrides)
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    n_fft, hop = cfg.stft_sizes(sr)
    _, times, z = stft(y, sr, n_fft, hop)
    _, voiced = first_pass_f0(y, sr, times, cfg)
    gated_transient_noise(np.ones(z.shape[0]), z, voiced, sr, n_fft, cfg)
    g = gated_transient_noise.gate
    return {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
            "gate_share": float(g.mean()), "gate_any": bool(g.any())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    jobs = [(dict(r), str(args.data_root), args.set) for _, r in rows.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_file, jobs, chunksize=8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    t = df.noise_category.isin(TARGET)
    print(f"target files {t.sum()}: gate opens in {df.loc[t, 'gate_any'].mean():.2f} of files, "
          f"{df.loc[t, 'gate_share'].mean():.4f} of frames")
    print(f"other files {(~t).sum()}: gate opens in {df.loc[~t, 'gate_any'].mean():.3f} of files, "
          f"{df.loc[~t, 'gate_share'].mean():.4f} of frames")
    by = df.groupby("noise_category")[["gate_any", "gate_share"]].mean().round(4)
    print(by.sort_values("gate_share", ascending=False).head(12).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
