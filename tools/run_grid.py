"""Run a list of evaluate.py configurations and collect their mean scores (round-02 step 10).

Each candidate is NAME followed by KEY=VALUE overrides; "floor_db=F" expands to the LSA floor
of all bands (lsa_gain_floor and band_mid/midhigh/high_floor_db).

Usage: python tools/run_grid.py --data-root DATA --split train --out-dir DIR [--subset CSV]
           --candidate "f-15_a0.95 floor_db=-15 lsa_smoothing=0.95" [--candidate ...]
Writes DIR/<name>/ (evaluate.py results) and DIR/grid.csv (name, overrides, pesq, snr).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def expand(items):
    out = []
    for item in items:
        key, _, value = item.partition("=")
        if key == "floor_db":
            f = float(value)
            out += [f"lsa_gain_floor={10 ** (f / 20)!r}", f"band_mid_floor_db={f}",
                    f"band_midhigh_floor_db={f}", f"band_high_floor_db={f}"]
        else:
            out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()
    rows = []
    for cand in args.candidate:
        name, *items = cand.split()
        target = args.out_dir / name
        summary = target / f"{args.split}_summary.csv"
        if not summary.is_file():
            cmd = [sys.executable, "-W", "ignore", str(ROOT / "tools" / "evaluate.py"),
                   "--data-root", str(args.data_root), "--split", args.split,
                   "--results-dir", str(target)]
            if args.subset:
                cmd += ["--subset", str(args.subset)]
            for item in expand(items):
                cmd += ["--set", item]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        s = pd.read_csv(summary).iloc[0]
        rows.append({"name": name, "overrides": " ".join(items), "files": int(s.files),
                     "pesq": round(float(s.pesq_enhanced), 5), "snr_db": round(float(s.snr_enhanced_db), 3)})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    path = args.out_dir / "grid.csv"
    if path.is_file():
        grid = pd.concat([pd.read_csv(path), grid]).drop_duplicates("name", keep="last")
    grid.sort_values("pesq", ascending=False).to_csv(path, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
