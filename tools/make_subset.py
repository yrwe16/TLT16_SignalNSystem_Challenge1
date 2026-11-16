"""Stratified Train subset for quick candidate screening (round-02 step 10).

Takes up to --per-cell segments (random, fixed seed) from every noise_category x
target_snr_db cell, so all noise types and SNR bins are represented evenly.

Usage: python tools/make_subset.py --data-root DATA --per-cell 6 --out subset.csv [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-cell", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    m = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    shuffled = m.sample(frac=1.0, random_state=args.seed)
    pick = shuffled.groupby(["noise_category", "target_snr_db"]).head(args.per_cell)
    pick = pick.sort_values("segment_id")[["segment_id", "noise_category", "target_snr_db"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(f"# stratified {args.split} subset: up to {args.per_cell} per noise_category x "
                f"target_snr_db cell, seed {args.seed}\n")
        pick.to_csv(f, index=False)
    print(f"{len(pick)} segments, {pick.groupby(['noise_category', 'target_snr_db']).ngroups} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
