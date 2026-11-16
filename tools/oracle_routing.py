"""Oracle routing upper bound per module (round-02 step 13, analysis only).

For each module run (evaluate.py results of the base + one module), the per-file PESQ change
against the base is computed. "Type oracle": the module is switched on exactly for the noise
types whose mean change is positive (perfect routing by the noise-type label); the bound is the
resulting mean change over all files. "File oracle": switched on exactly for the files that
improve (an even looser bound). Labels are used here only to bound what a rule could gain.

Usage: python tools/oracle_routing.py --base BASE_RUN_DIR --split train --out OUT.csv
           --module NAME=RUN_DIR [--module ...]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--module", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = pd.read_csv(args.base / f"{args.split}_per_file.csv")[
        ["segment_id", "noise_category", "target_snr_db", "pesq_enhanced"]]
    rows, types = [], []
    for item in args.module:
        name, _, run = item.partition("=")
        new = pd.read_csv(Path(run) / f"{args.split}_per_file.csv")[["segment_id", "pesq_enhanced"]]
        m = base.merge(new, on="segment_id", suffixes=("_base", "_new"))
        m["diff"] = m.pesq_enhanced_new - m.pesq_enhanced_base
        by = m.groupby("noise_category")["diff"].agg(["size", "mean"])
        on_types = by.index[by["mean"] > 0]
        type_oracle = m.loc[m.noise_category.isin(on_types), "diff"].sum() / len(m)
        file_oracle = m["diff"].clip(lower=0).sum() / len(m)
        rows.append({"module": name, "all_on": round(m["diff"].mean(), 4),
                     "type_oracle": round(type_oracle, 4), "file_oracle": round(file_oracle, 4),
                     "types_on": ", ".join(f"{t} ({int(by.loc[t, 'size'])}, {by.loc[t, 'mean']:+.3f})"
                                           for t in by.sort_values("mean", ascending=False).index
                                           if t in on_types)})
        types.append(by.rename(columns={"mean": name, "size": "files"})[[name]])
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    table = pd.concat(types, axis=1)
    table.insert(0, "files", base.groupby("noise_category").size())
    table.round(4).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    print(out.to_string(index=False))
    print(table.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
