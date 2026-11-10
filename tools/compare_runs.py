"""Paired comparison of two evaluate.py runs on the same split (file-by-file).

Usage:
    python tools/compare_runs.py BASE_DIR NEW_DIR [--split validation] [--out OUT_DIR]

Reads <split>_per_file.csv from both folders, pairs files by segment_id and reports:
  - overall mean PESQ-WB / SNR of both runs and the paired difference,
  - Wilcoxon signed-rank test (two-sided) on per-file PESQ differences, plus a paired t-test,
    a bootstrap 95 % CI of the mean difference and the share of files that improved,
  - the same per noise_category and per target_snr_db.
Requires requirements-eval.txt (pandas, scipy).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired(base, new, rng):
    d = (new - base).to_numpy()
    d = d[np.isfinite(d)]
    out = {"n": len(d), "base": float(np.nanmean(base)), "new": float(np.nanmean(new)),
           "mean_diff": float(d.mean()) if len(d) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan,
           "improved_share": float((d > 0).mean()) if len(d) else np.nan,
           "wilcoxon_p": np.nan, "ttest_p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if len(d) >= 2 and np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        out["ttest_p"] = float(stats.ttest_1samp(d, 0.0).pvalue)
        boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        out["ci_low"], out["ci_high"] = (float(q) for q in np.percentile(boots, [2.5, 97.5]))
    return out


def table(df, by, metric, rng):
    groups = [("all", df)] if by is None else df.groupby(by, sort=True)
    rows = []
    for key, part in groups:
        rows.append({by or "group": key, **paired(part[f"{metric}_base"], part[f"{metric}_new"], rng)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", type=Path, help="folder for the comparison CSV/JSON files")
    args = parser.parse_args()

    name = f"{args.split}_per_file.csv"
    keep = ["segment_id", "noise_category", "target_snr_db", "pesq_enhanced", "snr_enhanced_db"]
    base = pd.read_csv(args.base / name)[keep]
    new = pd.read_csv(args.new / name)[keep]
    df = base.merge(new, on=["segment_id", "noise_category", "target_snr_db"],
                    suffixes=("_base", "_new"), validate="one_to_one")
    if len(df) != len(base) or len(df) != len(new):
        print("Runs do not contain the same files.", file=sys.stderr)
        return 1
    df = df.rename(columns=lambda c: c.replace("pesq_enhanced", "pesq")
                   .replace("snr_enhanced_db", "snr"))

    rng = np.random.default_rng(0)
    results = {}
    for metric in ("pesq", "snr"):
        for by in (None, "noise_category", "target_snr_db"):
            results[f"{metric}_{by or 'all'}"] = table(df, by, metric, rng)

    pd.set_option("display.width", 160)
    cols = ["n", "base", "new", "mean_diff", "ci_low", "ci_high", "improved_share", "wilcoxon_p"]
    for key in ("pesq_all", "snr_all", "pesq_noise_category", "pesq_target_snr_db"):
        frame = results[key]
        print(f"\n== {key} ==")
        print(frame[[frame.columns[0]] + cols].round(4).to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, frame in results.items():
            frame.to_csv(args.out / f"compare_{key}.csv", index=False, encoding="utf-8-sig")
        (args.out / "compare_run.json").write_text(json.dumps(
            {"base": str(args.base), "new": str(args.new), "split": args.split,
             "test": "Wilcoxon signed-rank, two-sided, zero_method=wilcox; paired t-test; "
                     "bootstrap 95% CI of mean difference (10000 resamples, seed 0)"},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
