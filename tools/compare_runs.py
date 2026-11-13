"""Paired comparison of two evaluate.py runs on the same split (file-by-file).

Usage:
    python tools/compare_runs.py BASE_DIR NEW_DIR [--split train] [--out OUT_DIR]
                                 [--snr-floor 3.8]

Reads <split>_per_file.csv from both folders, pairs files by segment_id and reports for
PESQ-WB (and SNR):
  - overall means of both runs, the paired mean difference and its paired bootstrap 95 %
    interval (scipy.stats.bootstrap, 10,000 resamples of the per-file differences, BCa,
    seed 0), the Wilcoxon signed-rank p-value and the share of files that improved,
  - the mean per target SNR (0/5/10/15 dB) and the SNR-equally-weighted mean,
  - the change per noise_category (with file counts) and per target SNR,
  - the five files that got worst,
  - the round-02 adoption checks (common rules): Train mean +0.01 with the interval above 0,
    no noise type with >= 10 files down by more than 0.1, no SNR bin down by more than 0.02;
    Validation mean not lower; Validation mean SNR >= --snr-floor.
Requires requirements-eval.txt (pandas, scipy).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESAMPLES = 10_000


def bootstrap_ci(d):
    """Paired bootstrap 95 % interval of the mean of per-file differences."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.all(d == d[0]):
        return (float(d.mean()) if len(d) else np.nan,) * 2
    res = stats.bootstrap((d,), np.mean, n_resamples=RESAMPLES, confidence_level=0.95,
                          method="BCa", random_state=np.random.default_rng(0))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired(base, new, ci=True):
    d = (np.asarray(new, float) - np.asarray(base, float))
    d = d[np.isfinite(d)]
    out = {"n": len(d), "base": float(np.nanmean(base)), "new": float(np.nanmean(new)),
           "mean_diff": float(d.mean()) if len(d) else np.nan,
           "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
           "improved_share": float((d > 0).mean()) if len(d) else np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "wilcoxon_p": np.nan}
    if ci and len(d) >= 2:
        out["ci_low"], out["ci_high"] = bootstrap_ci(d)
    if len(d) >= 2 and np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    return out


def table(df, by, metric, ci=True):
    groups = [("all", df)] if by is None else df.groupby(by, sort=True)
    return pd.DataFrame([{by or "group": key,
                          **paired(part[f"{metric}_base"], part[f"{metric}_new"], ci)}
                         for key, part in groups])


def snr_weighted(df, column):
    """Mean over target SNR bins, each bin weighted equally."""
    return float(df.groupby("target_snr_db")[column].mean().mean())


def load_pairs(base_dir, new_dir, split):
    name = f"{split}_per_file.csv"
    keep = ["segment_id", "noise_category", "target_snr_db", "pesq_enhanced", "snr_enhanced_db"]
    base = pd.read_csv(base_dir / name)[keep]
    new = pd.read_csv(new_dir / name)[keep]
    df = base.merge(new, on=["segment_id", "noise_category", "target_snr_db"],
                    suffixes=("_base", "_new"), validate="one_to_one")
    if len(df) != len(base) or len(df) != len(new):
        raise SystemExit("Runs do not contain the same files.")
    return df.rename(columns=lambda c: c.replace("pesq_enhanced", "pesq")
                     .replace("snr_enhanced_db", "snr"))


def criteria(results, df, split, snr_floor):
    overall = results["pesq_all"].iloc[0]
    checks = {}
    if split == "train":
        checks["train_mean_+0.01"] = bool(overall.mean_diff >= 0.01)
        checks["train_ci_above_0"] = bool(overall.ci_low > 0)
        cat = results["pesq_noise_category"]
        big = cat[cat.n >= 10]
        worst_cat = big.loc[big.mean_diff.idxmin()] if len(big) else None
        checks["no_type_n>=10_down_>0.1"] = bool(worst_cat is None or worst_cat.mean_diff >= -0.1)
        snr = results["pesq_target_snr_db"]
        checks["no_snr_bin_down_>0.02"] = bool(snr.mean_diff.min() >= -0.02)
    else:
        checks["validation_mean_not_lower"] = bool(overall.mean_diff >= 0)
        checks[f"validation_snr_>={snr_floor}"] = bool(df.snr_new.mean() >= snr_floor)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", type=Path, help="folder for the comparison CSV/JSON files")
    parser.add_argument("--snr-floor", type=float, default=3.8,
                        help="minimum Validation mean SNR in dB (team decision 1)")
    args = parser.parse_args()

    df = load_pairs(args.base, args.new, args.split)
    results = {}
    for metric in ("pesq", "snr"):
        for by in (None, "noise_category", "target_snr_db"):
            results[f"{metric}_{by or 'all'}"] = table(df, by, metric, ci=by is None or metric == "pesq")
    df["pesq_diff"] = df.pesq_new - df.pesq_base
    worst = df.nsmallest(5, "pesq_diff")[["segment_id", "noise_category", "target_snr_db",
                                         "pesq_base", "pesq_new", "pesq_diff"]]
    weighted = {m: {"base": snr_weighted(df, f"{m}_base"), "new": snr_weighted(df, f"{m}_new")}
                for m in ("pesq", "snr")}
    checks = criteria(results, df, args.split, args.snr_floor)

    pd.set_option("display.width", 180)
    cols = ["n", "base", "new", "mean_diff", "ci_low", "ci_high", "improved_share", "wilcoxon_p"]
    for key in ("pesq_all", "snr_all", "pesq_target_snr_db", "pesq_noise_category"):
        frame = results[key]
        print(f"\n== {key} ==")
        print(frame[[frame.columns[0]] + cols].round(4).to_string(index=False))
    print("\n== SNR-equally-weighted mean ==")
    for m, v in weighted.items():
        print(f"{m}: base {v['base']:.4f}  new {v['new']:.4f}  diff {v['new'] - v['base']:+.4f}")
    print("\n== 5 files that got worst (PESQ-WB) ==")
    print(worst.round(4).to_string(index=False))
    print("\n== adoption checks (" + args.split + ") ==")
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, frame in results.items():
            frame.to_csv(args.out / f"compare_{key}.csv", index=False, encoding="utf-8-sig")
        worst.to_csv(args.out / "compare_worst5.csv", index=False, encoding="utf-8-sig")
        (args.out / "compare_run.json").write_text(json.dumps(
            {"base": str(args.base), "new": str(args.new), "split": args.split,
             "snr_weighted_mean": weighted, "checks": checks,
             "test": f"paired bootstrap 95% CI of the mean difference (scipy.stats.bootstrap, "
                     f"{RESAMPLES} resamples, BCa, seed 0); Wilcoxon signed-rank, two-sided"},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
