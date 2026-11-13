"""Baseline statistics for a fixed reference run (writes a Markdown summary).

Usage:
    python tools/baseline_summary.py --base RUN_DIR --contrast RUN_DIR --out summary.md

RUN_DIR holds <split>/<split>_per_file.csv for split in train and validation (evaluate.py
--results-dir RUN_DIR/<split>). For both splits the summary lists:
  - the number of files per target SNR x noise_category,
  - mean PESQ-WB and SNR per target SNR bin and the SNR-equally-weighted mean,
  - SD_d, the standard deviation of per-file PESQ differences between the base run and the
    contrast run, and the minimum detectable effect MDE = 2.8 * SD_d / sqrt(n)
    (two-sided alpha 0.05, power 0.8),
  - processing time per file (from <split>_run.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(run_dir, split):
    return pd.read_csv(run_dir / split / f"{split}_per_file.csv")


def md_table(frame, index=True):
    frame = frame.reset_index() if index else frame
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--contrast", type=Path, required=True,
                        help="second run used to measure SD_d (e.g. one switch off)")
    parser.add_argument("--contrast-label", default="contrast")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lines = [f"# Baseline statistics: {args.base.name}", ""]
    for split in ("train", "validation"):
        base, other = load(args.base, split), load(args.contrast, split)
        merged = base.merge(other[["segment_id", "pesq_enhanced"]], on="segment_id",
                            suffixes=("", "_contrast"), validate="one_to_one")
        counts = pd.crosstab(base.noise_category, base.target_snr_db, margins=True,
                             margins_name="total")
        counts.columns = [f"{c:g} dB" if isinstance(c, float) else c for c in counts.columns]
        by_snr = base.groupby("target_snr_db").agg(files=("segment_id", "size"),
                                                    pesq_noisy=("pesq_noisy", "mean"),
                                                    pesq=("pesq_enhanced", "mean"),
                                                    snr_db=("snr_enhanced_db", "mean")).round(4)
        weighted_pesq = by_snr.pesq.mean()
        weighted_snr = by_snr.snr_db.mean()
        d = (merged.pesq_enhanced - merged.pesq_enhanced_contrast).dropna()
        sd = float(d.std(ddof=1))
        mde = 2.8 * sd / np.sqrt(len(d))
        meta = json.loads((args.base / split / f"{split}_run.json").read_text(encoding="utf-8"))
        timing = meta.get("process_seconds_per_file", {})
        lines += [f"## {split.capitalize()} ({len(base)} files)", "",
                  f"- Mean PESQ-WB {base.pesq_enhanced.mean():.4f}, mean SNR "
                  f"{base.snr_enhanced_db.mean():.2f} dB (noisy: PESQ {base.pesq_noisy.mean():.4f}, "
                  f"SNR {base.snr_noisy_db.mean():.2f} dB)",
                  f"- SNR-equally-weighted mean: PESQ-WB {weighted_pesq:.4f}, SNR {weighted_snr:.2f} dB",
                  f"- Paired difference base - {args.contrast_label}: mean {d.mean():+.4f}, "
                  f"SD_d {sd:.4f}, n {len(d)}, MDE = 2.8 x SD_d / sqrt(n) = {mde:.4f}",
                  f"- Processing time per file: mean {timing.get('mean', float('nan')):.3f} s, "
                  f"median {timing.get('median', float('nan')):.3f} s, max "
                  f"{timing.get('max', float('nan')):.3f} s ({meta.get('workers')} parallel workers, "
                  f"BLAS threads {meta.get('blas_threads')})", "",
                  "### Mean per target SNR", "", md_table(by_snr), "",
                  "### Files per noise type x target SNR", "", md_table(counts), ""]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
