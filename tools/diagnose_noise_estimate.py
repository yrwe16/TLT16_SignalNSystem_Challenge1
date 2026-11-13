"""Diagnosis only: how far the noise PSD estimate is from the true noise, per band.

Usage:
    python tools/diagnose_noise_estimate.py --data-root PATH --split train --out DIR
        [--set KEY=VALUE ...] [--gain-base RUN_DIR --gain-new RUN_DIR --top 50]

The true noise is noisy - clean (the clean signal is used ONLY here, never by the
denoiser). For each file the estimate of the configured estimator
(denoiser.pipeline.estimate_noise_psd) and the true noise PSD (mean power over the
complete STFT frames) are summed per band; the ratio estimate / true in dB is reported per
band, overall and per noise_category and target_snr_db (mean and median over files).
With --gain-base/--gain-new (two evaluate.py result folders of the same split) the files
that improved most from base to new are summarised separately (e.g. the low cut).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.noise_estimation import complete_frames  # noqa: E402
from denoiser.pipeline import estimate_noise_psd  # noqa: E402
from denoiser.stft import stft  # noqa: E402
from evaluate import parse_overrides  # noqa: E402

BANDS = [(0, 325), (325, 1000), (1000, 4000), (4000, 8000)]
NAMES = [f"{lo}-{hi}Hz" for lo, hi in BANDS]


def file_ratios(job):
    root, row, cfg = job
    noisy, sr = sf.read(str(root / row["noisy_path"]), dtype="float64")
    clean, _ = sf.read(str(root / row["clean_path"]), dtype="float64")
    n_fft, hop = cfg.stft_sizes(sr)
    _, times, z = stft(noisy, sr, n_fft, hop)
    _, _, zn = stft(noisy - clean, sr, n_fft, hop)
    est = estimate_noise_psd(z, times, len(noisy), sr, n_fft, hop, cfg)
    est = est.mean(axis=1) if est.ndim == 2 else est
    full = complete_frames(times, len(noisy), sr, n_fft, z.shape[1])
    true = (np.abs(zn[:, full]) ** 2).mean(axis=1)
    freqs = np.arange(z.shape[0]) * sr / n_fft
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    for (lo, hi), name in zip(BANDS, NAMES):
        band = (freqs >= lo) & (freqs < hi) if hi < sr / 2 else (freqs >= lo)
        out[name] = 10 * np.log10(est[band].sum() / max(true[band].sum(), 1e-30))
    return out


def summary(frame, by=None):
    groups = [("all", frame)] if by is None else frame.groupby(by, sort=True)
    rows = []
    for key, part in groups:
        row = {by or "group": key, "files": len(part)}
        for name in NAMES:
            row[f"{name} mean"] = round(part[name].mean(), 2)
            row[f"{name} median"] = round(part[name].median(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def md(frame):
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    return "\n".join([head, sep] + ["| " + " | ".join(map(str, r)) + " |"
                                    for r in frame.itertuples(index=False)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gain-base", type=Path)
    parser.add_argument("--gain-new", type=Path)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    cfg = parse_overrides(args.set)
    root = args.data_root.resolve()
    manifest = pd.read_csv(root / args.split / "samples.csv")
    jobs = [(root, row._asdict(), cfg) for row in manifest.itertuples(index=False)]
    with ProcessPoolExecutor() as pool:
        detail = pd.DataFrame(pool.map(file_ratios, jobs, chunksize=16))
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / f"{args.split}_noise_ratio_per_file.csv", index=False,
                  encoding="utf-8-sig")

    parts = [f"# Noise estimate / true noise power, {args.split} ({len(detail)} files)", "",
             f"Estimator overrides: {args.set or 'none (defaults)'}. Ratio in dB per band "
             "(positive = estimate too large). True noise = noisy - clean (diagnosis only).", "",
             "## All files", "", md(summary(detail)), "",
             "## Per target SNR", "", md(summary(detail, "target_snr_db")), "",
             "## Per noise type", "", md(summary(detail, "noise_category")), ""]
    if args.gain_base and args.gain_new:
        name = f"{args.split}_per_file.csv"
        base = pd.read_csv(args.gain_base / name)[["segment_id", "pesq_enhanced"]]
        new = pd.read_csv(args.gain_new / name)[["segment_id", "pesq_enhanced"]]
        gain = base.merge(new, on="segment_id", suffixes=("_base", "_new"))
        gain["gain"] = gain.pesq_enhanced_new - gain.pesq_enhanced_base
        top = gain.nlargest(args.top, "gain")
        sub = detail[detail.segment_id.isin(top.segment_id)]
        top.merge(detail, on="segment_id").to_csv(args.out / f"{args.split}_top{args.top}.csv",
                                                  index=False, encoding="utf-8-sig")
        parts += [f"## {args.top} files with the largest PESQ gain {args.gain_base.name} -> "
                  f"{args.gain_new.name} (mean gain {top.gain.mean():+.3f})", "",
                  md(summary(sub)), "",
                  "Noise types among them: " + ", ".join(
                      f"{k} {v}" for k, v in sub.noise_category.value_counts().items()), ""]
    text = "\n".join(parts)
    (args.out / f"{args.split}_noise_ratio.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
