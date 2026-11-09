"""Validation evaluation: SNR and PESQ-WB, the same definitions as the baseline notebook.

The denoised WAV files that denoise.py writes are scored (not in-memory arrays), so the
numbers include the effect of saving to 16-bit PCM, just as the TA evaluation would.

Usage:
    python tools/evaluate.py --data-root PATH/noise_reduction_challenge
    python tools/evaluate.py --data-root PATH --split validation --enhanced-dir output

PATH must contain <split>/samples.csv. Requires requirements-eval.txt.
"""

import argparse
import json
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoiser import CONFIG, denoise_file  # noqa: E402
from dataclasses import asdict  # noqa: E402


def snr_db(clean, estimate):
    energy = np.sum(clean ** 2)
    if energy <= 0:
        return np.nan
    error = np.sum((clean - estimate) ** 2)
    return float(np.inf if error == 0 else 10 * np.log10(energy / error))


def pesq_wb(clean, estimate, sample_rate):
    from pesq import PesqError, pesq
    try:
        value = float(pesq(sample_rate, clean, estimate, mode="wb",
                           on_error=PesqError.RAISE_EXCEPTION))
        return (value, "") if np.isfinite(value) and value >= 0 else (np.nan, f"invalid {value}")
    except (PesqError, ValueError) as error:
        return np.nan, f"{type(error).__name__}: {error}"


def read_mono(path):
    x, sr = sf.read(str(path), dtype="float64")
    if x.ndim != 1:
        raise ValueError(f"expected mono: {path}")
    return x, sr


def score_row(job):
    root, row, enhanced_path = job
    clean, sr = read_mono(Path(root) / row["clean_path"])
    noisy, _ = read_mono(Path(root) / row["noisy_path"])
    enhanced, sr_out = read_mono(enhanced_path)
    if sr_out != sr or len(enhanced) != len(clean) or len(noisy) != len(clean):
        raise ValueError(f"shape/rate mismatch: {row['segment_id']}")
    in_pesq, in_err = pesq_wb(clean, noisy, sr)
    out_pesq, out_err = pesq_wb(clean, enhanced, sr)
    in_snr, out_snr = snr_db(clean, noisy), snr_db(clean, enhanced)
    return {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
            "noise_difficulty": row["noise_difficulty"], "target_snr_db": row["target_snr_db"],
            "snr_noisy_db": in_snr, "snr_enhanced_db": out_snr, "delta_snr_db": out_snr - in_snr,
            "pesq_noisy": in_pesq, "pesq_enhanced": out_pesq, "delta_pesq": out_pesq - in_pesq,
            "pesq_error": "; ".join(e for e in (in_err, out_err) if e)}


def summarize(detail, by=None):
    groups = detail.groupby(by, sort=True) if by else [("all", detail)]
    rows = []
    for key, part in groups:
        rows.append({by or "group": key, "files": len(part),
                     "snr_noisy_db": part.snr_noisy_db.mean(),
                     "snr_enhanced_db": part.snr_enhanced_db.mean(),
                     "delta_snr_db": part.delta_snr_db.mean(),
                     "pesq_noisy": part.pesq_noisy.mean(),
                     "pesq_enhanced": part.pesq_enhanced.mean(),
                     "delta_pesq": part.delta_pesq.mean(),
                     "pesq_failed": int(part.pesq_enhanced.isna().sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--enhanced-dir", type=Path,
                        help="folder with already denoised files named like the noisy files; "
                             "if omitted, denoise.py's algorithm is run first")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    root = args.data_root.resolve()
    manifest = pd.read_csv(root / args.split / "samples.csv")
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(args.workers) as pool:
        if args.enhanced_dir:
            enhanced_dir = args.enhanced_dir.resolve()
        else:
            enhanced_dir = Path(tmp)
            jobs = [(root / p, enhanced_dir / Path(p).name) for p in manifest.noisy_path]
            list(pool.map(_denoise_job, jobs, chunksize=8))
        targets = [enhanced_dir / Path(p).name for p in manifest.noisy_path]
        missing = [t for t in targets if not t.is_file()]
        if missing:
            print(f"Missing {len(missing)} enhanced files, e.g. {missing[0]}", file=sys.stderr)
            return 1
        jobs = [(str(root), row._asdict(), str(t))
                for row, t in zip(manifest.itertuples(index=False), targets)]
        detail = pd.DataFrame(pool.map(score_row, jobs, chunksize=8))

    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    overall = summarize(detail)
    detail.to_csv(out / f"{args.split}_per_file.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(out / f"{args.split}_summary.csv", index=False, encoding="utf-8-sig")
    for by in ("noise_category", "target_snr_db", "noise_difficulty"):
        summarize(detail, by).to_csv(out / f"{args.split}_by_{by}.csv", index=False,
                                     encoding="utf-8-sig")
    meta = {"split": args.split, "files": len(detail), "config": asdict(CONFIG),
            "python": platform.python_version(), "pesq_mode": "wb",
            "snr": "10*log10(sum(clean**2)/sum((clean-output)**2)), no alignment",
            "aggregation": "unweighted mean over files",
            "scored_files": "denoised WAV files as written by denoise.py"}
    (out / f"{args.split}_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pd.set_option("display.width", 120)
    print(overall.round(4).to_string(index=False))
    failed = int(detail.pesq_enhanced.isna().sum())
    if failed:
        print(f"PESQ failed for {failed} file(s); see pesq_error in the per-file CSV.")
    print(f"Saved to {out}")
    return 0


def _denoise_job(job):
    source, target = job
    denoise_file(source, target, CONFIG)


if __name__ == "__main__":
    sys.exit(main())
