"""Round-03 Phase C0a/C0b: in-file periodic noise -- can it be detected from the noisy input
alone, and how much would subtracting a period template gain? (analysis only; the oracle
variants use the clean signal)

Per file:
  true_r, true_T   : max normalised autocorrelation of the TRUE noise (noisy - clean) over lags
                     0.1-1.0 s, and its lag (samples). 1.0 s keeps >= 3 periods in 3 s (median).
  hat_r, hat_T     : the same on the NOISY input (what a denoiser can see)
  hat_r_hp, ...    : the same on the noisy input high-passed at 1 kHz (less speech energy)
PESQ-WB of
  current          : the current denoiser
  sub_oracle       : noisy - template, template = median over aligned periods of the TRUE noise
                     at true_T (no further processing): ceiling of exact template subtraction
  sub_oracle_dn    : current denoiser applied to sub_oracle
  sub_hat          : noisy - template, template = median over aligned periods of the NOISY
                     input at hat_T (a realistic first prototype of C1, no level fit)
  sub_hat_dn       : current denoiser applied to sub_hat
Template: the signal is cut into consecutive periods of T samples (integer lag, our choice);
the template sample at offset j is the median over all periods of x[k T + j]; it is tiled
over the file (the last partial period uses the same offsets). Median over periods after
REPET (Rafii & Pardo, IEEE TASLP 21(1), 2013; there on the magnitude spectrogram, original
not checked); waveform-domain use is our choice.

Usage: python tools/diagnose_periodic.py --data-root DATA --split train --out OUT.csv
       [--subset CSV] [--summarize-only]
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
from scipy.signal import butter, sosfiltfilt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser import CONFIG  # noqa: E402
from denoiser.pipeline import denoise_signal  # noqa: E402
from evaluate import pesq_wb  # noqa: E402

LAG_MIN_S, LAG_MAX_S = 0.1, 1.0
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def autocorr_peak(x, sr):
    """Max over lags in [LAG_MIN_S, LAG_MAX_S] of the overlap-normalised autocorrelation."""
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(np.abs(f) ** 2, nfft)[:n]
    tail = np.cumsum((x ** 2)[::-1])[::-1]   # energy of x[lag:]
    head = np.cumsum(x ** 2)                 # energy of x[:n-lag] is head[n-1-lag]
    lags = np.arange(n)
    r = ac / (np.sqrt(tail[lags] * head[n - 1 - lags]) + 1e-12)
    lo, hi = int(LAG_MIN_S * sr), int(LAG_MAX_S * sr)
    k = lo + int(np.argmax(r[lo:hi]))
    return float(r[k]), int(k)


def period_template(x, period):
    """Median over consecutive periods at each offset, tiled to len(x)."""
    n = len(x)
    k = int(np.ceil(n / period))
    padded = np.full(k * period, np.nan)
    padded[:n] = x
    template = np.nanmedian(padded.reshape(k, period), axis=0)
    return np.tile(template, k)[:n]


def one_file(args):
    row, data_root = args
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    noise = y - s
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "noise_collection": row["noise_collection"], "target_snr_db": row["target_snr_db"]}
    out["true_r"], out["true_T"] = autocorr_peak(noise, sr)
    out["hat_r"], out["hat_T"] = autocorr_peak(y, sr)
    hp = sosfiltfilt(butter(4, 1000, "highpass", fs=sr, output="sos"), y)
    out["hat_r_hp"], out["hat_T_hp"] = autocorr_peak(hp, sr)
    est = {"current": denoise_signal(y, sr, CONFIG)}
    est["sub_oracle"] = y - period_template(noise, out["true_T"])
    est["sub_oracle_dn"] = denoise_signal(est["sub_oracle"], sr, CONFIG)
    est["sub_hat"] = y - period_template(y, out["hat_T"])
    est["sub_hat_dn"] = denoise_signal(est["sub_hat"], sr, CONFIG)
    for k, v in est.items():
        out[k] = pesq_wb(s, v, sr)[0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    if args.subset:
        rows = rows[rows.segment_id.isin(set(pd.read_csv(args.subset, comment="#").segment_id))]
    if args.summarize_only:
        df = pd.read_csv(args.out)
    else:
        jobs = [(dict(r), str(args.data_root)) for _, r in rows.iterrows()]
        with ProcessPoolExecutor(args.workers) as pool:
            df = pd.DataFrame(pool.map(one_file, jobs, chunksize=4))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    methods = ["current", "sub_oracle", "sub_oracle_dn", "sub_hat", "sub_hat_dn"]
    by = df.groupby("noise_category").agg(files=("segment_id", "size"),
                                          true_r=("true_r", "median"),
                                          hat_r=("hat_r", "median"),
                                          hat_r_hp=("hat_r_hp", "median"),
                                          **{m: (m, "mean") for m in methods})
    by = by.sort_values("true_r", ascending=False)
    by.loc["ALL"] = [len(df), df.true_r.median(), df.hat_r.median(), df.hat_r_hp.median()] + \
        [df[m].mean() for m in methods]
    by.round(3).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    print(by.round(3).to_string())
    # C0a: detection of truly periodic files (true_r >= 0.9) from the noisy input
    truth = df.true_r >= 0.9
    lines = []
    for feat in ("hat_r", "hat_r_hp"):
        tcol = "hat_T" if feat == "hat_r" else "hat_T_hp"
        for th in THRESHOLDS:
            det = df[feat] >= th
            tp = (det & truth).sum()
            same = (det & truth & ((df[tcol] - df.true_T).abs() <= 2)).sum()
            lines.append({"feature": feat, "threshold": th, "detected": int(det.sum()),
                          "precision": tp / max(det.sum(), 1), "recall": tp / max(truth.sum(), 1),
                          "period_match_of_tp": same / max(tp, 1),
                          # C0b: use sub_hat_dn on detected files, current elsewhere
                          "mean_pesq_rule": np.where(det, df.sub_hat_dn, df.current).mean(),
                          "mean_pesq_rule_best": np.where(
                              det, np.maximum(df.sub_hat_dn, df.sub_hat), df.current).mean()})
    det = pd.DataFrame(lines)
    det.round(4).to_csv(args.out.with_name(args.out.stem + "_detection.csv"), index=False)
    print(f"truly periodic (true_r >= 0.9): {int(truth.sum())} of {len(df)} files")
    print(det.round(4).to_string(index=False))
    oracle_rule = np.where(truth, np.maximum(df.sub_oracle_dn, df.current), df.current).mean()
    print(f"oracle rule (true_r >= 0.9 -> best of sub_oracle_dn/current): {oracle_rule:.4f}; "
          f"current {df.current.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
