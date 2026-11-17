"""Round-03 Phase A (E0, A2 + A4): where does the current denoiser lose PESQ? (analysis only)

Uses the CLEAN signal, so none of these is a denoiser. Splits the gap between the current
output and the ideal mask into "noise PSD estimate" and "a priori SNR (speech) estimate":

  current        : the current denoiser (defaults)
  on_period      : current pipeline with the true noise periodogram (3-frame smoothing), the
                   round-02 ceiling (20261117-ceiling), here with noise_estimator=spp_iter and
                   tonal_module=False so the oracle is not modified by tracking or tonal steps
  on_smooth      : same, but a realistic smoothed true noise PSD: recursive time smoothing
                   (0.95 / 0.05 per 8 ms frame) then a 3x3 (bin x frame) mean (our choice, E0)
  os_speech      : LSA gain with the ORACLE a priori SNR xi = |S|^2 / lambda and the current
                   noise estimate lambda (the last noise PSD the current pipeline passed to its
                   gain), current band floors and low-band cap: perfect speech estimate only
  os_both        : LSA with xi = |S|^2 / lambda_s and gamma = |Y|^2 / lambda_s, lambda_s = the
                   smoothed true noise PSD (on_smooth): both statistics oracle, LSA structure
  irm            : sqrt(|S|^2 / (|S|^2 + |N|^2)) on the noisy STFT (round-02 ceiling)
  wiener_oracle  : |S|^2 / (|S|^2 + |N|^2) (power ratio, "oracle Wiener")

Also scores the noisy input and the current output with PESQ-NB (ITU-T P.862 + P.862.1 mapping
in the `pesq` package, mode "nb") so published NB numbers can be compared with our WB scale,
and per band ([0, 325), [325, 1k), [1k, 4k), [4k, 8k) Hz) the current output's
  speech_db   : 10 log10(sum |E|^2 / sum |S|^2) over speech cells (clean cell power within
                30 dB of the file's maximum clean cell power): speech level change
  resid_db    : 10 log10(sum |E|^2 / sum |Y|^2) over non-speech cells: residual noise level
                relative to the input (energy-weighted, dominated by the loudest noise cells)
  residmed_db : median over non-speech cells of 10 log10(|E|^2 / |Y|^2) (typical suppression)
(the 30 dB threshold is our choice).

Usage: python tools/oracle_decompose.py --data-root DATA --split train --out OUT.csv [--subset CSV]
       [--summarize-only]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.ndimage import uniform_filter, uniform_filter1d  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import denoiser.pipeline as pipeline  # noqa: E402
from denoiser import CONFIG  # noqa: E402
from denoiser.gains import band_floor_profile, lsa_gain  # noqa: E402
from denoiser.stft import istft, stft  # noqa: E402
from evaluate import pesq_wb  # noqa: E402

BANDS = [(0, 325), (325, 1000), (1000, 4000), (4000, 8001)]
METHODS = ["noisy", "current", "on_period", "on_smooth", "os_speech", "os_both", "irm",
           "wiener_oracle"]


def pesq_nb(clean, estimate, sample_rate):
    from pesq import PesqError, pesq
    try:
        return float(pesq(sample_rate, clean, estimate, mode="nb",
                          on_error=PesqError.RAISE_EXCEPTION))
    except (PesqError, ValueError):
        return np.nan


def smoothed_noise_psd(pn):
    """Recursive smoothing over frames (0.95 / 0.05), then a 3 x 3 mean. Our choice."""
    rec = lfilter([0.05], [1, -0.95], pn, axis=1)
    rec[:, :20] = np.maximum(rec[:, :20], uniform_filter1d(pn, 5, axis=1)[:, :20])  # warm-up
    return uniform_filter(rec, size=3, mode="nearest") + 1e-12


def with_noise(y, sr, noise_tf):
    """Current pipeline with a fixed noise PSD (no tracking, no tonal step)."""
    cfg = replace(CONFIG, noise_estimator="spp_iter", tonal_module=False)
    original = pipeline.estimate_noise_psd
    pipeline.estimate_noise_psd = lambda *a, **k: noise_tf
    try:
        return pipeline.denoise_signal(y, sr, cfg)
    finally:
        pipeline.estimate_noise_psd = original


def current_and_noise(y, sr):
    """Current output, and the noise PSD of the outermost gain call (the final estimate)."""
    calls = []
    original = pipeline.lsa_with_prior

    def recorder(z, noise, *a, **k):
        calls.append(np.broadcast_to(noise[:, None] if noise.ndim == 1 else noise, z.shape))
        return original(z, noise, *a, **k)

    pipeline.lsa_with_prior = recorder
    try:
        out = pipeline.denoise_signal(y, sr, CONFIG)
    finally:
        pipeline.lsa_with_prior = original
    return out, np.array(calls[-1])


def lsa_oracle(zy, xi, gamma, sr, n_fft):
    floor, cap = band_floor_profile(zy.shape[0], sr, n_fft, CONFIG)
    gain = np.maximum(lsa_gain(xi, gamma), floor[:, None])
    gain = np.minimum(gain, cap[:, None])  # cap is inf above the cut-off
    return gain * zy


def band_stats(zs, ze, zy, sr, n_fft):
    freqs = np.arange(zs.shape[0]) * sr / n_fft
    ps, pe, py = np.abs(zs) ** 2, np.abs(ze) ** 2, np.abs(zy) ** 2
    speech = ps > ps.max() * 1e-3
    out = {}
    for lo, hi in BANDS:
        b = (freqs >= lo) & (freqs < hi)
        sp, ns = speech[b], ~speech[b]
        name = f"{lo}-{hi if hi < 8001 else 8000}"
        out[f"speech_db_{name}"] = 10 * np.log10(pe[b][sp].sum() / max(ps[b][sp].sum(), 1e-30)) \
            if sp.any() else np.nan
        out[f"resid_db_{name}"] = 10 * np.log10(pe[b][ns].sum() / max(py[b][ns].sum(), 1e-30)) \
            if ns.any() else np.nan
        cell = 10 * np.log10(np.maximum(pe[b][ns], 1e-30) / np.maximum(py[b][ns], 1e-30))
        out[f"residmed_db_{name}"] = float(np.median(cell)) if ns.any() else np.nan
    return out


def one_file(args):
    row, data_root = args
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, _, zy = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    _, _, zn = stft(y - s, sr, n_fft, hop)
    ps, pn, py = np.abs(zs) ** 2, np.abs(zn) ** 2, np.abs(zy) ** 2
    est = {"noisy": y}
    est["current"], lam = current_and_noise(y, sr)
    est["on_period"] = with_noise(y, sr, uniform_filter1d(pn, 3, axis=1, mode="nearest") + 1e-12)
    lam_s = smoothed_noise_psd(pn)
    est["on_smooth"] = with_noise(y, sr, lam_s)
    est["os_speech"] = istft(lsa_oracle(zy, ps / lam, py / lam, sr, n_fft), len(y), sr, n_fft, hop)
    est["os_both"] = istft(lsa_oracle(zy, ps / lam_s, py / lam_s, sr, n_fft), len(y), sr, n_fft,
                           hop)
    est["irm"] = istft(np.sqrt(ps / np.maximum(ps + pn, 1e-30)) * zy, len(y), sr, n_fft, hop)
    est["wiener_oracle"] = istft(ps / np.maximum(ps + pn, 1e-30) * zy, len(y), sr, n_fft, hop)
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    for k, v in est.items():
        out[k] = pesq_wb(s, v, sr)[0]
    out["noisy_nb"] = pesq_nb(s, y, sr)
    out["current_nb"] = pesq_nb(s, est["current"], sr)
    _, _, ze = stft(est["current"], sr, n_fft, hop)
    out.update(band_stats(zs, ze, zy, sr, n_fft))
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
    cols = METHODS + ["noisy_nb", "current_nb"]
    by = df.groupby("noise_category")[cols].mean()
    by.insert(0, "files", df.groupby("noise_category").size())
    # points lost (A3): files x (ceiling - current) / all files, in PESQ of the overall mean
    for ceiling in ("on_smooth", "os_speech", "os_both"):
        gap = (df[ceiling] - df["current"]).groupby(df.noise_category).sum() / len(df)
        by[f"share_{ceiling}"] = gap
    by = by.sort_values("share_os_speech", ascending=False)
    total = [len(df)] + [df[c].mean() for c in cols] + \
        [(df[c] - df["current"]).mean() for c in ("on_smooth", "os_speech", "os_both")]
    by.loc["ALL (mean of files)"] = total
    by.round(4).to_csv(args.out.with_name(args.out.stem + "_by_type.csv"))
    snr = df.groupby("target_snr_db")[cols].mean().round(4)
    snr.to_csv(args.out.with_name(args.out.stem + "_by_snr.csv"))
    band_cols = [c for c in df.columns if c.startswith(("speech_db", "resid_db", "residmed_db"))]
    bands = df.groupby("target_snr_db")[band_cols].mean()
    bands.loc["ALL"] = df[band_cols].mean()
    bands.round(2).to_csv(args.out.with_name(args.out.stem + "_bands.csv"))
    pd.set_option("display.width", 250)
    print(by.round(3).to_string())
    print(snr.to_string())
    print(bands.round(2).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
