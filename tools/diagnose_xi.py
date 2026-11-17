"""Round-03 Phase C0c/C0d: how accurate must the a priori SNR be before lower LSA floors pay
off, and how far is our decision-directed estimate from that? (analysis only; uses clean)

Noise PSD lambda = the final estimate of the current pipeline (as in oracle_decompose.py),
gamma = |Y|^2 / lambda. A priori SNR sources:
  or_s{0,2,4,8} : oracle xi = |S|^2 / lambda times iid log-normal error with sigma 0/2/4/8 dB
                  per cell (seeded per file; our choice of error model)
  dd            : our estimate, mean of forward and backward DD (alpha = cfg.lsa_smoothing),
                  exactly as gains.lsa_with_prior computes it; the DD recursion uses the same
                  floor as the final gain (its feedback otherwise keeps xi above the old floor)
  or_nonspeech  : oracle xi only in non-speech cells (clean power more than 30 dB below the
                  file maximum), dd in speech cells: value of a perfect cell-level speech
                  presence decision
  or_speech     : oracle xi only in speech cells, dd elsewhere
  fix_over      : min(dd, oracle): only DD over-estimates are corrected
  fix_under     : max(dd, oracle): only DD under-estimates are corrected
  fix_over_6db  : min(dd, 4 x oracle): over-estimates above +6 dB are clipped
  harm          : oracle xi only in voiced harmonic cells (YIN f0 of the CLEAN signal, harmonics
                  +-1 bin, voiced frames), dd elsewhere: the ceiling of fixing voiced cells
                  only (C3, harmonic model)
Floors: cur = current band floors; -25/-30/-40 = one floor above the low band; none. The
low-band cap (300 Hz, -25 dB) is always kept. PESQ-WB of every source x floor.
C0d error stats per file: median and RMS of 10 log10(xi_dd / xi_oracle) with both clipped
to >= -30 dB, in speech cells (clean cell power within 30 dB of the file maximum) and in
non-speech cells.

Usage: python tools/diagnose_xi.py --data-root DATA --split train --out OUT.csv --subset CSV
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser import CONFIG  # noqa: E402
from denoiser.gains import _dd_prior_snr, band_floor_profile, lsa_gain  # noqa: E402
from denoiser.phase import smooth_track, yin_f0  # noqa: E402
from denoiser.stft import istft, stft  # noqa: E402
from evaluate import pesq_wb  # noqa: E402
from oracle_decompose import current_and_noise  # noqa: E402

FLOORS = ("cur", -25, -30, -40, "none")
SIGMAS = (0, 2, 4, 8)


def floor_vector(name, base_floor, freqs):
    if name == "cur":
        return base_floor
    if name == "none":
        return np.where(freqs < CONFIG.band_cutoff_hz, base_floor, 0.0)
    return np.where(freqs < CONFIG.band_cutoff_hz, base_floor, 10 ** (name / 20))


def harmonic_cells(s, sr, times, n_bins, n_fft):
    centers = np.rint(times * sr).astype(int)
    f0, voiced = smooth_track(*yin_f0(s, sr, centers, CONFIG), CONFIG)
    mask = np.zeros((n_bins, len(times)), bool)
    for t in np.flatnonzero(voiced):
        h = np.arange(f0[t], sr / 2, f0[t])
        k = np.rint(h * n_fft / sr).astype(int)
        for d in (-1, 0, 1):
            kk = k + d
            mask[kk[(kk >= 0) & (kk < n_bins)], t] = True
    return mask


def one_file(args):
    row, data_root = args
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, times, zy = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    _, lam = current_and_noise(y, sr)
    ps, py = np.abs(zs) ** 2, np.abs(zy) ** 2
    gamma = py / lam
    freqs = np.arange(zy.shape[0]) * sr / n_fft
    floor, cap = band_floor_profile(zy.shape[0], sr, n_fft, CONFIG)
    xi_or = ps / lam

    def dd_with(fl_vec):
        fwd = _dd_prior_snr(gamma, lam, zy, CONFIG.lsa_smoothing, fl_vec)
        bwd = _dd_prior_snr(gamma[:, ::-1], lam[:, ::-1], zy[:, ::-1], CONFIG.lsa_smoothing,
                            fl_vec)[:, ::-1]
        return 0.5 * (fwd + bwd)

    xi_dd = dd_with(floor)
    rng = np.random.default_rng(abs(hash(row["segment_id"])) % (2 ** 32))
    oracle = {f"or_s{sg}": xi_or * 10 ** (sg * rng.standard_normal(xi_or.shape) / 10)
              for sg in SIGMAS}
    harm = harmonic_cells(s, sr, times, zy.shape[0], n_fft)
    speech_cells = ps > ps.max() * 1e-3
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"], "harm_cell_share": float(harm.mean())}
    for fl in FLOORS:
        fv = floor_vector(fl, floor, freqs)
        # the DD recursion uses the same floor as the final gain (as in the pipeline)
        xi_dd_fl = xi_dd if fl == "cur" else dd_with(fv)
        sources = dict(oracle, dd=xi_dd_fl)
        if fl in ("cur", "none", -30):
            sources["harm"] = np.where(harm, xi_or, xi_dd_fl)
            sources["or_nonspeech"] = np.where(speech_cells, xi_dd_fl, xi_or)
            sources["or_speech"] = np.where(speech_cells, xi_or, xi_dd_fl)
            sources["fix_over"] = np.minimum(xi_dd_fl, xi_or)
            sources["fix_under"] = np.maximum(xi_dd_fl, xi_or)
            sources["fix_over_6db"] = np.minimum(xi_dd_fl, xi_or * 10 ** 0.6)
        for src, xi in sources.items():
            g = np.minimum(np.maximum(lsa_gain(xi, gamma), fv[:, None]), cap[:, None])
            out[f"{src}|{fl}"] = pesq_wb(s, istft(g * zy, len(y), sr, n_fft, hop), sr)[0]
    err = 10 * np.log10(np.maximum(xi_dd, 1e-3) / np.maximum(xi_or, 1e-3))
    speech = ps > ps.max() * 1e-3
    for name, m in (("speech", speech), ("nonspeech", ~speech)):
        e = err[m]
        out[f"dd_err_median_{name}"] = float(np.median(e)) if e.size else np.nan
        out[f"dd_err_rms_{name}"] = float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
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
    srcs = [f"or_s{sg}" for sg in SIGMAS] + ["dd", "harm", "or_nonspeech", "or_speech",
                                             "fix_over", "fix_under", "fix_over_6db"]
    table = pd.DataFrame({fl: [df[f"{s}|{fl}"].mean() if f"{s}|{fl}" in df else np.nan
                               for s in srcs] for fl in FLOORS}, index=srcs)
    table.round(4).to_csv(args.out.with_name(args.out.stem + "_floor_curve.csv"))
    print("mean PESQ-WB, rows = a priori SNR source, columns = floor")
    print(table.round(3).to_string())
    snr = df.groupby("target_snr_db")[[f"{s}|{fl}" for s in ("dd", "or_s4")
                                       for fl in ("cur", -30, "none")]].mean()
    snr.round(4).to_csv(args.out.with_name(args.out.stem + "_by_snr.csv"))
    print(snr.round(3).to_string())
    err_cols = [c for c in df.columns if c.startswith("dd_err")]
    print(df[err_cols].median().round(2).to_string())
    print("harmonic cell share (median):", round(df.harm_cell_share.median(), 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
