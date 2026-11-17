"""Round-03 Phase C2-0: how much does a speech presence decision on top of the current output
gain, at frame or cell level, oracle vs realistic? (analysis only; the oracles use clean)

The current output E (defaults) is re-analysed with the same STFT and multiplied by an extra
attenuation in cells judged speech-free (the low band stays as it is: it is capped already).
  frame oracle   : speech frame if the clean frame power (bins >= the low-band cut-off) is within
                   40 dB of the file maximum (our choice)
  cell oracle    : speech cell if the clean cell power is within 30 dB of the file maximum
                   (as in diagnose_xi.py)
  realistic frame VADs (noisy input only):
    v_yin        : YIN voiced frames of the first pass (pipeline.first_pass_f0), dilated by
                   +-12 frames (~100 ms, our choice) to keep onsets and unvoiced neighbours
    v_snr{3,6}   : mean posterior SNR (|Y|^2 / lambda, lambda = the final noise estimate of the
                   current pipeline) over 300-4000 Hz above 3 / 6 dB, dilated by +-4 frames
    v_or{3,6}    : v_yin OR v_snr
Attenuations: -10, -20, -40 dB. Reports PESQ-WB and frame precision / recall of each VAD
against the frame oracle (positive class = speech).

Usage: python tools/diagnose_presence.py --data-root DATA --split train --out OUT.csv --subset CSV
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
from scipy.ndimage import binary_dilation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import denoiser.pipeline as pipeline  # noqa: E402
from denoiser import CONFIG  # noqa: E402
from denoiser.stft import istft, stft  # noqa: E402
from evaluate import pesq_wb  # noqa: E402
from oracle_decompose import current_and_noise  # noqa: E402

ATTEN_DB = (-10, -20, -40)


def dilate(v, n):
    return binary_dilation(v, structure=np.ones(2 * n + 1, bool)) if n else v


def one_file(args):
    row, data_root = args
    y, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    s, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, times, zy = stft(y, sr, n_fft, hop)
    _, _, zs = stft(s, sr, n_fft, hop)
    e, lam = current_and_noise(y, sr)
    _, _, ze = stft(e, sr, n_fft, hop)
    freqs = np.arange(zy.shape[0]) * sr / n_fft
    above = freqs >= CONFIG.band_cutoff_hz
    ps = np.abs(zs) ** 2
    fp = ps[above].sum(axis=0)
    vads = {"frame_or": fp > fp.max() * 1e-4}
    cell_speech = ps > ps.max() * 1e-3
    _, voiced = pipeline.first_pass_f0(y, sr, times, CONFIG)
    v_yin = dilate(voiced, 12)
    mid = (freqs >= 300) & (freqs < 4000)
    post = 10 * np.log10(np.mean(np.abs(zy[mid]) ** 2 / lam[mid], axis=0) + 1e-12)
    vads["v_yin"] = v_yin
    for th in (3, 6):
        vads[f"v_snr{th}"] = dilate(post > th, 4)
        vads[f"v_or{th}"] = v_yin | vads[f"v_snr{th}"]
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"], "current": pesq_wb(s, e, sr)[0],
           "speech_frame_share": float(vads["frame_or"].mean())}
    ref = vads["frame_or"]
    for name, v in vads.items():
        if name != "frame_or":
            out[f"{name}_prec"] = float((v & ref).sum() / max(v.sum(), 1))
            out[f"{name}_rec"] = float((v & ref).sum() / max(ref.sum(), 1))
        for a in ATTEN_DB:
            g = np.ones(ze.shape)
            g[np.ix_(above, ~v)] = 10 ** (a / 20)
            out[f"{name}|{a}"] = pesq_wb(s, istft(g * ze, len(y), sr, n_fft, hop), sr)[0]
    for a in ATTEN_DB:
        g = np.where(cell_speech | ~above[:, None], 1.0, 10 ** (a / 20))
        out[f"cell_or|{a}"] = pesq_wb(s, istft(g * ze, len(y), sr, n_fft, hop), sr)[0]
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
    names = ["frame_or", "cell_or", "v_yin", "v_snr3", "v_snr6", "v_or3", "v_or6"]
    table = pd.DataFrame({f"{a} dB": [df[f"{n}|{a}"].mean() for n in names] for a in ATTEN_DB},
                         index=names)
    table["precision"] = [df[f"{n}_prec"].mean() if f"{n}_prec" in df else np.nan for n in names]
    table["recall"] = [df[f"{n}_rec"].mean() if f"{n}_rec" in df else np.nan for n in names]
    table.round(4).to_csv(args.out.with_name(args.out.stem + "_summary.csv"))
    print(f"current {df.current.mean():.4f}; speech frame share {df.speech_frame_share.mean():.3f}")
    print(table.round(3).to_string())
    snr = df.groupby("target_snr_db")[["current", "frame_or|-20", "cell_or|-20", "v_or3|-20"]].mean()
    print(snr.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
