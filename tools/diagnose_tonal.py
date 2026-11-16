"""Diagnostics for the round-02 step-7 tonal module (evaluation only; clean is used only here).

Per file:
- false detection on CLEAN speech: share of time-frequency bins masked by the 6a tracker and by
  the revised tracker (tonal_mode="r02", per grid setting) when run on the clean signal;
- on the noisy input, per r02 setting: masked share, whether any track is kept (module "on"),
  and precision = share of masked bins where the true noise power (|STFT(noisy - clean)|^2)
  exceeds the clean power (low precision = false detection on speech, low share on siren files
  = missed tracks).

Usage: python tools/diagnose_tonal.py --data-root DATA --split train --out OUT.json
       [--peak 8 10] [--unvoiced 0.3 0.5] [--workers N]
"""

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denoiser import CONFIG  # noqa: E402
from denoiser.noise_estimation import tonal_tracks, tonal_tracks_r02  # noqa: E402
from denoiser.pipeline import tonal_f0  # noqa: E402
from denoiser.stft import stft  # noqa: E402


def one_file(args):
    row, data_root, peaks, unvoiced = args
    noisy, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    clean, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, times, zn = stft(noisy, sr, n_fft, hop)
    _, _, zc = stft(clean, sr, n_fft, hop)
    _, _, zd = stft(noisy - clean, sr, n_fft, hop)
    pn, pc, pd_ = np.abs(zn) ** 2, np.abs(zc) ** 2, np.abs(zd) ** 2
    hop_s = hop / sr
    out = {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
           "target_snr_db": row["target_snr_db"]}
    old_cfg = replace(CONFIG, tonal_module=True, tonal_mode="6a")
    out["clean_6a_share"] = float(tonal_tracks(pc, sr, n_fft, hop_s, old_cfg).mean())
    out["noisy_6a_share"] = float(tonal_tracks(pn, sr, n_fft, hop_s, old_cfg).mean())
    fc = tonal_f0(clean, sr, times, CONFIG)
    fn = tonal_f0(noisy, sr, times, CONFIG)
    out["voiced_share_noisy"] = float(np.mean(fn[1]))
    for p in peaks:
        for u in unvoiced:
            cfg = replace(CONFIG, tonal_module=True, tonal_mode="r02", tonal_peak_db=p,
                          tonal_unvoiced_min=u)
            key = f"p{p:g}_u{u:g}"
            mc, _ = tonal_tracks_r02(pc, *fc, sr, n_fft, hop_s, cfg)
            mn, kept = tonal_tracks_r02(pn, *fn, sr, n_fft, hop_s, cfg)
            out[f"{key}_clean_share"] = float(mc.mean())
            out[f"{key}_noisy_share"] = float(mn.mean())
            out[f"{key}_tracks"] = len(kept)
            out[f"{key}_precision"] = float(np.mean(pd_[mn] > pc[mn])) if mn.any() else np.nan
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--peak", type=float, nargs="+", default=[8.0, 10.0])
    parser.add_argument("--unvoiced", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    jobs = [(dict(r), str(args.data_root), args.peak, args.unvoiced) for _, r in rows.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_file, jobs, chunksize=8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    siren = df.noise_category.isin(["alarm_siren", "bell", "phone_dialing"])
    summary = {"split": args.split, "files": len(df),
               "clean_false_share_6a": df.clean_6a_share.mean(),
               "clean_files_with_detection_6a": float((df.clean_6a_share > 0).mean()),
               "settings": {}}
    for p in args.peak:
        for u in args.unvoiced:
            key = f"p{p:g}_u{u:g}"
            on = df[f"{key}_tracks"] > 0
            per_type = df.assign(on=on).groupby("noise_category").agg(
                files=("on", "size"), on_rate=("on", "mean"),
                share=(f"{key}_noisy_share", "mean"), precision=(f"{key}_precision", "mean"))
            summary["settings"][key] = {
                "clean_false_share": df[f"{key}_clean_share"].mean(),
                "clean_files_with_detection": float((df[f"{key}_clean_share"] > 0).mean()),
                "on_rate_target_types": float(on[siren].mean()),
                "on_rate_other_types": float(on[~siren].mean()),
                "precision_target_types": float(df.loc[siren & on, f"{key}_precision"].mean()),
                "precision_other_types": float(df.loc[~siren & on, f"{key}_precision"].mean()),
                "per_type": per_type.round(4).reset_index().to_dict(orient="records")}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=1))
    for k, v in summary["settings"].items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()
                  if a != "per_type"})
        t = pd.DataFrame(v["per_type"]).set_index("noise_category")
        print(t.loc[[c for c in ("alarm_siren", "bell", "phone_dialing", "machine", "wind")
                     if c in t.index]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
