"""Diagnostics for the round-02 step-8 masking floor (evaluation only; clean labels frames).

For each file the default pipeline is run with masking=True and the speech-frame decision is
captured, with the frames where any gain actually rose. A frame is a clean-speech frame when its clean power is within 25 dB of the file's
loudest clean frame and a pause frame when it is more than 40 dB below (our choice). Reports
how many pause frames get a raised floor (must be ~0) and how many speech frames are covered.

Usage: python tools/diagnose_masking.py --data-root DATA --split train --out OUT.json [--workers N]
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

import denoiser.pipeline as pipeline  # noqa: E402
from denoiser import CONFIG  # noqa: E402
from denoiser.stft import stft  # noqa: E402


def one_file(args):
    row, data_root = args
    noisy, sr = sf.read(str(Path(data_root) / row["noisy_path"]), dtype="float64")
    clean, _ = sf.read(str(Path(data_root) / row["clean_path"]), dtype="float64")
    pipeline.denoise_signal(noisy, sr, replace(CONFIG, masking=True))
    speech_hat = pipeline.apply_masking_floor.speech
    raised_any = pipeline.apply_masking_floor.raised.any(axis=0)   # a bin's gain actually rose
    n_fft, hop = CONFIG.stft_sizes(sr)
    _, _, c = stft(clean, sr, n_fft, hop)
    cp = np.sum(np.abs(c) ** 2, axis=0)
    rel = 10 * np.log10(np.maximum(cp, 1e-30) / max(cp.max(), 1e-30))
    speech, pause = rel >= -25, rel < -40
    return {"segment_id": row["segment_id"], "noise_category": row["noise_category"],
            "target_snr_db": row["target_snr_db"], "frames": len(rel),
            "speech": int(speech.sum()), "pause": int(pause.sum()),
            "flag_speech": int((speech_hat & speech).sum()),
            "flag_pause": int((speech_hat & pause).sum()),
            "flag_mid": int((speech_hat & ~speech & ~pause).sum()),
            "flag_total": int(speech_hat.sum()),
            "raised_speech": int((raised_any & speech).sum()),
            "raised_pause": int((raised_any & pause).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    rows = pd.read_csv(args.data_root / args.split / "samples.csv", encoding="utf-8-sig")
    jobs = [(dict(r), str(args.data_root)) for _, r in rows.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        df = pd.DataFrame(pool.map(one_file, jobs, chunksize=8))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    s = df.sum(numeric_only=True)
    summary = {"split": args.split, "files": len(df),
               "flagged_share_of_speech_frames": float(s.flag_speech / s.speech),
               "flagged_share_of_pause_frames": float(s.flag_pause / max(s.pause, 1)),
               "pause_frames": int(s.pause), "flagged_pause_frames": int(s.flag_pause),
               "files_with_flagged_pause": int((df.flag_pause > 0).sum()),
               "flagged_frames_that_are_speech": float(s.flag_speech / max(s.flag_total, 1)),
               "raised_share_of_speech_frames": float(s.raised_speech / s.speech),
               "raised_pause_frames": int(s.raised_pause),
               "files_with_raised_pause": int((df.raised_pause > 0).sum()),
               "by_type": df.groupby("noise_category")[["flag_pause", "pause", "flag_speech", "speech"]]
                            .sum().assign(pause_rate=lambda t: t.flag_pause / t.pause.clip(lower=1),
                                          speech_rate=lambda t: t.flag_speech / t.speech.clip(lower=1))
                            [["pause_rate", "speech_rate"]].round(4).reset_index().to_dict(orient="records")}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_type"}, indent=1))
    print(pd.DataFrame(summary["by_type"]).sort_values("pause_rate", ascending=False).head(8).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
