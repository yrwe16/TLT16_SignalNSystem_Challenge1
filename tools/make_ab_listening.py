"""Blind A/B listening set: two configurations, randomly labelled X / Y per file.

Usage:
    python tools/make_ab_listening.py --data-root PATH --out listening/NAME \
        --a "r02-base" --b "low-cut-off:wind_module=false" [--split validation]
        [--list results/listening_set.txt] [--seed 0]

Each side is "LABEL" or "LABEL:KEY=VALUE,KEY=VALUE" with denoiser Config overrides (as in
evaluate.py --set). For every segment of the list the output folder gets
    NN_<segment>_0_noisy.wav, NN_<segment>_X.wav, NN_<segment>_Y.wav, NN_<segment>_9_clean.wav
and a sheet listening_sheet.csv (preference X / Y / same, comments) for the listeners.
Which side is X is decided per file with the seed and written ONLY to ANSWERS.json, kept
next to the folder (OUT_answers.json), so the sheet can be filled before looking.
WAV files stay local: listening/ is in .gitignore.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from denoiser.pipeline import denoise_file  # noqa: E402
from evaluate import parse_overrides  # noqa: E402


def parse_side(text):
    label, _, rest = text.partition(":")
    items = [s.strip() for s in rest.split(",") if s.strip()]
    return label.strip(), items, parse_overrides(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--list", type=Path, default=ROOT / "results" / "listening_set.txt")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sides = [parse_side(args.a), parse_side(args.b)]
    manifest = pd.read_csv(args.data_root / args.split / "samples.csv").set_index("segment_id")
    chosen = pd.read_csv(args.list, comment="#")
    rng = np.random.default_rng(args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    answers, sheet = [], []
    for i, seg in enumerate(chosen.segment_id, start=1):
        row = manifest.loc[seg]
        stem = f"{i:02d}_{seg}"
        shutil.copy(args.data_root / row.noisy_path, args.out / f"{stem}_0_noisy.wav")
        shutil.copy(args.data_root / row.clean_path, args.out / f"{stem}_9_clean.wav")
        order = rng.permutation(2)  # order[0] -> X, order[1] -> Y
        for letter, side in zip("XY", order):
            denoise_file(args.data_root / row.noisy_path, args.out / f"{stem}_{letter}.wav",
                         sides[side][2])
        answers.append({"no": i, "segment_id": seg, "noise_category": row.noise_category,
                        "target_snr_db": float(row.target_snr_db),
                        "X": sides[order[0]][0], "Y": sides[order[1]][0]})
        sheet.append({"no": i, "segment_id": seg, "noise_category": row.noise_category,
                      "target_snr_db": row.target_snr_db,
                      "preference (X/Y/same)": "", "comment": ""})

    with open(args.out / "listening_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    key = args.out.with_name(args.out.name + "_answers.json")
    key.write_text(json.dumps({"seed": args.seed, "split": args.split,
                               "sides": {label: items for label, items, _ in sides},
                               "files": answers}, indent=2), encoding="utf-8")
    print(f"{len(sheet)} files -> {args.out}  (answer key: {key})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
