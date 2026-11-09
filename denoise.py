"""Denoise every WAV file in the input folder and save results to the output folder.

Usage:
    python denoise.py                       # ./input -> ./output
    python denoise.py --input IN --output OUT
"""

import argparse
import sys
import time
from pathlib import Path

from denoiser import CONFIG, denoise_folder

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Speech noise reduction (Wiener filter).")
    parser.add_argument("--input", type=Path, default=ROOT / "input",
                        help="folder with noisy WAV files (default: ./input)")
    parser.add_argument("--output", type=Path, default=ROOT / "output",
                        help="folder for denoised WAV files (default: ./output)")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Input folder not found: {args.input}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        files, failures = denoise_folder(args.input, args.output, CONFIG)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    if not files:
        print(f"No WAV files found in {args.input}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - started
    print(f"Done: {len(files)} files -> {args.output.resolve()} ({elapsed:.1f}s)")
    if failures:
        print(f"WARNING: {len(failures)} file(s) could not be processed and were copied unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
