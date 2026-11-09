"""Check the submission I/O rules: every input WAV has an output WAV with the same
relative path, sample rate, channel count and number of samples.

Usage:
    python tools/check_io.py                    # ./input vs ./output
    python tools/check_io.py --input IN --output OUT
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def wavs(folder):
    return {p.relative_to(folder): p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() == ".wav"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "input")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()

    inputs, outputs = wavs(args.input), wavs(args.output)
    problems, warnings = [], []
    if not inputs:
        problems.append("no input WAV files")
    for relative, source in inputs.items():
        target = outputs.get(relative)
        if target is None:
            problems.append(f"missing output: {relative}")
            continue
        try:
            a = sf.info(str(source))
        except Exception as error:  # unreadable input: denoise.py copies it unchanged
            if target.read_bytes() == source.read_bytes():
                warnings.append(f"{relative}: unreadable input, copied unchanged ({error})")
            else:
                problems.append(f"{relative}: unreadable input and output differs")
            continue
        try:
            b = sf.info(str(target))
        except Exception as error:
            problems.append(f"{relative}: unreadable output ({error})")
            continue
        for field in ("samplerate", "channels", "frames"):
            if getattr(a, field) != getattr(b, field):
                problems.append(f"{relative}: {field} {getattr(a, field)} -> {getattr(b, field)}")
        data, _ = sf.read(str(target), dtype="float64", always_2d=True)
        if not np.isfinite(data).all():
            problems.append(f"{relative}: non-finite samples")
    extra = sorted(set(outputs) - set(inputs))
    if extra:
        problems.append(f"{len(extra)} output file(s) without a matching input, e.g. {extra[0]}")

    print(f"inputs: {len(inputs)}, outputs: {len(outputs)}")
    for line in warnings:
        print("WARN", line)
    if problems:
        for line in problems[:50]:
            print("FAIL", line)
        if len(problems) > 50:
            print(f"... and {len(problems) - 50} more")
        return 1
    print("OK: names, sample rates, channels and lengths all match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
