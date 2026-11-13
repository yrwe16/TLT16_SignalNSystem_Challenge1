"""Submission I/O rules on temporary WAV files (no data needed): same file names and
relative paths, 16-bit PCM kept, same sample rate, channel count and length, and no
clipped samples (even for a loud input)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tests.common import SAMPLE_RATE, synthetic_signal
from denoiser.pipeline import denoise_folder


class InputOutputRules(unittest.TestCase):
    def test_folder_io(self):
        x = synthetic_signal()
        loud = np.clip(x / np.abs(x).max() * 0.999, -1, 1)
        cases = {"a_noisy.wav": (x, SAMPLE_RATE), "loud.WAV": (loud, SAMPLE_RATE),
                 "sub/stereo.wav": (np.column_stack([x, loud]), SAMPLE_RATE),
                 "short_8k.wav": (x[:4000], 8000)}
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "input", Path(tmp) / "output"
            for name, (signal, sr) in cases.items():
                path = inp / name
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), signal, sr, subtype="PCM_16")
            files, failures = denoise_folder(inp, out, log=lambda *a: None)
            self.assertEqual(failures, [])
            self.assertEqual(len(files), len(cases))
            for name in cases:
                with self.subTest(file=name):
                    src, dst = sf.info(str(inp / name)), out / name
                    self.assertTrue(dst.is_file())
                    info = sf.info(str(dst))
                    self.assertEqual(info.subtype, "PCM_16")
                    self.assertEqual(info.samplerate, src.samplerate)
                    self.assertEqual(info.channels, src.channels)
                    self.assertEqual(info.frames, src.frames)
                    data, _ = sf.read(str(dst), dtype="int16", always_2d=True)
                    clipped = int(np.sum((data == 32767) | (data == -32768)))
                    self.assertEqual(clipped, 0)


if __name__ == "__main__":
    unittest.main()
