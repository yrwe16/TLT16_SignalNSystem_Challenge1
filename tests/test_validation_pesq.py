"""PESQ-WB of five fixed Validation files with the current defaults (needs the data).

Runs only when the environment variable TLT16_DATA_ROOT points to the folder that holds
validation/samples.csv, and when the evaluation package `pesq` is installed
(requirements-eval.txt); otherwise the test is skipped. Expected values are in
tests/data/validation_pesq.json (recorded from the r02-base run); tolerance 0.001.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import soundfile as sf

from tests.common import DATA
from denoiser.pipeline import denoise_file

EXPECTED = DATA / "validation_pesq.json"


@unittest.skipUnless(os.environ.get("TLT16_DATA_ROOT"), "TLT16_DATA_ROOT not set")
class ValidationPesq(unittest.TestCase):
    def test_fixed_files(self):
        try:
            from pesq import pesq
        except ImportError:
            self.skipTest("pesq not installed (pip install -r requirements-eval.txt)")
        root = Path(os.environ["TLT16_DATA_ROOT"])
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            for item in expected["files"]:
                with self.subTest(segment=item["segment_id"]):
                    target = Path(tmp) / Path(item["noisy_path"]).name
                    denoise_file(root / item["noisy_path"], target)
                    clean, sr = sf.read(str(root / item["clean_path"]), dtype="float64")
                    enhanced, _ = sf.read(str(target), dtype="float64")
                    score = pesq(sr, clean, enhanced, "wb")
                    self.assertAlmostEqual(score, item["pesq_wb"], delta=0.001)


if __name__ == "__main__":
    unittest.main()
