"""Output regression on a fixed synthetic signal (no data needed).

The default configuration and the stage-0 Wiener baseline must reproduce the stored
reference outputs (tests/data/synthetic_*.npy) within 1e-6. Regenerate them with
tests/make_reference.py only when an output change is intended.
"""

import unittest

import numpy as np

from tests.common import CONFIGS, SAMPLE_RATE, reference_path, synthetic_signal
from denoiser.pipeline import denoise_signal


class SyntheticRegression(unittest.TestCase):
    def test_outputs_match_reference(self):
        x = synthetic_signal()
        for name, cfg in CONFIGS.items():
            with self.subTest(config=name):
                ref = np.load(reference_path(name))
                y = denoise_signal(x, SAMPLE_RATE, cfg)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.isfinite(y).all())
                np.testing.assert_allclose(y, ref, rtol=0, atol=1e-6)

    def test_silence_stays_silent(self):
        for name, cfg in CONFIGS.items():
            with self.subTest(config=name):
                y = denoise_signal(np.zeros(SAMPLE_RATE), SAMPLE_RATE, cfg)
                self.assertEqual(np.abs(y).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
