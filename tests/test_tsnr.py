"""Round-03 step B3: two-step a priori SNR (TSNR) switch."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal


def _signal(seed=3, sr=16000):
    rng = np.random.default_rng(seed)
    t = np.arange(sr) / sr
    speech = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.3) * (t < 0.7)
    return speech + 0.05 * rng.standard_normal(sr), sr


class TsnrTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(CONFIG.tsnr)

    def test_changes_output_and_keeps_length(self):
        x, sr = _signal()
        base = denoise_signal(x, sr, CONFIG)
        for first in ("wiener", "lsa"):
            out = denoise_signal(x, sr, replace(CONFIG, tsnr=True, tsnr_first_gain=first))
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertFalse(np.allclose(out, base))

    def test_invalid_first_gain(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, tsnr_first_gain="x")


if __name__ == "__main__":
    unittest.main()
