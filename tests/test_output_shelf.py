"""Round-03 step R3-5: fixed output band attenuation (adopted in R3-5)."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal, output_shelf_gain
from tests.common import synthetic_signal, SAMPLE_RATE


class OutputShelfTest(unittest.TestCase):
    def test_default_is_adopted_setting(self):
        self.assertEqual((CONFIG.output_shelf, CONFIG.output_shelf_hz, CONFIG.output_shelf_db,
                          CONFIG.output_shelf_smooth_bins), (True, 700.0, -25.0, 0))

    def test_gain_profile(self):
        cfg = replace(CONFIG, output_shelf=True, output_shelf_hz=1000.0, output_shelf_db=-20.0)
        g = output_shelf_gain(257, 16000, 512, cfg)
        freqs = np.arange(257) * 16000 / 512
        np.testing.assert_allclose(g[freqs < 1000], 1.0)
        np.testing.assert_allclose(g[freqs >= 1000], 0.1)

    def test_changes_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, output_shelf=False))
        out = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, output_shelf=True,
                                                     output_shelf_smooth_bins=2))
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(np.allclose(out, base))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, output_shelf_db=3.0)


if __name__ == "__main__":
    unittest.main()
