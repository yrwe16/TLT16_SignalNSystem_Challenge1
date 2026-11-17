"""Round-03 step C2: DD combine and a priori SNR cap switches (off by default)."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal
from tests.common import synthetic_signal, SAMPLE_RATE


class XiOverestimateTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual((CONFIG.dd_combine, CONFIG.xi_cap_k), ("mean", 0.0))

    def test_switches_change_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, CONFIG)
        for cfg in (replace(CONFIG, dd_combine="min"), replace(CONFIG, dd_combine="geo"),
                    replace(CONFIG, xi_cap_k=2.0)):
            out = denoise_signal(x, SAMPLE_RATE, cfg)
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertFalse(np.allclose(out, base))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, dd_combine="max")
        with self.assertRaises(ValueError):
            replace(CONFIG, xi_cap_k=-1.0)


if __name__ == "__main__":
    unittest.main()
