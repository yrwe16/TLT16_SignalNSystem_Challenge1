"""Round-03 step E8: modulation-domain Kalman a priori SNR (adopted in R3-E8)."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal
from tests.common import synthetic_signal, SAMPLE_RATE


class MdkfTest(unittest.TestCase):
    def test_default_is_adopted_setting(self):
        self.assertEqual((CONFIG.mdkf, CONFIG.mdkf_order, CONFIG.mdkf_r0, CONFIG.mdkf_mix),
                         (True, 3, 100.0, 0.5))

    def test_changes_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, mdkf=False))
        for order in (1, 2, 3):
            out = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, mdkf=True, mdkf_order=order))
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertFalse(np.allclose(out, base))

    def test_mix_zero_is_dd(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, mdkf=False))
        out = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, mdkf=True, mdkf_mix=0.0))
        np.testing.assert_allclose(out, base, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
