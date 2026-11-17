"""Round-03 step R3-7: CASA modulation weighting (adopted in R3-7)."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import casa_mod_weight, denoise_signal
from denoiser.stft import stft
from tests.common import synthetic_signal, SAMPLE_RATE


class CasaModTest(unittest.TestCase):
    def test_default_is_adopted_setting(self):
        self.assertEqual((CONFIG.casa_mod, CONFIG.casa_mod_k, CONFIG.casa_mod_min,
                          CONFIG.casa_mod_ref_pct), (True, 0.75, 0.12, 50.0))

    def test_weight_range(self):
        x = synthetic_signal()
        n_fft, hop = CONFIG.stft_sizes(SAMPLE_RATE)
        _, _, z = stft(x, SAMPLE_RATE, n_fft, hop)
        w = casa_mod_weight(z, SAMPLE_RATE, hop, CONFIG)
        self.assertEqual(w.shape, z.shape)
        self.assertGreaterEqual(w.min(), CONFIG.casa_mod_min ** CONFIG.casa_mod_k - 1e-12)
        self.assertLessEqual(w.max(), 1.0)
        low = np.arange(z.shape[0]) * SAMPLE_RATE / n_fft < 80
        np.testing.assert_array_equal(w[low], 1.0)

    def test_changes_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, casa_mod=False, casa_extra=False))
        out = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, casa_extra=False))
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(np.allclose(out, base))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, casa_mod_min=0.0)
        with self.assertRaises(ValueError):
            replace(CONFIG, casa_mod_ref_pct=150.0)


if __name__ == "__main__":
    unittest.main()
