"""Round-03 step R3-6: harmonic regeneration (HRNR), not adopted; switch stays off."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal
from tests.common import synthetic_signal, SAMPLE_RATE


class HrnrTest(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(CONFIG.hrnr)

    def test_changes_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, CONFIG)
        for nl, gain, mix in (("max", "lsa", 1.0), ("abs", "wiener", 0.5)):
            out = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, hrnr=True, hrnr_nl=nl,
                                                         hrnr_gain=gain, hrnr_mix=mix))
            self.assertEqual(out.shape, x.shape)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertFalse(np.allclose(out, base))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, hrnr_nl="square")
        with self.assertRaises(ValueError):
            replace(CONFIG, hrnr_mix=1.5)


if __name__ == "__main__":
    unittest.main()
