"""Round-03 step R3-8: extra CASA cues (slow modulation, frame harmonicity), adopted in R3-8."""

import unittest
from dataclasses import replace

import numpy as np

from denoiser import CONFIG
from denoiser.pipeline import denoise_signal, frame_harmonicity
from tests.common import synthetic_signal, SAMPLE_RATE


class CasaExtraTest(unittest.TestCase):
    def test_default_is_adopted_setting(self):
        self.assertEqual((CONFIG.casa_extra, CONFIG.casa_slow_k, CONFIG.casa_harm_k,
                          CONFIG.casa_extra_min), (True, 0.5, 0.5, 0.12))

    def test_harmonicity(self):
        t = np.arange(16000) / SAMPLE_RATE
        tone = np.sin(2 * np.pi * 150 * t)
        noise = np.random.default_rng(0).standard_normal(t.size)
        centers = np.arange(2000, 14000, 1000)
        self.assertGreater(frame_harmonicity(tone, centers).min(), 0.8)
        self.assertLess(frame_harmonicity(noise, centers).max(), 0.4)

    def test_changes_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, casa_extra=False))
        out = denoise_signal(x, SAMPLE_RATE, CONFIG)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(np.allclose(out, base))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            replace(CONFIG, casa_extra_min=0.0)
        with self.assertRaises(ValueError):
            replace(CONFIG, casa_slow_k=-1.0)


if __name__ == "__main__":
    unittest.main()
