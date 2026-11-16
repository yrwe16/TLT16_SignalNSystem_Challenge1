"""Round-02 step 9: two-stage transient estimate (transient_mode="two_stage", not adopted).

- clicks in white noise give a transient PSD at the click frames and none elsewhere;
- stationary white noise alone flags no frames once the stationary floor is used;
- the module is off by default and the pipeline gives finite output when on.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.noise_estimation import two_stage_transient
from denoiser.pipeline import denoise_signal

CFG = replace(CONFIG, transient_module=True, transient_mode="two_stage", transient_delta=6.0,
              transient_file_index=0.0)


class TwoStage(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.noise = 0.01 * rng.standard_normal(3 * SAMPLE_RATE)
        self.n_fft, self.hop = CONFIG.stft_sizes(SAMPLE_RATE)
        self.frames = 3 * SAMPLE_RATE // self.hop + 1
        self.flat = np.full(self.n_fft // 2 + 1, 0.0001 * 1.5 / self.n_fft)  # white-noise PSD

    def test_clicks(self):
        x = self.noise.copy()
        for c in (8000, 24000, 40000):
            x[c:c + 40] += 0.5 * np.random.default_rng(c).standard_normal(40)   # broadband burst
        extra, index = two_stage_transient(x, SAMPLE_RATE, (self.n_fft // 2 + 1, self.frames),
                                           self.n_fft, self.hop, CFG, self.flat)
        self.assertIsNotNone(extra)
        power = extra.sum(axis=0)
        click_frames = [c // self.hop for c in (8000, 24000, 40000)]
        quiet = np.ones(self.frames, bool)
        for f in click_frames:
            quiet[max(f - 8, 0):f + 9] = False
            self.assertGreater(power[f - 2:f + 3].max(), 0)
        self.assertEqual(power[quiet].max(), 0.0)

    def test_stationary_noise_not_flagged(self):
        _, index = two_stage_transient(self.noise, SAMPLE_RATE, (self.n_fft // 2 + 1, self.frames),
                                       self.n_fft, self.hop, CFG, self.flat)
        self.assertLess(index, 0.005)

    def test_pipeline(self):
        self.assertFalse(CONFIG.transient_module)
        x = synthetic_signal()
        y = denoise_signal(x, SAMPLE_RATE, CFG)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
