"""Round-02 step 7: revised narrowband tracker (tonal_mode="r02").

- a tone sweeping at 1.5 kHz/s through unvoiced frames is tracked frame by frame;
- peaks at multiples of a voiced f0 are excluded; nothing below tonal_r02_fmin is tracked;
- the module is on by default since R2-7 and the pipeline gives finite output of the right length.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.noise_estimation import tonal_tracks_r02
from denoiser.pipeline import denoise_signal
from denoiser.stft import stft

CFG = replace(CONFIG, tonal_module=True, tonal_mode="r02")


def sweep(f_start=1000.0, rate=1500.0, seconds=1.5):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    phase = 2 * np.pi * (f_start * t + 0.5 * rate * t ** 2)
    rng = np.random.default_rng(0)
    return 0.1 * np.sin(phase) + 0.001 * rng.standard_normal(t.size)


class TonalR02(unittest.TestCase):
    def setUp(self):
        n_fft, hop = CONFIG.stft_sizes(SAMPLE_RATE)
        self.n_fft, self.hop = n_fft, hop
        _, _, z = stft(sweep(), SAMPLE_RATE, n_fft, hop)
        self.power = np.abs(z) ** 2
        self.frames = z.shape[1]

    def test_sweep_tracked(self):
        unvoiced = (np.zeros(self.frames), np.zeros(self.frames, bool))
        mask, kept = tonal_tracks_r02(self.power, *unvoiced, SAMPLE_RATE, self.n_fft,
                                      self.hop / SAMPLE_RATE, CFG)
        self.assertGreaterEqual(len(kept), 1)
        longest = max(kept, key=len)
        self.assertGreater((longest[-1][0] - longest[0][0]) * self.hop / SAMPLE_RATE, 1.0)
        bins = np.array([k for _, k in longest])
        self.assertGreater(bins[-1] - bins[0], 50)   # follows the sweep (~70 bins)

    def test_voiced_guard_and_fmin(self):
        f0 = np.full(self.frames, 125.0)               # every peak of the sweep is near 8 x ...
        voiced = np.ones(self.frames, bool)
        mask, kept = tonal_tracks_r02(self.power, f0, voiced, SAMPLE_RATE, self.n_fft,
                                      self.hop / SAMPLE_RATE, CFG)
        self.assertEqual(len(kept), 0)                 # all frames voiced -> no track kept
        low = replace(CFG, tonal_r02_fmin=4000.0)
        unvoiced = (np.zeros(self.frames), np.zeros(self.frames, bool))
        mask, _ = tonal_tracks_r02(self.power, *unvoiced, SAMPLE_RATE, self.n_fft,
                                   self.hop / SAMPLE_RATE, low)
        self.assertFalse(mask.any())                   # the sweep stays below 4 kHz

    def test_pipeline(self):
        self.assertTrue(CONFIG.tonal_module and CONFIG.tonal_mode == "r02")  # adopted R2-7
        x = synthetic_signal()
        x[: len(sweep())] += sweep()
        y = denoise_signal(x, SAMPLE_RATE, CFG)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
