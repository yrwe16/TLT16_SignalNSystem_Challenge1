"""Round-02 step 1: band_floor at its default parameters must reproduce the round-01
wind_module (325 Hz, -15 dB) output exactly, and band_floor off must equal the defaults."""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.pipeline import denoise_signal


class BandFloor(unittest.TestCase):
    def test_defaults_equal_wind_module(self):
        x = synthetic_signal()
        wind = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, wind_module=True))
        band = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, band_floor=True))
        np.testing.assert_array_equal(band, wind)

    def test_off_equals_default(self):
        x = synthetic_signal()
        self.assertFalse(CONFIG.band_floor)
        np.testing.assert_array_equal(
            denoise_signal(x, SAMPLE_RATE, replace(CONFIG, band_floor=False)),
            denoise_signal(x, SAMPLE_RATE, CONFIG))

    def test_settings_change_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, band_floor=True))
        for kw in ({"band_low_depth_db": -25.0}, {"band_high_floor_db": -18.0},
                   {"band_cutoff_hz": 200.0}, {"band_smooth_bins": 2}):
            with self.subTest(**kw):
                y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, band_floor=True, **kw))
                self.assertTrue(np.isfinite(y).all())
                self.assertGreater(np.abs(y - base).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
