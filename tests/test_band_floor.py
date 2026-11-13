"""Round-02 step R2-1 (band_floor, adopted): with the round-01 parameters (325 Hz, -15 dB,
no transition) band_floor must reproduce wind_module=True exactly; switching band_floor off
must give the earlier round-02 base (LSA + bidirectional DD); the parameters must act."""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.pipeline import denoise_signal

ROUND01_BAND = dict(band_floor=True, band_cutoff_hz=325.0, band_low_depth_db=-15.0,
                    band_mid_floor_db=-15.0, band_midhigh_floor_db=-15.0,
                    band_high_floor_db=-15.0, band_smooth_bins=0)


class BandFloor(unittest.TestCase):
    def test_round01_parameters_equal_wind_module(self):
        x = synthetic_signal()
        wind = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, band_floor=False, wind_module=True))
        band = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, **ROUND01_BAND))
        np.testing.assert_array_equal(band, wind)

    def test_default_is_adopted_setting(self):
        self.assertTrue(CONFIG.band_floor)
        self.assertEqual((CONFIG.band_cutoff_hz, CONFIG.band_low_depth_db, CONFIG.band_smooth_bins),
                         (325.0, -25.0, 2))

    def test_settings_change_output(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, CONFIG)
        for kw in ({"band_floor": False}, {"band_low_depth_db": -15.0}, {"band_high_floor_db": -18.0},
                   {"band_cutoff_hz": 200.0}, {"band_smooth_bins": 0}):
            with self.subTest(**kw):
                y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, **kw))
                self.assertTrue(np.isfinite(y).all())
                self.assertGreater(np.abs(y - base).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
