"""Round-02 step 4 part 2: snr_adaptive off equals the defaults; with the 15 dB floor equal to
the band floor it changes nothing; the offset grows linearly between the start SNR and 15 dB."""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.pipeline import denoise_signal, snr_adaptive_offset_db


class SnrAdaptive(unittest.TestCase):
    def test_off_and_neutral_equal_default(self):
        x = synthetic_signal()
        base = denoise_signal(x, SAMPLE_RATE, CONFIG)
        self.assertFalse(CONFIG.snr_adaptive)
        neutral = replace(CONFIG, snr_adaptive=True, snr_adapt_floor15_db=CONFIG.band_mid_floor_db)
        np.testing.assert_array_equal(denoise_signal(x, SAMPLE_RATE, neutral), base)

    def test_offset(self):
        cfg = replace(CONFIG, snr_adaptive=True, snr_adapt_start_db=8.0, snr_adapt_floor15_db=-10.0,
                      band_mid_floor_db=-15.0)   # the expected offsets assume a -15 dB floor
        self.assertEqual(snr_adaptive_offset_db(-3.0, cfg), 0.0)
        self.assertEqual(snr_adaptive_offset_db(8.0, cfg), 0.0)
        self.assertAlmostEqual(snr_adaptive_offset_db(11.5, cfg), 2.5)
        self.assertAlmostEqual(snr_adaptive_offset_db(15.0, cfg), 5.0)
        self.assertAlmostEqual(snr_adaptive_offset_db(30.0, cfg), 5.0)

    def test_changes_output(self):
        x = synthetic_signal()
        y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, snr_adaptive=True, snr_adapt_start_db=0.0,
                                                   snr_adapt_floor15_db=-10.0))
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(np.abs(y - denoise_signal(x, SAMPLE_RATE, CONFIG)).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
