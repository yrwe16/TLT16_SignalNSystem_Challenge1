"""Round-02 step 8: masking-threshold gain floor (Virag, IEEE TSAP 1999).

- spreading function: 0 dB at dz = 0, about -100 dB at dz = -5 (Fig. 3a);
- 22 critical bands at 16 kHz (Sec. III-A);
- the floor equals the base floor in non-speech frames and stays within [base, base + range];
- the switch is off by default and gives finite output of the right length when on.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.masking import band_index, masking_floor, masking_threshold, speech_frames, spreading_db
from denoiser.pipeline import denoise_signal


class Masking(unittest.TestCase):
    def test_spreading_and_bands(self):
        self.assertLess(abs(spreading_db(0.0)), 0.05)
        self.assertLess(spreading_db(-5.0), -95)
        self.assertGreater(spreading_db(5.0), spreading_db(-5.0))   # upward spread is wider
        self.assertEqual(band_index(257, 16000, 512).max() + 1, 22)

    def test_floor_limits(self):
        rng = np.random.default_rng(0)
        first = rng.exponential(1.0, (257, 30)) * 30
        noisy = first + 1.0
        noise = np.ones((257, 30))
        noisy[:, :10] = 1.0                       # frames 0-9: noise only (0 dB frame SNR)
        speech = speech_frames(noisy, noise, 10.0)
        self.assertFalse(speech[:10].any())
        self.assertTrue(speech[10:].all())
        cfg = replace(CONFIG, masking=True, masking_range_db=6.0, masking_offset_db=0.0)
        base = 10 ** (-15 / 20)
        floor = masking_floor(first, noise, speech, base, 16000, 512, cfg)
        np.testing.assert_array_equal(floor[:, ~speech], base)
        self.assertTrue(np.all(floor >= base))
        self.assertTrue(np.all(floor <= base * 10 ** (6 / 20) + 1e-12))
        self.assertTrue(np.any(floor[:, speech] > base))
        t = masking_threshold(first, 16000, 512)
        self.assertEqual(t.shape, first.shape)
        self.assertTrue(np.all(t > 0))

    def test_pipeline(self):
        self.assertFalse(CONFIG.masking)
        x = synthetic_signal()
        y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, masking=True))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())
        z = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, masking=True, masking_range_db=0.0))
        self.assertEqual(z.tobytes(), denoise_signal(x, SAMPLE_RATE).tobytes())   # no raise = same bytes


if __name__ == "__main__":
    unittest.main()
