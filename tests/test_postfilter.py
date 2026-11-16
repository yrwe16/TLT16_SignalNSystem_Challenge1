"""Round-02 step 6: musical-noise postfilter (Esch & Vary, ICASSP 2009).

- frames with zeta >= zeta_thr (eq. 6) are left exactly as they are;
- the window length follows eq. (7) and the smoothing is a moving average (eqs. 8-9);
- the mean squared gain of a frame never rises; an isolated gain peak is spread out;
- the switch is off by default and gives finite output of the right length when on.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.gains import musical_noise_postfilter
from denoiser.pipeline import denoise_signal


class Postfilter(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.power = rng.exponential(1.0, (257, 40))
        self.gain = np.full((257, 40), 0.18)
        self.gain[100, :20] = 1.0          # isolated peaks in quiet frames
        self.gain[:, 20:] = 0.9            # loud frames (zeta ~ 0.81)

    def test_length_and_untouched_frames(self):
        out, n, zeta = musical_noise_postfilter(self.gain, self.power, 0.4, 7)
        np.testing.assert_array_equal(out[:, 20:], self.gain[:, 20:])
        self.assertTrue(np.all(n[20:] == 1))
        expected = 2 * np.floor((1 - zeta[:20] / 0.4) * 7 + 0.5).astype(int) + 1
        np.testing.assert_array_equal(n[:20], expected)
        self.assertTrue(np.all((n[:20] >= 3) & (n[:20] <= 15)))

    def test_energy_and_peak(self):
        out, n, _ = musical_noise_postfilter(self.gain, self.power, 0.4, 7)
        self.assertTrue(np.all(np.mean(out ** 2, axis=0) <= np.mean(self.gain ** 2, axis=0) + 1e-12))
        e_out = np.sum(out ** 2 * self.power, axis=0)
        e_in = np.sum(self.gain ** 2 * self.power, axis=0)
        self.assertTrue(np.all(e_out <= e_in * (1 + 1e-12)))
        self.assertTrue(np.all(out[100, :20] < 0.5))
        frame = 0
        half = n[frame] // 2
        average = np.array([np.mean(self.gain[k - half:k + half + 1, frame]) for k in (40, 98, 100)])
        ratio = out[[40, 98, 100], frame] / average   # moving average up to one scale <= 1
        np.testing.assert_allclose(ratio, ratio[0])
        self.assertLessEqual(ratio[0], 1.0)

    def test_gain_above_one_not_spread(self):
        gain = self.gain.copy()
        gain[150, :20] = 25.0               # LSA gain > 1 in a bin with tiny |Y|
        power = self.power.copy()
        power[150, :20] = 1e-6
        out, n, _ = musical_noise_postfilter(gain, power, 0.4, 7)
        self.assertTrue(np.all(n[:20] > 1))
        self.assertTrue(np.all(out[:, :20] <= 1.0))

    def test_psi_zero_is_identity(self):
        out, n, _ = musical_noise_postfilter(self.gain, self.power, 0.4, 0)
        np.testing.assert_array_equal(out, self.gain)
        self.assertTrue(np.all(n == 1))

    def test_pipeline(self):
        self.assertFalse(CONFIG.postfilter)
        x = synthetic_signal()
        y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, postfilter=True, postfilter_psi=7))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())
        same = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, postfilter=True, postfilter_psi=0))
        np.testing.assert_allclose(same, denoise_signal(x, SAMPLE_RATE), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
