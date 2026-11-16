"""Round-02 step 11: speech-protected time-varying noise tracker (adopted as default).

- a noise level step in unprotected frames is followed in both directions;
- protected frames (posterior frame SNR above the threshold) are not updated;
- with tracked_floor_db = 0 the estimate never falls below the stationary estimate.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG
from denoiser.noise_estimation import tracked_protected_noise


class TrackedProtected(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.power = rng.exponential(1.0, (257, 300))
        self.power[:, 150:] *= 4.0                    # noise level rises by 6 dB at frame 150
        self.fixed = np.full(257, 2.0)
        self.f0 = np.zeros(300)
        self.voiced = np.zeros(300, bool)
        self.cfg = replace(CONFIG, tracked_tau_s=0.1, tracked_frame_snr_db=10.0,
                           tracked_floor_db=-200.0)

    def run_tracker(self, power, cfg):
        return tracked_protected_noise(power, self.fixed, self.f0, self.voiced,
                                       np.ones(300, bool), 0.008, 16000, 512, cfg)

    def test_follows_level_step(self):
        est = self.run_tracker(self.power, self.cfg)
        early = est[:, 20:130].mean()
        late = est[:, 170:280].mean()
        self.assertGreater(late / early, 2.0)

    def test_protected_frames_not_updated(self):
        power = self.power.copy()
        power[:, 60:80] *= 1000.0                     # loud "speech" frames, frame SNR >> 10 dB
        est = self.run_tracker(power, self.cfg)
        ref = self.run_tracker(self.power, self.cfg)
        self.assertLess(est[:, 60:80].mean() / ref[:, 60:80].mean(), 1.5)

    def test_floor(self):
        est = self.run_tracker(self.power * 0.01, replace(self.cfg, tracked_floor_db=0.0))
        self.assertTrue(np.all(est >= self.fixed[:, None] - 1e-12))


if __name__ == "__main__":
    unittest.main()
