"""Round-02 step 2: noise estimate variants.

- Defaults (noise_estimator="lowenergy", all betas 1.0) leave the output unchanged.
- beta_low only changes bins below 325 Hz: in the estimate (applied after the 3-bin
  smoothing) and in the enhanced STFT (the decision-directed recursion is per bin).
- The band-frame and SPP-weighted estimators are finite and positive.
"""

import unittest
from dataclasses import replace

import numpy as np

from tests.common import CONFIG, SAMPLE_RATE, reference_path, synthetic_signal
from denoiser.gains import lsa_with_prior
from denoiser.pipeline import denoise_signal, estimate_noise_psd
from denoiser.stft import stft


def spectrum(cfg):
    x = synthetic_signal()
    n_fft, hop = cfg.stft_sizes(SAMPLE_RATE)
    _, times, z = stft(x, SAMPLE_RATE, n_fft, hop)
    return x, n_fft, hop, times, z


class NoiseEstimate(unittest.TestCase):
    def test_defaults_unchanged(self):
        y = denoise_signal(synthetic_signal(), SAMPLE_RATE, CONFIG)
        np.testing.assert_allclose(y, np.load(reference_path("default")), rtol=0, atol=1e-6)

    def test_beta_low_stays_below_325_hz(self):
        x, n_fft, hop, times, z = spectrum(CONFIG)
        base = estimate_noise_psd(z, times, len(x), SAMPLE_RATE, n_fft, hop, CONFIG)
        cfg = replace(CONFIG, noise_beta_low=10.0)
        scaled = estimate_noise_psd(z, times, len(x), SAMPLE_RATE, n_fft, hop, cfg)
        low = np.arange(len(base)) * SAMPLE_RATE / n_fft < 325.0
        np.testing.assert_array_equal(scaled[~low], base[~low])
        np.testing.assert_allclose(scaled[low], 10.0 * base[low])
        # enhanced STFT: bins at or above 325 Hz must be identical
        e_base = lsa_with_prior(z, base, SAMPLE_RATE, hop / SAMPLE_RATE, CONFIG)
        e_scaled = lsa_with_prior(z, scaled, SAMPLE_RATE, hop / SAMPLE_RATE, CONFIG)
        np.testing.assert_array_equal(e_scaled[~low], e_base[~low])
        self.assertGreater(np.abs(e_scaled[low] - e_base[low]).max(), 0)

    def test_variants_finite_positive(self):
        x, n_fft, hop, times, z = spectrum(CONFIG)
        for kw in ({"noise_estimator": "band_frames", "noise_band_fraction": 0.1},
                   {"noise_estimator": "spp_iter", "noise_spp_iterations": 3},
                   {"noise_beta": 1.5, "noise_beta_high": 0.8}):
            with self.subTest(**kw):
                n = estimate_noise_psd(z, times, len(x), SAMPLE_RATE, n_fft, hop,
                                       replace(CONFIG, **kw))
                self.assertTrue(np.isfinite(n).all() and (n > 0).all())
                y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, **kw))
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
