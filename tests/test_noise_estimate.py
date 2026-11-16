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



class QuantileWindow(unittest.TestCase):
    """Round-02 step 3: the bias-corrected sliding-window quantile matches the true power of
    stationary white noise and follows a 12 dB level step."""

    def _estimate(self, x, **kw):
        # noise_beta_high pinned to 1 (the R2-15 default 1.25 scales the band above 4 kHz)
        cfg = replace(CONFIG, noise_estimator="quantile_window", noise_beta_high=1.0, **kw)
        n_fft, hop = cfg.stft_sizes(SAMPLE_RATE)
        _, times, z = stft(x, SAMPLE_RATE, n_fft, hop)
        return estimate_noise_psd(z, times, len(x), SAMPLE_RATE, n_fft, hop, cfg)

    def test_unbiased_on_white_noise(self):
        x = np.random.default_rng(7).standard_normal(SAMPLE_RATE * 3) * 0.1
        n_fft = CONFIG.stft_sizes(SAMPLE_RATE)[0]
        true = 0.01 * np.hanning(n_fft + 1)[:-1].__pow__(2).sum() / (np.hanning(n_fft + 1)[:-1].sum() ** 2)
        for q in (0.1, 0.2, 0.3):
            for w in (0.5, 1.0):
                with self.subTest(q=q, window=w):
                    est = self._estimate(x, noise_quantile=q, noise_window_s=w, noise_combine="mix",
                                         noise_mix_weight=1.0)
                    ratio_db = 10 * np.log10(est[5:-5, 40:-40].mean() / true)
                    self.assertLess(abs(ratio_db), 0.5)

    def test_follows_level_step(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal(SAMPLE_RATE * 3) * 0.02
        x[SAMPLE_RATE * 3 // 2:] *= 4.0  # +12 dB in the second half
        est = self._estimate(x, noise_quantile=0.2, noise_window_s=0.5, noise_combine="mix",
                             noise_mix_weight=1.0)
        first, second = est[5:-5, 20:120].mean(), est[5:-5, -120:-20].mean()
        self.assertGreater(10 * np.log10(second / first), 9.0)

    def test_combine_modes_finite(self):
        x = synthetic_signal()
        for combine in ("max", "mix", "level"):
            with self.subTest(combine=combine):
                y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, noise_estimator="quantile_window",
                                                           noise_combine=combine))
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
