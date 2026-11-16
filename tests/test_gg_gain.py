"""Round-02 step 5: generalized gamma amplitude estimators (Erkelens et al., IEEE TASLP 2007).

- gamma = 2, nu = 1 equals the Ephraim-Malah MMSE-STSA gain (the paper's Rayleigh special case).
- gamma = 1 combined estimator stays within the paper's reported error range of the exact
  MMSE gain (numerical integration of eq. (14)): at most ~0.2 dB above, and below it only by a
  limited amount (paper: up to 3.7 dB at the smallest nu; we allow 2.5 dB for nu >= 0.6).
- the interpolated table is within 0.15 dB of the formulas;
- gain_rule="gg" gives finite output of the right length; defaults are unchanged (LSA).
"""

import unittest
from dataclasses import replace

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e, i1e

from tests.common import CONFIG, SAMPLE_RATE, synthetic_signal
from denoiser.gg_gain import gain_exact, gain_gamma1, gain_gamma2, gg_gain
from denoiser.pipeline import denoise_signal


def exact_gamma1(xi, zeta, nu):
    mu = np.sqrt(nu * (nu + 1))

    def f(y, p):
        return np.exp(p * np.log(y) - y ** 2 / (4 * zeta) - mu * y / (2 * np.sqrt(zeta * xi))
                      + np.log(i0e(y)) + y)
    hi = 60 * np.sqrt(zeta) + 60
    num = quad(lambda y: f(y, nu), 0, hi, limit=400, points=[1, 10])[0]
    den = quad(lambda y: f(y, nu - 1), 0, hi, limit=400, points=[1, 10])[0]
    return num / den / (2 * zeta)


class GeneralizedGamma(unittest.TestCase):
    def test_gamma2_nu1_is_stsa(self):
        xi = 10 ** (np.array([-10, -5, 0, 5, 15]) / 10)[:, None]
        zeta = 10 ** (np.array([-5, 0, 5, 10, 20]) / 10)[None, :]
        v = xi * zeta / (1 + xi)
        stsa = np.sqrt(np.pi) / 2 * np.sqrt(v) / zeta * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
        np.testing.assert_allclose(gain_gamma2(xi, zeta, 1.0), stsa, rtol=1e-10)

    def test_gamma1_combined_error_range(self):
        for nu in (0.6, 1.0, 2.0):
            for xd in (-20, -5, 5, 15):
                for zd in (-10, 0, 5, 14):
                    with self.subTest(nu=nu, xi_db=xd, zeta_db=zd):
                        x, z = 10 ** (xd / 10), 10 ** (zd / 10)
                        err = 20 * np.log10(gain_gamma1(np.array(x), np.array(z), nu)
                                            / exact_gamma1(x, z, nu))
                        self.assertLess(err, 0.2)
                        self.assertGreater(err, -2.5)

    def test_table_interpolation(self):
        rng = np.random.default_rng(0)
        xi = 10 ** (rng.uniform(-29, 29, 2000) / 10)
        zeta = 10 ** (rng.uniform(-29, 29, 2000) / 10)
        for g, nu in ((1, 0.6), (2, 0.1)):
            with self.subTest(gamma=g, nu=nu):
                err = 20 * np.log10(gg_gain(xi, zeta, g, nu) / gain_exact(xi, zeta, g, nu))
                self.assertLess(np.abs(err).max(), 0.15)

    def test_pipeline(self):
        x = synthetic_signal()
        self.assertEqual(CONFIG.gain_rule, "lsa")
        for g, nu in ((1, 0.6), (2, 0.2)):
            with self.subTest(gamma=g, nu=nu):
                y = denoise_signal(x, SAMPLE_RATE, replace(CONFIG, gain_rule="gg", gg_gamma=g, gg_nu=nu))
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
