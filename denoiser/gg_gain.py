"""Round-02 step 5: MMSE amplitude estimators under a generalized gamma speech prior.

J. S. Erkelens, R. C. Hendriks, R. Heusdens, J. Jensen, "Minimum Mean-Square Error Estimation
of Discrete Fourier Coefficients With Generalized Gamma Priors", IEEE TASLP 15(6):1741-1752,
2007, DOI 10.1109/TASL.2007.899233 (read in the original). Notation: xi = a priori SNR,
zeta = a posteriori SNR (our code calls it gamma; here `gamma` is the prior's shape class).

Amplitude prior (1): f_A(a) = gamma beta^nu / Gamma(nu) a^(gamma nu - 1) exp(-beta a^gamma).

gamma = 2, eq. (12), exact:
    G = Gamma(nu+1/2)/Gamma(nu) sqrt(xi / (zeta (nu+xi)))
        1F1(nu+1/2; 1; zeta xi/(nu+xi)) / 1F1(nu; 1; zeta xi/(nu+xi))
    nu = 1 is the Rayleigh prior, i.e. Ephraim & Malah's MMSE-STSA (1984).
gamma = 1, combined estimator of Sec. IV-C, G_C,K = max(G_<<,K, G_>>), K = 5 as in Fig. 5,
with mu = sqrt(nu (nu+1)):
    eq. (16): G_<<,K = 1/sqrt(2 zeta) sum_k (1/k!)^2 (zeta/2)^k Gamma(nu+2k+1) D_-(nu+2k+1)(z)
                                    / sum_k (1/k!)^2 (zeta/2)^k Gamma(nu+2k)   D_-(nu+2k)(z),
              z = mu / sqrt(2 xi), k = 0..K-1
    eq. (18): G_>> = (nu - 1/2)/sqrt(2 zeta) D_-(nu+1/2)(w) / D_-(nu-1/2)(w),
              w = mu / sqrt(2 xi) - sqrt(2 zeta), valid for nu > 1/2
The parabolic-cylinder arguments were re-derived from eq. (14) with Gradshteyn & Ryzhik
3.462.1 (the PDF text loses subscripts). For speed the gain is tabulated from these formulas
on a (xi, zeta) grid in 0.1 dB steps and interpolated bilinearly in log gain, as the paper suggests (Sec. IV-E);
xi and zeta are clipped to [-30, 30] dB to stay inside the range where the special functions
are finite in double precision (our choice).
"""

from functools import lru_cache

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammaln, hyp1f1, pbdv

GRID_DB = np.arange(-30.0, 30.0 + 1e-9, 0.1)


def _ratio_1f1(a, x):
    """1F1(a+1/2; 1; x) / 1F1(a; 1; x), using Kummer's transformation for large x."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 50
    out[small] = hyp1f1(a + 0.5, 1.0, x[small]) / hyp1f1(a, 1.0, x[small])
    big = ~small
    # 1F1(a; b; x) = e^x 1F1(b-a; b; -x)  ->  the e^x factors cancel in the ratio
    out[big] = hyp1f1(0.5 - a, 1.0, -x[big]) / hyp1f1(1.0 - a, 1.0, -x[big])
    return out


def gain_gamma2(xi, zeta, nu):
    """Eq. (12) divided by r."""
    xi, zeta = np.asarray(xi, float), np.asarray(zeta, float)
    x = zeta * xi / (nu + xi)
    pre = np.exp(gammaln(nu + 0.5) - gammaln(nu))
    return pre * np.sqrt(xi / (zeta * (nu + xi))) * _ratio_1f1(nu, x)


def gain_gamma1_low(xi, zeta, nu, k_terms=5):
    """Eq. (16) divided by r."""
    xi, zeta = np.asarray(xi, float), np.asarray(zeta, float)
    z = np.sqrt(nu * (nu + 1)) / np.sqrt(2 * xi)
    num = np.zeros_like(z)
    den = np.zeros_like(z)
    for k in range(k_terms):
        c = (zeta / 2) ** k / gamma_fn(k + 1) ** 2
        num += c * gamma_fn(nu + 2 * k + 1) * pbdv(-(nu + 2 * k + 1), z)[0]
        den += c * gamma_fn(nu + 2 * k) * pbdv(-(nu + 2 * k), z)[0]
    return num / (den * np.sqrt(2 * zeta))


def gain_gamma1_high(xi, zeta, nu):
    """Eq. (18) divided by r (nu > 1/2)."""
    xi, zeta = np.asarray(xi, float), np.asarray(zeta, float)
    w = np.sqrt(nu * (nu + 1)) / np.sqrt(2 * xi) - np.sqrt(2 * zeta)
    return (nu - 0.5) / np.sqrt(2 * zeta) * pbdv(-(nu + 0.5), w)[0] / pbdv(-(nu - 0.5), w)[0]


def gain_gamma1(xi, zeta, nu, k_terms=5):
    """Combined estimator G_C,K = max(G_<<,K, G_>>) of Sec. IV-C."""
    low = gain_gamma1_low(xi, zeta, nu, k_terms)
    if nu <= 0.5:
        return low
    return np.fmax(low, gain_gamma1_high(xi, zeta, nu))


def gain_exact(xi, zeta, gamma, nu):
    """Closed-form / combined gain for gamma in {1, 2} (no table)."""
    return gain_gamma2(xi, zeta, nu) if gamma == 2 else gain_gamma1(xi, zeta, nu)


@lru_cache(maxsize=16)
def gain_table(gamma, nu):
    """Log gain on GRID_DB x GRID_DB (rows xi, columns zeta), from the formulas above."""
    xi = 10 ** (GRID_DB[:, None] / 10)
    zeta = 10 ** (GRID_DB[None, :] / 10)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        table = gain_exact(np.broadcast_to(xi, (len(GRID_DB),) * 2),
                           np.broadcast_to(zeta, (len(GRID_DB),) * 2), gamma, nu)
    bad = ~np.isfinite(table) | (table <= 0)
    if bad.any():
        # At the lowest a priori SNRs the parabolic-cylinder functions underflow (0/0). Those
        # gains are far below any gain floor; take the value of the nearest valid xi row above
        # (the gain grows with xi). Tables without such cells are left untouched.
        for j in np.flatnonzero(bad.any(axis=0)):
            valid = np.flatnonzero(~bad[:, j])
            if not len(valid):
                raise FloatingPointError(f"no valid gain for gamma={gamma}, nu={nu}")
            for i in np.flatnonzero(bad[:, j]):
                table[i, j] = table[valid[np.argmin(np.abs(valid - i))], j]
    return np.log(table)


def gg_gain(xi, zeta, gamma, nu):
    """Interpolated gain for arrays of a priori / a posteriori SNR (linear power ratios)."""
    table = gain_table(gamma, nu)
    step = GRID_DB[1] - GRID_DB[0]
    n = len(GRID_DB)

    def index(v):
        db = 10 * np.log10(np.maximum(v, 1e-30))
        pos = (np.clip(db, GRID_DB[0], GRID_DB[-1]) - GRID_DB[0]) / step
        lo = np.minimum(np.floor(pos).astype(int), n - 2)
        return lo, pos - lo

    i, fi = index(xi)
    j, fj = index(zeta)
    return np.exp((1 - fi) * (1 - fj) * table[i, j] + fi * (1 - fj) * table[i + 1, j]
                  + (1 - fi) * fj * table[i, j + 1] + fi * fj * table[i + 1, j + 1])
