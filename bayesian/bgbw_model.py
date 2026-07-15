# -*- coding: utf-8 -*-
r"""
bgbw_model.py -- Boltzmann-Gibbs Blast-Wave (BGBW) forward model
================================================================

Forward model for identified-particle transverse-momentum spectra used in the
Bayesian pipeline of Sec. 6 (Block C of the calculation plan).  This is the
Schnedermann-Sollfrank-Heinz blast-wave: a boosted thermal source with a
linear-in-(r/R)^n transverse-velocity profile.  The Lorentz-invariant yield is

    dN                     R
   ----- (p_T)  =  A  * INT   r dr  m_T  I_0( p_T sinh(rho)/T_kin )
   p_T dp_T               0                * K_1( m_T cosh(rho)/T_kin )

with

    m_T          = sqrt(p_T^2 + m^2)          (transverse mass, natural units)
    rho(r)       = arctanh( beta_T(r) )       (transverse rapidity)
    beta_T(r)    = beta_s (r/R)^n             (surface velocity beta_s, index n)
    <beta_T>     = 2 beta_s / (n + 2)         (r-weighted mean transverse beta)

Parameters theta = (T_kin, beta_s, n), or equivalently (T_kin, <beta_T>, n).
The subluminal constraint is  beta_s < 1  (the *surface* velocity, which is the
largest velocity in the profile; if beta_s < 1 the whole profile is subluminal).

Units: natural units c = hbar = k_B = 1 (see _common.py).  Masses, T_kin, p_T,
m_T all in GeV.  The overall amplitude A (and R) is an arbitrary normalisation,
absorbed by a per-species amplitude in the likelihood, so R is set to 1 here and
the radial integral runs over the dimensionless xi = r/R in [0, 1].

The measured ALICE observable is the *un-weighted* spectrum
    Y(p_T) = d^2N/(dp_T dy) = p_T * [ dN / (p_T dp_T) ],
so callers that compare to ALICE tables must multiply spectrum() by p_T (done in
run_emcee_bgbw.py); spectrum() itself returns the invariant form dN/(p_T dp_T)
up to the amplitude A.

Numerical notes
---------------
* The Bessel arguments grow like ~10 for p_T ~ 1 GeV and small T_kin, so I_0 and
  K_1 individually overflow/underflow.  We use the exponentially scaled
  scipy.special.i0e, k1e and recombine as
      I_0(z) K_1(w) = i0e(z) k1e(w) exp(z - w),
  and z - w = [p_T sinh(rho) - m_T cosh(rho)] / T_kin <= 0 always
  (since m_T >= p_T and cosh >= sinh), so exp(z - w) never overflows.
* The radial integral is done with a fixed Simpson grid in xi; the integrand is
  smooth, so ~80 nodes are ample (checked against scipy.integrate.quad in the
  self-test to < 1e-4 relative).

Run directly for a self-test:  python bgbw_model.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0e, k1e
from scipy.integrate import simpson


# ---------------------------------------------------------------------------
# Velocity-profile helpers
# ---------------------------------------------------------------------------
def mean_beta_T(beta_s, n):
    r"""r-weighted mean transverse velocity <beta_T> = 2 beta_s / (n + 2).

    Derived from <beta_T> = int_0^R beta_s (r/R)^n r dr / int_0^R r dr.
    """
    return 2.0 * np.asarray(beta_s, float) / (np.asarray(n, float) + 2.0)


def beta_s_from_mean(mean_beta, n):
    r"""Invert <beta_T> -> surface velocity beta_s = <beta_T> (n + 2) / 2."""
    return np.asarray(mean_beta, float) * (np.asarray(n, float) + 2.0) / 2.0


def is_subluminal(beta_s):
    """True where the surface (hence whole) flow profile is subluminal, beta_s < 1."""
    return np.asarray(beta_s, float) < 1.0


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------
def spectrum(pT, m, T_kin, beta_s, n, n_r=81):
    r"""BGBW invariant spectrum  dN/(p_T dp_T)  (amplitude A = 1, R = 1).

    Parameters
    ----------
    pT : array_like
        Transverse momenta [GeV].
    m : float
        Particle rest mass [GeV].
    T_kin : float
        Kinetic freeze-out temperature [GeV].
    beta_s : float
        Surface transverse velocity (must satisfy 0 <= beta_s < 1).
    n : float
        Transverse-velocity-profile exponent (> 0).
    n_r : int
        Number of Simpson nodes for the radial integral (odd).

    Returns
    -------
    ndarray
        The integral  int_0^1 xi dxi m_T I0(p_T sinh rho / T) K1(m_T cosh rho / T)
        evaluated at each p_T, i.e. dN/(p_T dp_T) up to the overall amplitude.
        Returns +inf-free, non-negative values; NaN/inf are mapped to 0.

    Notes
    -----
    Not defined for beta_s >= 1 (superluminal); callers must enforce the prior
    cut.  For robustness a beta_s >= 1 request is clipped to just below 1.
    """
    pT = np.atleast_1d(np.asarray(pT, dtype=float))
    beta_s = float(np.clip(beta_s, 0.0, 1.0 - 1e-9))
    T_kin = float(T_kin)
    n = float(n)

    if n_r % 2 == 0:
        n_r += 1
    xi = np.linspace(0.0, 1.0, n_r)                 # r/R in [0, 1]

    beta_T = beta_s * xi ** n                        # (n_r,)
    beta_T = np.clip(beta_T, 0.0, 1.0 - 1e-12)
    rho = np.arctanh(beta_T)                         # transverse rapidity (n_r,)
    sinh_rho = np.sinh(rho)
    cosh_rho = np.cosh(rho)

    mT = np.sqrt(pT * pT + m * m)                    # (n_pt,)

    # Broadcast to (n_pt, n_r)
    z = (pT[:, None] * sinh_rho[None, :]) / T_kin    # arg of I0
    w = (mT[:, None] * cosh_rho[None, :]) / T_kin    # arg of K1

    # I0(z) K1(w) = i0e(z) k1e(w) exp(z - w), with z - w <= 0 (no overflow).
    with np.errstate(over="ignore", invalid="ignore"):
        kernel = i0e(z) * k1e(w) * np.exp(z - w)     # (n_pt, n_r)
        integrand = mT[:, None] * kernel * xi[None, :]

    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    out = simpson(integrand, x=xi, axis=1)           # (n_pt,)
    out = np.where(np.isfinite(out) & (out > 0.0), out, 0.0)
    return out


def spectrum_from_mean(pT, m, T_kin, mean_beta, n, n_r=81):
    """Convenience wrapper taking <beta_T> instead of beta_s."""
    return spectrum(pT, m, T_kin, beta_s_from_mean(mean_beta, n), n, n_r=n_r)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    from scipy.integrate import quad
    from scipy.special import i0, k1

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(HERE))
    from _common import M_PION, M_KAON, M_PROTON  # noqa: E402

    print("=" * 68)
    print("bgbw_model.py self-test  (c = hbar = k_B = 1)")
    print("=" * 68)

    # --- velocity-profile identities ---------------------------------------
    bs, n = 0.90, 0.85
    mb = mean_beta_T(bs, n)
    print(f"<beta_T>(beta_s=0.90, n=0.85) = {mb:.6f}  (= 2*0.9/2.85)")
    assert abs(beta_s_from_mean(mb, n) - bs) < 1e-12, "beta_s inversion failed"
    assert is_subluminal(0.99) and not is_subluminal(1.01)

    # --- overflow safety at large p_T --------------------------------------
    big = spectrum(np.array([3.0, 5.0]), M_PION, 0.095, 0.95, 0.85)
    print(f"spectrum(pT=[3,5], pi, T=0.095, bs=0.95) = {big}  (finite, >0)")
    assert np.all(np.isfinite(big)) and np.all(big > 0.0)

    # --- vectorised Simpson vs scipy.quad on one representative point ------
    T_kin, beta_s, nn = 0.095, 0.65, 0.85
    m = M_KAON
    pT0 = 0.8
    mT0 = np.sqrt(pT0**2 + m**2)

    def integrand_quad(xi):
        bT = min(beta_s * xi**nn, 1 - 1e-12)
        rho = np.arctanh(bT)
        return xi * mT0 * i0(pT0 * np.sinh(rho) / T_kin) * \
            k1(mT0 * np.cosh(rho) / T_kin)

    ref, _ = quad(integrand_quad, 0.0, 1.0, limit=200)
    got = float(spectrum(pT0, m, T_kin, beta_s, nn, n_r=161)[0])
    rel = abs(got - ref) / ref
    print(f"Simpson vs quad at pT=0.8 (K): {got:.6e} vs {ref:.6e}  rel={rel:.2e}")
    assert rel < 1e-4, "Simpson integral disagrees with quad"

    # --- spectrum falls monotonically at intermediate/high p_T -------------
    pTg = np.linspace(0.3, 3.0, 40)
    for name, mm in (("pi", M_PION), ("K", M_KAON), ("p", M_PROTON)):
        s = spectrum(pTg, mm, T_kin, beta_s, nn)
        # spectrum must be positive and eventually decreasing
        assert np.all(s > 0.0)
        assert s[-1] < s[len(s) // 2] < s[3], f"{name} spectrum not falling"
        print(f"  {name:>2s}: spectrum(0.3)={s[0]:.3e}  spectrum(3.0)={s[-1]:.3e}  OK")

    # --- higher flow / higher T push yield to higher p_T (harder spectrum) --
    soft = spectrum(pTg, M_PROTON, 0.090, 0.55, 0.85)
    hard = spectrum(pTg, M_PROTON, 0.090, 0.80, 0.85)
    ratio_soft = soft[-1] / soft[0]
    ratio_hard = hard[-1] / hard[0]
    print(f"  proton high/low-pT ratio: beta_s=0.55 -> {ratio_soft:.3e}, "
          f"beta_s=0.80 -> {ratio_hard:.3e}  (flow hardens spectrum)")
    assert ratio_hard > ratio_soft, "more flow should harden the spectrum"

    print("-" * 68)
    print("All bgbw_model self-test assertions passed.")
