# -*- coding: utf-8 -*-
"""
_common.py -- Shared utilities for all figure and calculation scripts.

Code and data for:
    "Statistical Distributions for Relativistic Particles: A Unified Framework
     from Maxwell-Juttner to Non-Extensive Generalizations with Bayesian
     Parameter Inference"  (Annals of Physics, Elsevier).

This module centralises everything the individual figNN_*.py / calculation
scripts share, so that no rcParams or physics constant is duplicated:

    * Physical constants in NATURAL UNITS (c = hbar = k_B = 1) plus the SI
      conversion factors actually needed (GeV, MeV, fm, K).
    * Relativistic thermodynamic helpers:
        - coldness / "cold z":  zeta = m / T          (z_cold)
        - fugacity:             z    = exp(mu / T)
        - Macdonald / modified Bessel wrappers  K_n(x)  (scipy.special.kv)
    * Deformed exponentials:
        - Tsallis q-exponential   exp_q(x, q)
        - Kaniadakis kappa-exp    exp_k(x, kappa)
    * Relativistic density of states  dos(eps, m, D).
    * Plotting helpers:
        - apply_style()          load the canonical style.mplstyle
        - savefig_dual(fig, name) write BOTH figures/<name>.pdf and .png

Run directly for a self-test:

    python _common.py
"""

from __future__ import annotations

import os

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Paths (resolved relative to THIS file so scripts work from any CWD)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(HERE, "style.mplstyle")
FIG_DIR = os.path.join(HERE, "figures")

# ===========================================================================
# 1. Physical constants
# ===========================================================================
# Natural units: c = hbar = k_B = 1.  Energies/masses/temperatures in GeV,
# lengths/times in GeV^-1.  In these units everything is a power of energy.

C_LIGHT = 1.0        # speed of light
HBAR = 1.0           # reduced Planck constant
K_B = 1.0            # Boltzmann constant

# --- SI / practical conversion factors --------------------------------------
# hbar * c  = 0.1973269804 GeV * fm  (CODATA-consistent).  This is the single
# bridge between an energy in GeV and a length in fm.
HBARC_GEV_FM = 0.1973269804          # GeV * fm
GEV = 1.0                            # base energy unit
MEV = 1.0e-3                         # 1 MeV in GeV
KEV = 1.0e-6
TEV = 1.0e3
FM = 1.0 / HBARC_GEV_FM              # 1 fm expressed in GeV^-1  (~5.0677 GeV^-1)
FM_INV = HBARC_GEV_FM               # 1 fm^-1 expressed in GeV   (~0.19733 GeV)

# Convenience particle masses (GeV) used across the figures.
M_PION = 0.13957039    # charged pion mass  m_pi
M_KAON = 0.493677      # charged kaon mass
M_PROTON = 0.938272088 # proton mass

# Kelvin <-> GeV (k_B = 1): 1 GeV corresponds to 1.160451812e13 K.
GEV_PER_KELVIN = 8.617333262e-14     # GeV / K
KELVIN_PER_GEV = 1.0 / GEV_PER_KELVIN


# ===========================================================================
# 2. Relativistic thermodynamic helpers
# ===========================================================================
def coldness(m, T):
    """Coldness (a.k.a. 'cold z'): z_cold = m / T = m c^2 / (k_B T).

    Dimensionless relativistic parameter; large => non-relativistic regime,
    small => ultra-relativistic.  Note this is the *inverse* of the
    temperature parameter theta = k_B T / (m c^2).
    """
    m = np.asarray(m, dtype=float)
    T = np.asarray(T, dtype=float)
    return m / T


def theta(m, T):
    """Relativistic temperature parameter theta = k_B T / (m c^2) = T / m."""
    m = np.asarray(m, dtype=float)
    T = np.asarray(T, dtype=float)
    return T / m


def fugacity(mu, T):
    """Fugacity z = exp(mu / (k_B T)) = exp(mu / T)."""
    mu = np.asarray(mu, dtype=float)
    T = np.asarray(T, dtype=float)
    return np.exp(mu / T)


def bessel_k(n, x):
    """Modified Bessel function of the second kind K_n(x) (Macdonald function).

    Thin wrapper over scipy.special.kv so every script uses one convention.
    Integer or real order ``n`` is accepted.
    """
    return special.kv(n, x)


def bessel_k_ratio(n, m, x):
    """Ratio K_n(x) / K_m(x), evaluated stably via the exponentially scaled
    Bessel functions ``kve`` (the exp(x) factors cancel).  Useful for the
    Maxwell-Juttner thermodynamics (e.g. K_3/K_2, K_1/K_2) at large x.
    """
    return special.kve(n, x) / special.kve(m, x)


# ===========================================================================
# 3. Deformed exponentials
# ===========================================================================
def exp_q(x, q):
    """Tsallis q-exponential.

        exp_q(x) = [1 + (1 - q) x]_+^{1 / (1 - q)},      q != 1
                 = exp(x),                                q  = 1

    The base is clipped at zero (Tsallis cut-off): where 1 + (1-q)x <= 0 the
    q-exponential is defined to be 0.  Reduces to the ordinary exponential as
    q -> 1 and satisfies exp_q(0) = 1 for every q.
    """
    x = np.asarray(x, dtype=float)
    if np.isclose(q, 1.0):
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    base = np.where(base > 0.0, base, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.power(base, 1.0 / (1.0 - q))
    return np.where(base > 0.0, out, 0.0)


def exp_k(x, kappa):
    """Kaniadakis kappa-exponential.

        exp_kappa(x) = ( sqrt(1 + kappa^2 x^2) + kappa x )^{1 / kappa},  kappa != 0
                     = exp(x),                                            kappa  = 0

    ``kappa`` is the DIMENSIONLESS entropic index (|kappa| < 1); the argument
    x = beta (E - mu) is itself dimensionless.  Satisfies exp_k(0) = 1, the
    reciprocity exp_k(x) exp_k(-x) = 1, and exp_k -> exp as kappa -> 0.
    """
    x = np.asarray(x, dtype=float)
    if np.isclose(kappa, 0.0):
        return np.exp(x)
    return np.power(np.sqrt(1.0 + kappa**2 * x**2) + kappa * x, 1.0 / kappa)


def ln_q(y, q):
    """Inverse of exp_q: the Tsallis q-logarithm ln_q(y) = (y^{1-q} - 1)/(1-q)."""
    y = np.asarray(y, dtype=float)
    if np.isclose(q, 1.0):
        return np.log(y)
    return (np.power(y, 1.0 - q) - 1.0) / (1.0 - q)


# ===========================================================================
# 4. Relativistic density of states
# ===========================================================================
def dos(eps, m, D=3):
    """Relativistic single-particle density of states Omega(eps) in D dims.

        Omega(eps) ~ eps * (eps^2 - m^2)^{(D-2)/2},     eps >= m   (c = 1)

    (Overall V, hbar and angular prefactors are dropped -- scripts that need
    an absolute normalisation multiply in their own constants.)  Returns 0
    below the rest-mass threshold eps < m.  Limits: NR  Omega ~ eps^{D/2-1},
    UR  Omega ~ eps^{D-1}.
    """
    eps = np.asarray(eps, dtype=float)
    inside = eps * eps - m * m
    val = np.where(inside > 0.0, eps * np.power(np.clip(inside, 0.0, None),
                                                (D - 2.0) / 2.0), 0.0)
    return np.where(eps >= m, val, 0.0)


# ===========================================================================
# 5. Plotting helpers
# ===========================================================================
def apply_style():
    """Load the canonical style.mplstyle for consistent, CB-safe figures.

    Returns the matplotlib.pyplot module for convenience:

        plt = apply_style()
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if os.path.exists(STYLE_PATH):
        plt.style.use(STYLE_PATH)
    else:  # pragma: no cover - defensive; style ships with the repo
        import warnings
        warnings.warn(f"style.mplstyle not found at {STYLE_PATH!r}; "
                      "using matplotlib defaults.")
    return plt


def savefig_dual(fig, name, fig_dir=FIG_DIR, **kwargs):
    """Save ``fig`` as BOTH a vector PDF and a raster PNG preview.

    Writes ``<fig_dir>/<name>.pdf`` and ``<fig_dir>/<name>.png`` (300 dpi from
    style.mplstyle).  ``name`` may include or omit an extension -- any given
    extension is stripped.  Returns the list of paths written.
    """
    os.makedirs(fig_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(name))[0]
    paths = []
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


# ===========================================================================
# 6. Self-test
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("_common.py self-test  (c = hbar = k_B = 1)")
    print("=" * 60)

    # Deformed exponentials at the origin must equal 1.
    v_q = float(exp_q(0.0, 1.1))
    v_k = float(exp_k(0.0, 0.3))
    print(f"exp_q(0, q=1.1)         = {v_q:.12f}   (expect 1.0)")
    print(f"exp_k(0, kappa=0.3)     = {v_k:.12f}   (expect 1.0)")

    # q -> 1 and kappa -> 0 reduce to the ordinary exponential.
    print(f"exp_q(0.5, q=1.0)       = {float(exp_q(0.5, 1.0)):.12f}   "
          f"(expect e^0.5 = {np.exp(0.5):.12f})")
    print(f"exp_k(0.5, kappa=0.0)   = {float(exp_k(0.5, 0.0)):.12f}   "
          f"(expect e^0.5 = {np.exp(0.5):.12f})")

    # Kaniadakis reciprocity  exp_k(x) exp_k(-x) = 1.
    recip = float(exp_k(0.7, 0.3) * exp_k(-0.7, 0.3))
    print(f"exp_k(0.7)*exp_k(-0.7)  = {recip:.12f}   (expect 1.0)")

    # Bessel wrappers vs known values.
    print(f"K_2(1)                  = {float(bessel_k(2, 1.0)):.10f}   "
          f"(scipy.special.kv, expect 1.6248388986)")
    print(f"K_1(1)                  = {float(bessel_k(1, 1.0)):.10f}")
    print(f"K_3(1)/K_2(1)           = {float(bessel_k_ratio(3, 2, 1.0)):.10f}")

    # Thermodynamic helpers.
    print(f"coldness(m=0.14, T=0.1) = {float(coldness(0.14, 0.1)):.6f}   (= m/T)")
    print(f"fugacity(mu=0, T=0.1)   = {float(fugacity(0.0, 0.1)):.6f}   (expect 1.0)")

    # Density of states threshold behaviour.
    ds = dos(np.array([0.05, 0.14, 0.5, 1.0]), m=0.14, D=3)
    print(f"dos([0.05,0.14,0.5,1.0], m=0.14, D=3) = "
          f"[{ds[0]:.4g}, {ds[1]:.4g}, {ds[2]:.4g}, {ds[3]:.4g}]  "
          f"(first two = 0 below/at threshold)")

    # Unit-conversion sanity: hbar*c.
    print(f"HBARC = {HBARC_GEV_FM} GeV*fm ; 1 fm = {FM:.6f} GeV^-1")

    # Assertions (fail loudly if anything regresses).
    assert abs(v_q - 1.0) < 1e-12
    assert abs(v_k - 1.0) < 1e-12
    assert abs(recip - 1.0) < 1e-12
    assert abs(float(bessel_k(2, 1.0)) - 1.6248388986) < 1e-8
    assert ds[0] == 0.0 and ds[1] == 0.0 and ds[2] > 0.0
    print("-" * 60)
    print("All self-test assertions passed.")
