# -*- coding: utf-8 -*-
r"""
bec_critical_temperature.py
===========================

Critical temperature of Bose--Einstein condensation for a *relativistic ideal
charged* Bose gas (Haber--Weldon), at fixed **net charge density**

        n  =  n_particles  -  n_antiparticles   (conserved),

as a function of the boson mass m.  This is the calculation behind Fig. 11 and
resolves risk R07 / red-flag O3 (the sign / direction of the relativistic shift
of T_c).

Physics
-------
For a relativistic ideal gas of charged bosons the particles carry chemical
potential +mu and the antiparticles -mu (the antiparticle is the CP conjugate;
its charge is opposite, so its chemical potential is opposite).  With
E(p) = sqrt(p^2 + m^2) the net charge density is (natural units c = hbar = kB = 1,
spin/internal degeneracy g)

    n(T, mu) = (g / 2 pi^2) * int_0^inf dp p^2
                 [ 1/(exp((E-mu)/T) - 1)  -  1/(exp((E+mu)/T) - 1) ].         (1)

The single-particle spectrum has its minimum at E = m, so the Bose occupation of
the particle branch is positive only for mu <= m.  Condensation sets in when the
chemical potential reaches the boundary of its allowed range,

        mu = mu_c = m                                                          (2)

(the relativistic replacement of the non-relativistic mu_c = 0; the ground-state
energy is the rest mass, not zero).  The critical temperature T_c(m, n) is the
temperature at which (1), evaluated at mu = m, equals the prescribed net charge
density n.  We solve that constraint numerically with the full Bose integrals.

Asymptotic limits (derived in the companion note bec_tc_note.tex)
-----------------------------------------------------------------
Writing a = m / T, Eq. (1) at mu = m reads n = (g T^3 / 2 pi^2) J(a), with
J(a) the dimensionless integral evaluated below.

  * Non-relativistic (m >> T_c, a >> 1): antiparticles are exponentially
    suppressed and one recovers the standard single-species result
        n = g zeta(3/2) (m T / 2 pi)^{3/2}
    =>  T_c^{NR} = (2 pi / m) ( n / (g zeta(3/2)) )^{2/3}   ~  m^{-1}.         (3)

  * Ultra-relativistic (m << T_c, a << 1): pair production dominates the census
    but the *net* charge is carried by the linear-in-mu response,
        n = (g / 3) mu T^2  = (g / 3) m T_c^2   (at mu = m)
    =>  T_c^{UR} = sqrt( 3 n / (g m) )   ~  m^{-1/2}.                          (4)

BOTH asymptotes DIVERGE as m -> 0 at fixed n, but the non-relativistic one
diverges FASTER (m^{-1} vs m^{-1/2}).  Hence the full relativistic T_c lies
*below* the naive NR extrapolation for light bosons, and there is NO
mass-independent plateau: T_c(m) -> infinity as m -> 0, monotonically.
The number "~398 MeV" quoted in an earlier draft as a mass-independent UR plateau
is checked explicitly below; it is in fact the *mass-dependent* value of Eq. (4)
at one particular mass (near the pion mass), not an asymptote.

Run
---
    python bec_critical_temperature.py

Prints the T_c(m) table for a few net charge densities, the NR/UR asymptotes and
crossover, the "398 MeV" check, and saves arrays to code/data/bec_tc.npz.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.optimize import brentq

# --- make the shared helpers importable regardless of CWD -------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402  (constants: FM, HBARC_GEV_FM, M_PION, ...)

ZETA_3_2 = special.zeta(1.5)  # zeta(3/2) = 2.6123753486...


# ===========================================================================
# 1. The dimensionless net-charge integral  J(a),  a = m / T
# ===========================================================================
def _bose(x):
    """1 / (exp(x) - 1) with a safe large-x cutoff (returns 0 for x -> inf)."""
    if x > 700.0:
        return 0.0
    # expm1 is accurate for small x; near x -> 0 this ~ 1/x (integrable here).
    return 1.0 / np.expm1(x)


def J_integral(a):
    r"""Dimensionless net-charge integral at mu = m:

        J(a) = int_0^inf du u^2 [ 1/(exp(sqrt(u^2+a^2) - a) - 1)
                                  - 1/(exp(sqrt(u^2+a^2) + a) - 1) ],   u = p/T.

    so that  n = (g T^3 / 2 pi^2) J(m/T).  The particle branch is finite at
    u -> 0: sqrt(u^2+a^2) - a ~ u^2/(2a) so u^2/(exp(.)-1) -> 2a.
    Limits:  J -> (2 pi^2 / 3) a  (a -> 0, UR);
             J -> 2 pi^2 zeta(3/2) (a / 2 pi)^{3/2}  (a -> inf, NR).
    """
    a = float(a)

    def integrand(u):
        if u <= 0.0:
            return 0.0
        e = np.sqrt(u * u + a * a)
        # E - m computed as u^2/(E + m) to avoid catastrophic cancellation
        # (and an accidental exact 0 in the Bose denominator) at small u.
        xp = u * u / (e + a) if (e + a) > 0 else e - a
        return u * u * (_bose(xp) - _bose(e + a))

    # In the NR regime (large a) the integrand is peaked near u ~ sqrt(2 a) and
    # cut off once E - m ~ 60 T; integrate to a finite upper bound with helper
    # break-points (quad forbids break-points on an infinite interval).
    if a > 3.0:
        upk = np.sqrt(2.0 * a)
        u_max = np.sqrt((a + 60.0) ** 2 - a * a)  # E - m = 60 T
        pts = [p for p in (0.5 * upk, upk, 2.0 * upk, 4.0 * upk) if 0 < p < u_max]
        val, _ = quad(integrand, 0.0, u_max, points=pts, limit=400)
    else:
        val, _ = quad(integrand, 0.0, np.inf, limit=400)
    return val


def net_charge_density(T, m, g=1.0):
    """Net charge density n(T, mu=m) [GeV^3] for the relativistic charged Bose
    gas at the condensation threshold mu = m."""
    a = m / T
    return g * T ** 3 / (2.0 * np.pi ** 2) * J_integral(a)


# ===========================================================================
# 2. Asymptotes
# ===========================================================================
def Tc_NR(m, n, g=1.0):
    """Non-relativistic BEC critical temperature, Eq. (3):  ~ m^{-1}."""
    return (2.0 * np.pi / m) * (n / (g * ZETA_3_2)) ** (2.0 / 3.0)


def Tc_UR(m, n, g=1.0):
    """Ultra-relativistic (Haber--Weldon) critical temperature, Eq. (4): ~ m^{-1/2}."""
    return np.sqrt(3.0 * n / (g * m))


# ===========================================================================
# 3. Exact solve:  find T_c such that n(T_c, mu=m) = n_target
# ===========================================================================
def solve_Tc(m, n_target, g=1.0):
    """Solve the full Haber--Weldon constraint for T_c at fixed net charge n_target.

    n(T) is monotonically increasing in T at fixed m, mu=m, so a sign change is
    bracketed and refined with Brent's method.
    """
    def F(T):
        return net_charge_density(T, m, g) - n_target

    # Bracket using the two asymptotes (the true T_c lies near the smaller-slope
    # crossover); widen generously to be safe.
    guesses = np.array([Tc_NR(m, n_target, g), Tc_UR(m, n_target, g)])
    lo = 1e-3 * guesses.min()
    hi = 1e3 * guesses.max()
    # ensure a genuine sign change
    flo, fhi = F(lo), F(hi)
    n_expand = 0
    while flo > 0 and n_expand < 60:
        lo *= 0.5
        flo = F(lo)
        n_expand += 1
    n_expand = 0
    while fhi < 0 and n_expand < 60:
        hi *= 2.0
        fhi = F(hi)
        n_expand += 1
    return brentq(F, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=200)


# ===========================================================================
# 4. Driver: build tables, run checks, save data
# ===========================================================================
def fm3_to_GeV3(n_fm3):
    """Convert a number density in fm^-3 to GeV^3 (1 fm^-1 = 0.19733 GeV)."""
    return n_fm3 * C.HBARC_GEV_FM ** 3  # (GeV*fm)^3 / fm^3 -> GeV^3


def build_table(n_target, g=1.0, m_grid=None):
    if m_grid is None:
        # span ultralight (m << T_c) to non-relativistic (m >> T_c)
        m_grid = np.logspace(-3, 1.0, 60)  # GeV : 1 MeV ... 10 GeV
    Tc = np.array([solve_Tc(m, n_target, g) for m in m_grid])
    Tnr = Tc_NR(m_grid, n_target, g)
    Tur = Tc_UR(m_grid, n_target, g)
    return m_grid, Tc, Tnr, Tur


def main():
    g = 1.0  # scalar boson; a spin degeneracy rescales n -> n/g in Eqs. (3),(4)
    print("=" * 74)
    print("Relativistic charged Bose gas: T_c(m) at fixed NET charge density")
    print("Haber-Weldon condition, mu_c = m, full Bose integrals (g = %g)" % g)
    print("=" * 74)
    print("zeta(3/2) = %.10f" % ZETA_3_2)

    n_fm3_list = [1.0, 0.1, 5.0]
    data = {}

    for n_fm3 in n_fm3_list:
        n = fm3_to_GeV3(n_fm3)
        print("\n" + "-" * 74)
        print("Net charge density n = %g fm^-3 = %.6e GeV^3" % (n_fm3, n))
        print("-" * 74)
        header = ("   m [GeV]   m/Tc      Tc_exact    Tc_NR      Tc_UR    "
                  "  regime   Tc/Tc_NR")
        print(header)
        m_tab = np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.05, M := C.M_PION,
                          0.3, 0.5, 0.938272, 2.0, 5.0])
        rows = []
        for m in m_tab:
            Tc = solve_Tc(m, n, g)
            tnr = Tc_NR(m, n, g)
            tur = Tc_UR(m, n, g)
            aC = m / Tc
            regime = "NR" if aC > 3 else ("UR" if aC < 0.3 else "cross")
            print("  %8.4f  %7.3f  %9.4f  %9.4f  %9.4f  %7s   %6.3f"
                  % (m, aC, Tc, tnr, tur, regime, Tc / tnr))
            rows.append((m, Tc, tnr, tur, aC))
        data["n_fm3_%g" % n_fm3] = np.array(rows)

        # --- the "398 MeV" check --------------------------------------------
        # For which mass does the UR formula give T_c ~ 0.398 GeV?
        # T_UR = sqrt(3 n / m) = 0.398  =>  m = 3 n / 0.398^2
        m_star = 3.0 * n / (g * 0.398 ** 2)
        Tc_star = solve_Tc(m_star, n, g)
        print("  '398 MeV' check:  T_UR = 398 MeV corresponds to m = %.1f MeV"
              % (m_star * 1e3))
        print("                    (exact T_c there = %.1f MeV; m_pi = %.1f MeV)"
              % (Tc_star * 1e3, C.M_PION * 1e3))

    # --- crossover and m -> 0 behaviour statement ---------------------------
    n = fm3_to_GeV3(1.0)
    print("\n" + "=" * 74)
    print("m -> 0 behaviour at fixed net charge (n = 1 fm^-3):")
    print("=" * 74)
    print("     m [MeV]      Tc_exact [MeV]   Tc*sqrt(m) [MeV*GeV^0.5]  (UR: const)")
    for m in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        Tc = solve_Tc(m, n, g)
        inv = Tc * np.sqrt(m)  # should approach sqrt(3 n / g) = const in UR
        print("   %9.4f     %12.2f      %14.5f"
              % (m * 1e3, Tc * 1e3, inv * 1e3))
    print("  sqrt(3 n / g) = %.5f  (GeV^{3/2}); Tc*sqrt(m) -> this in the UR limit"
          % np.sqrt(3.0 * n / g))
    print("  => T_c DIVERGES as m^{-1/2}; it does NOT saturate to a plateau.")

    # --- save arrays for Fig. 11 -------------------------------------------
    m_grid, Tc, Tnr, Tur = build_table(n, g)
    out_dir = os.path.join(CODE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "bec_tc.npz")
    np.savez(out,
             m_grid_GeV=m_grid, Tc_exact_GeV=Tc, Tc_NR_GeV=Tnr, Tc_UR_GeV=Tur,
             n_GeV3=n, n_fm3=1.0, g=g, zeta_3_2=ZETA_3_2)
    print("\nSaved Fig.11 arrays -> %s" % out)

    # crossover mass where m = T_c(m)
    from scipy.optimize import brentq as _bq
    f_cross = lambda m: m - solve_Tc(m, n, g)
    m_cross = _bq(f_cross, 1e-3, 10.0)
    print("Crossover m = k_B T_c at n = 1 fm^-3:  m ~ %.1f MeV (T_c ~ %.1f MeV)"
          % (m_cross * 1e3, solve_Tc(m_cross, n, g) * 1e3))


if __name__ == "__main__":
    main()
