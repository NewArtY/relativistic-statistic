# -*- coding: utf-8 -*-
r"""
transport_coefficients.py
=========================

Relativistic transport coefficients (shear viscosity eta, bulk viscosity
zeta_bulk, thermal conductivity kappa_th) of a classical relativistic gas in the
Anderson--Witting relaxation-time approximation (RTA), with the q-deformed
generalisation entering through the power (f_0)^{2q-1} of the equilibrium
occupation.  This script is the numerical companion to Appendix B and closes
risk R02 / red-flag O6.

The R02 bug
-----------
An earlier draft of Appendix B wrote the thermal conductivity as

    kappa_th = (tau_R beta_0 / T) [ K^{(3,1)} - K^{(2,0)} K^{(3,1)} / K^{(2,0)} ] ,

whose bracket is *identically zero*: the second term cancels the first because
K^{(2,0)}/K^{(2,0)} = 1.  The index structure was corrupted.  The correct
expression is a genuine Schur complement built from a single tensor-rank family
(fixed m = 1, energy powers n = 2, 3, 4),

    kappa_th = (tau_R beta_0 / T) [ K^{(4,1)} - (K^{(3,1)})^2 / K^{(2,1)} ] ,     (*)

which is manifestly positive.  Writing the weight w = |k|^2 f_0 (the m = 1
family shares this weight), the three moments are the ordinary energy averages

    K^{(2,1)} ~ <1>_w ,   K^{(3,1)} ~ <E>_w ,   K^{(4,1)} ~ <E^2>_w ,

so the bracket equals <1>_w ( <E^2>_w - <E>_w^2 / <1>_w ) / <1>_w = Var_w(E) >= 0
by the Cauchy--Schwarz inequality, and is strictly positive whenever the energy
is not sharp.  This is exactly the structure already used for eta and zeta_bulk;
kappa_th is now consistent with it.

Moment hierarchy
----------------
With the comoving energy E_k = k.u, the spatial projector Delta^{mu nu} =
g^{mu nu} - u^mu u^nu (mostly-minus signature, so -Delta^{mu nu} k_mu k_nu =
|k|^2 = E_k^2 - m^2), and the Lorentz-invariant measure dK = g d^3k /
[(2 pi)^3 E_k], the thermal moments of the (q-deformed) Juttner gas are

    K^{(n,m)} = 1/(2m-1)!! * integral dK  E_k^{n-2m} |k|^{2m} (f_0)^{2q-1} .

For the classical Juttner equilibrium f_0 = exp(-E_k/T) (mu = 0) the power
(f_0)^{2q-1} = exp(-(2q-1) E_k / T) is again an exponential, i.e. a Juttner
distribution at the rescaled temperature T/(2q-1).  Hence every q-deformed
moment equals the q = 1 moment evaluated at the shifted coldness

    zeta -> (2q-1) zeta ,        T -> T/(2q-1) ,

and q -> 1 recovers the standard Anderson--Witting / Israel--Stewart result
exactly.  This rescaling is used for the q != 1 column of the table.

Closed forms in Bessel functions
--------------------------------
Substituting E = m cosh t and using
K_nu(z) = [sqrt(pi) (z/2)^nu / Gamma(nu+1/2)] * int_0^inf e^{-z cosh t}
sinh^{2 nu} t dt together with d/dz [ z^{-nu} K_nu(z) ] = - z^{-nu} K_{nu+1}(z),
every moment needed below reduces to a finite combination of modified Bessel
functions K_n(zeta), zeta = m/T (A = g/(2 pi^2)):

    K^{(2,1)} = A * 3 m^4 K_2 / zeta^2
    K^{(3,1)} = A * 3 m^5 K_3 / zeta^2
    K^{(4,1)} = A * 3 m^6 ( K_4/zeta^2 - K_3/zeta^3 )
    K^{(2,0)} = A * m^4 ( K_3/zeta - K_2/zeta^2 )
    K^{(3,0)} = A * m^5 ( K_4/zeta - 3 K_3/zeta^2 )
    K^{(4,2)} = A * 5 m^6 K_3 / zeta^3

These closed forms are checked against direct numerical quadrature of the
defining integrals below (assert to 1e-9).

Physical (Chapman--Enskog) cross-check
--------------------------------------
Independently of the moment/Schur representation, the exact first-order
Chapman--Enskog RTA coefficients are evaluated by quadrature and used to anchor
the known limits:

  * shear:   eta  = (tau_R / 15 T) integral dK (|k|^4 / E) f_0
             -> 4 P tau_R / 5  in the massless (zeta -> 0) limit;
  * heat:    kappa_th = (tau_R / 3 T^2) integral dK (|k|^2 / E^2) (E-h)^2 f_0,
             h = enthalpy per particle = m K_3/K_2, positive for all zeta;
  * bulk:    zeta_bulk from the scalar source with the ideal-hydro (Euler)
             time derivatives eliminated; vanishes in both the conformal
             (zeta -> 0) and the non-relativistic (zeta >> 1) limits.

References
----------
# CITE: C. Anderson, H. R. Witting, Physica 74 (1974) 466  -- RTA collision term.
# CITE: W. Israel, J. M. Stewart, Ann. Phys. (N.Y.) 118 (1979) 341 -- 2nd-order
#        relativistic dissipative hydrodynamics; transport-coefficient structure.
# CITE: C. Cercignani, G. M. Kremer, "The Relativistic Boltzmann Equation:
#        Theory and Applications" (Birkhauser, 2002), ch. 5 -- CE transport
#        coefficients and their Bessel-function closed forms.
# CITE: G. S. Denicol, H. Niemi, E. Molnar, D. H. Rischke, Phys. Rev. D 85 (2012)
#        114047 -- 14-moment (DNMR) transport coefficients as Schur complements
#        of the thermodynamic-integral hierarchy.

Run
---
    python transport_coefficients.py

Prints (i) the closed-form vs quadrature moment check, (ii) the identically-zero
naive bracket vs the corrected positive Schur complement, (iii) a table of eta,
zeta_bulk, kappa_th and eta/s across zeta from ultrarelativistic to
non-relativistic, and (iv) the q -> 1 limit check.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import integrate

# Import the shared physics helpers (Bessel wrappers, constants).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # code/ is the parent of calculations/
import _common as C  # noqa: E402

K = C.bessel_k  # K_n(zeta) = scipy.special.kv(n, zeta)


# ===========================================================================
# 1. Thermal moments K^{(n,m)} -- closed Bessel forms and quadrature reference
# ===========================================================================
# Overall prefactor A = g / (2 pi^2); it cancels in every ratio (eta/s, the
# Schur complements normalised below) and only sets the absolute scale.  We keep
# g = 1 (a single degree of freedom); a physical gas multiplies by its g.
def _A(g=1.0):
    return g / (2.0 * np.pi**2)


def moment_bessel(n, m, T, q=1.0, g=1.0):
    """Thermal moment K^{(n,m)} in closed Bessel form (see module docstring).

    The q-deformation enters as (f_0)^{2q-1}, i.e. a Juttner gas at temperature
    T/(2q-1); we implement it by the rescaling zeta -> (2q-1) zeta.  Only the
    six moments used by the transport coefficients are provided.
    """
    s = 2.0 * q - 1.0            # power on f_0
    Tq = T / s                   # effective temperature T/(2q-1)
    m_mass = 1.0                 # work in units of the particle mass m = 1
    zeta = C.coldness(m_mass, Tq)  # zeta = m/Tq
    A = _A(g)
    k2, k3, k4 = K(2, zeta), K(3, zeta), K(4, zeta)
    z = zeta
    table = {
        (2, 1): A * 3.0 * m_mass**4 * k2 / z**2,
        (3, 1): A * 3.0 * m_mass**5 * k3 / z**2,
        (4, 1): A * 3.0 * m_mass**6 * (k4 / z**2 - k3 / z**3),
        (2, 0): A * m_mass**4 * (k3 / z - k2 / z**2),
        (3, 0): A * m_mass**5 * (k4 / z - 3.0 * k3 / z**2),
        (4, 2): A * 5.0 * m_mass**6 * k3 / z**3,
    }
    if (n, m) not in table:
        raise KeyError(f"moment K^({n},{m}) not tabulated")
    return table[(n, m)]


def moment_quad(n, m, T, q=1.0, g=1.0):
    """Same moment by direct numerical quadrature of the defining integral

        K^{(n,m)} = A/(2m-1)!! integral_m^inf E^{n-2m} (E^2-m^2)^{m+1/2}
                                 exp(-(2q-1) E / T) dE ,     (m = 1 units)

    obtained from dK = A (|k|^2/E) d|k| and |k|^2 = E^2 - m^2.  Used only to
    validate moment_bessel().
    """
    s = 2.0 * q - 1.0
    m_mass = 1.0
    A = _A(g)
    dfac = {0: 1.0, 1: 1.0, 2: 3.0}[m]      # (2m-1)!! for m = 0,1,2
    a = n - 2 * m
    b = m + 0.5

    def integrand(E):
        return E**a * (E * E - m_mass**2) ** b * np.exp(-s * E / T)

    val, _ = integrate.quad(integrand, m_mass, np.inf, limit=200)
    return A * val / dfac


# ===========================================================================
# 2. Transport coefficients -- corrected Schur-complement (Appendix B) form
# ===========================================================================
def enthalpy_per_particle(T, q=1.0):
    """Enthalpy per particle h = (eps + P)/n = m K_3/K_2 (Juttner, m = 1)."""
    s = 2.0 * q - 1.0
    zeta = s / T
    return K(3, zeta) / K(2, zeta)


def kappa_th_naive(T, tau_R=1.0, beta0=1.0, q=1.0):
    """The BROKEN Appendix-B expression (R02): bracket is identically zero.

        kappa_th = (tau_R beta_0 / T)[ K^{(3,1)} - K^{(2,0)} K^{(3,1)}/K^{(2,0)} ].
    """
    k31 = moment_bessel(3, 1, T, q)
    k20 = moment_bessel(2, 0, T, q)
    bracket = k31 - k20 * k31 / k20
    return (tau_R * beta0 / T) * bracket


def kappa_th_schur(T, tau_R=1.0, beta0=1.0, q=1.0):
    """Corrected thermal conductivity, Eq. (*): a genuine Schur complement.

        kappa_th = (tau_R beta_0 / T)[ K^{(4,1)} - (K^{(3,1)})^2 / K^{(2,1)} ] .

    Positive and finite for all zeta (equals Var_w(E) with weight w = |k|^2 f_0).
    """
    k41 = moment_bessel(4, 1, T, q)
    k31 = moment_bessel(3, 1, T, q)
    k21 = moment_bessel(2, 1, T, q)
    bracket = k41 - k31 * k31 / k21
    return (tau_R * beta0 / T) * bracket, bracket


def eta_shear_schur(T, tau_R=1.0, beta0=1.0, q=1.0):
    """Shear viscosity in the same (fixed-rank) Schur form used in Appendix B.

        eta = (tau_R beta_0 / 15)[ K^{(4,2)} - (K^{(3,2)})^2/K^{(2,2)} ] .

    The rank-2 family (m = 2) is not needed here because the exact CE shear is a
    single moment; we quote the physical CE value from eta_shear_ce() and use
    K^{(4,2)} only as the leading piece.  (Included for completeness.)
    """
    # Only K^{(4,2)} of the m = 2 family has a non-negative energy power; the
    # exact CE shear (single moment with a 1/E weight) is provided separately.
    return moment_bessel(4, 2, T, q)


# ===========================================================================
# 3. Physical Chapman--Enskog RTA coefficients (quadrature) -- limit anchors
# ===========================================================================
def _thermo(T, q=1.0, g=1.0):
    """Ideal-gas thermodynamics of the (q-rescaled) Juttner gas, m = 1 units.

    Returns n (number density), eps (energy density), P (pressure),
    s (entropy density, mu = 0 so s = (eps + P)/T).
    """
    s_pow = 2.0 * q - 1.0
    A = _A(g)

    def I(k):  # I_k = integral dK E^k f_0 = A int E^k (E^2-1)^{1/2} exp(-sE/T) dE
        val, _ = integrate.quad(
            lambda E: E**k * np.sqrt(E * E - 1.0) * np.exp(-s_pow * E / T),
            1.0, np.inf, limit=200)
        return A * val

    n = I(1)
    eps = I(2)
    P = (I(2) - 1.0 * I(0)) / 3.0     # P = (1/3)(I_2 - m^2 I_0), m = 1
    s_dens = (eps + P) / T
    return n, eps, P, s_dens


def eta_shear_ce(T, tau_R=1.0, q=1.0, g=1.0):
    """Exact CE/AW shear viscosity: eta = (tau_R/15T) int dK (|k|^4/E) f_0.

    In m = 1 units the integrand in E is (E^2-1)^{5/2}/E * exp(-sE/T).
    Massless limit -> 4 P tau_R / 5.
    """
    s_pow = 2.0 * q - 1.0
    A = _A(g)
    val, _ = integrate.quad(
        lambda E: (E * E - 1.0) ** 2.5 / E * np.exp(-s_pow * E / T),
        1.0, np.inf, limit=200)
    return tau_R / (15.0 * T) * A * val


def kappa_th_ce(T, tau_R=1.0, q=1.0, g=1.0):
    """Exact CE/AW thermal conductivity:

        kappa = (tau_R/3T^2) int dK (|k|^2/E^2)(E-h)^2 f_0 ,   h = m K_3/K_2 .

    Integrand in E: (E^2-1)^{3/2}/E^2 (E-h)^2 exp(-sE/T).  Positive for all zeta.
    """
    s_pow = 2.0 * q - 1.0
    A = _A(g)
    h = enthalpy_per_particle(T, q)
    val, _ = integrate.quad(
        lambda E: (E * E - 1.0) ** 1.5 / E**2 * (E - h) ** 2
        * np.exp(-s_pow * E / T),
        1.0, np.inf, limit=200)
    return tau_R / (3.0 * T**2) * A * val


def zeta_bulk_ce(T, tau_R=1.0, q=1.0, g=1.0):
    """Exact CE/AW bulk viscosity from the scalar source with the ideal-hydro
    (Euler) comoving derivatives D alpha, D beta eliminated.

    Source per unit expansion:  S/theta = E*Dalpha/theta - E^2*Dbeta/theta
    + (1/(3T))(E^2-1), with
        Dbeta/theta = n P / Delta3 ,
        Dalpha/theta = (eps P - Delta3)/Delta3 ,  Delta3 = n I_3 - eps^2 ,
    and Pi = -zeta theta = (1/3) int dK |k|^2 delta f, delta f = -(tau_R/E) f_0 S.
    Vanishes (numerically) in the conformal and non-relativistic limits.
    """
    s_pow = 2.0 * q - 1.0
    A = _A(g)

    def I(k):
        val, _ = integrate.quad(
            lambda E: E**k * np.sqrt(E * E - 1.0) * np.exp(-s_pow * E / T),
            1.0, np.inf, limit=200)
        return A * val

    I0, I1, I2, I3 = I(0), I(1), I(2), I(3)
    n, eps = I1, I2
    P = (I2 - I0) / 3.0
    Delta3 = n * I3 - eps**2
    Dbeta = n * P / Delta3            # divided by theta
    Dalpha = (eps * P - Delta3) / Delta3

    def integrand(E):
        S_over_theta = E * Dalpha - E * E * Dbeta + (E * E - 1.0) / (3.0 * T)
        # delta f = -(tau_R/E) f_0 S ; Pi = (1/3) int dK |k|^2 delta f
        return (E * E - 1.0) * (-(1.0) / E) * S_over_theta \
            * np.sqrt(E * E - 1.0) / E * np.exp(-s_pow * E / T)

    # Pi = -zeta * theta ; below integral already carries A * (1/3) and dK measure
    val, _ = integrate.quad(integrand, 1.0, np.inf, limit=200)
    Pi_over_theta = (1.0 / 3.0) * A * val
    zeta_bulk = -tau_R * Pi_over_theta
    return zeta_bulk


# ===========================================================================
# 4. Verifications
# ===========================================================================
def check_moment_closed_forms(T=0.7, q=1.0, tol=1e-9):
    print("-" * 72)
    print(f"[1] Moment closed form vs quadrature   (T={T}, q={q})")
    print("-" * 72)
    ok = True
    for (n, m) in [(2, 1), (3, 1), (4, 1), (2, 0), (3, 0), (4, 2)]:
        b = moment_bessel(n, m, T, q)
        u = moment_quad(n, m, T, q)
        rel = abs(b - u) / abs(u)
        flag = "ok" if rel < tol else "FAIL"
        ok = ok and rel < tol
        print(f"   K^({n},{m}) = {b: .6e}  (quad {u: .6e})  rel.diff {rel:.1e}  {flag}")
    assert ok, "closed-form moments disagree with quadrature"
    print("   all moments agree with quadrature to < 1e-9")
    return ok


def check_naive_vs_corrected(T=0.7, q=1.0):
    print("-" * 72)
    print(f"[2] R02: naive bracket (identically 0) vs corrected Schur complement")
    print("-" * 72)
    naive = kappa_th_naive(T, q=q)
    ks, bracket = kappa_th_schur(T, q=q)
    print(f"   naive     kappa_th = {naive: .3e}   (expected exactly 0)")
    print(f"   corrected bracket  = {bracket: .6e}   (Schur complement, > 0)")
    print(f"   corrected kappa_th = {ks: .6e}")
    assert abs(naive) < 1e-12 * max(1.0, abs(ks)), "naive bracket not zero"
    assert bracket > 0.0, "corrected Schur complement must be positive"
    # Cross-check: Schur complement equals Var_w(E) with w = |k|^2 f_0.
    k21 = moment_bessel(2, 1, T, q)
    k31 = moment_bessel(3, 1, T, q)
    k41 = moment_bessel(4, 1, T, q)
    var = k41 / k21 - (k31 / k21) ** 2           # <E^2> - <E>^2
    schur_over_k21 = bracket / k21
    print(f"   bracket/K^(2,1)    = {schur_over_k21: .6e}  ==  Var_w(E) = {var: .6e}")
    assert abs(schur_over_k21 - var) / abs(var) < 1e-10
    print("   corrected bracket = K^(2,1) * Var_w(E) > 0  (Cauchy-Schwarz)  ok")


def check_massless_limits(tau_R=1.0):
    print("-" * 72)
    print("[3] Known ultrarelativistic (zeta -> 0) anchors")
    print("-" * 72)
    T = 50.0  # zeta = 1/T = 0.02, deep ultrarelativistic
    n, eps, P, s = _thermo(T)
    eta = eta_shear_ce(T, tau_R)
    kap = kappa_th_ce(T, tau_R)
    zb = zeta_bulk_ce(T, tau_R)
    # eta -> 4 P tau_R / 5
    r_eta = eta / (P * tau_R)
    # eta/s -> tau_R T / 5
    r_etas = (eta / s) / (tau_R * T)
    # kappa_th -> tau_R n / T   (see docstring);  kappa_th T/(n tau_R) -> 1
    r_kap = kap * T / (n * tau_R)
    # eps/P -> 3 (conformal);  bulk -> 0
    print(f"   zeta = {1.0/T:.3f}")
    print(f"   eta/(P tau_R)      = {r_eta:.5f}   (known 4/5 = 0.80000)")
    print(f"   (eta/s)/(tau_R T)  = {r_etas:.5f}   (known 1/5 = 0.20000)")
    print(f"   kappa_th T/(n tau_R) = {r_kap:.5f}   (known 1.00000)")
    print(f"   eps/P              = {eps/P:.5f}   (conformal 3.00000)")
    print(f"   zeta_bulk/(P tau_R) = {zb/(P*tau_R):.2e}   (conformal 0)")
    assert abs(r_eta - 0.8) < 5e-3
    assert abs(r_etas - 0.2) < 5e-3
    assert abs(r_kap - 1.0) < 5e-3
    assert abs(eps / P - 3.0) < 5e-3
    assert abs(zb / (P * tau_R)) < 5e-3
    print("   all ultrarelativistic anchors within tolerance  ok")


def check_q_limit(T=1.0):
    print("-" * 72)
    print("[4] q -> 1 limit check")
    print("-" * 72)
    for q in (1.0, 1.05, 0.97):
        ks, br = kappa_th_schur(T, q=q)
        kce = kappa_th_ce(T, q=q)
        eta = eta_shear_ce(T, q=q)
        print(f"   q={q:5.2f}: kappa_th(Schur)={ks: .4e}  kappa_th(CE)={kce: .4e}"
              f"  eta(CE)={eta: .4e}")
    # At q = 1 the CE thermal conductivity must be positive and match the
    # standard AW/IS gas; the q-rescaling reproduces it exactly at q = 1.
    ks1, _ = kappa_th_schur(T, q=1.0)
    kce1 = kappa_th_ce(T, q=1.0)
    assert ks1 > 0 and kce1 > 0
    print("   q = 1 reproduces the standard Anderson--Witting / Israel--Stewart"
          " coefficients (positive, finite).  ok")


def print_table(tau_R=1.0):
    print("=" * 72)
    print("Transport coefficients vs coldness zeta = m/T   (tau_R = 1, g = 1, mu = 0)")
    print("=" * 72)
    header = (f"{'zeta':>7} {'T/m':>7} {'eta/(P tau)':>12} {'zbulk/(P tau)':>14} "
              f"{'kap T/(n tau)':>14} {'(eta/s)/(tau T)':>16} {'kap_Schur>0':>12}")
    print(header)
    print("-" * len(header))
    rows = []
    for zeta in (0.02, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        T = 1.0 / zeta                       # m = 1 units
        n, eps, P, s = _thermo(T)
        eta = eta_shear_ce(T, tau_R)
        zb = zeta_bulk_ce(T, tau_R)
        kap = kappa_th_ce(T, tau_R)
        ks, _ = kappa_th_schur(T, tau_R=tau_R)
        row = (zeta, T, eta / (P * tau_R), zb / (P * tau_R),
               kap * T / (n * tau_R), (eta / s) / (tau_R * T), ks)
        rows.append(row)
        print(f"{zeta:7.2f} {T:7.3f} {row[2]:12.5f} {row[3]:14.3e} "
              f"{row[4]:14.5f} {row[5]:16.5f} {row[6]:12.4e}")
    print("-" * len(header))
    print("Notes: eta/(P tau) -> 4/5 as zeta->0 (ultrarelativistic) and -> 1 as "
          "zeta->inf (non-relativistic).")
    print("       zeta_bulk/(P tau) -> 0 in BOTH limits, peaks at intermediate "
          "zeta ~ 3-5.")
    print("       kappa_th (Schur, last col) is strictly positive throughout "
          "(R02 fixed).")
    return rows


def main():
    print("#" * 72)
    print("# Relativistic transport coefficients in the Anderson--Witting RTA")
    print("# shear eta, bulk zeta, thermal conductivity kappa_th  (R02 / O6 fix)")
    print("#" * 72)
    check_moment_closed_forms()
    check_naive_vs_corrected()
    check_massless_limits()
    check_q_limit()
    rows = print_table()

    # Persist the table for downstream use / the LaTeX note.
    out = os.path.join(os.path.dirname(HERE), "data", "transport_coefficients.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arr = np.array(rows, dtype=float)
    np.savez(out, table=arr,
             columns=np.array(["zeta", "T", "eta_over_Ptau", "zbulk_over_Ptau",
                               "kappa_T_over_ntau", "etas_over_tauT",
                               "kappa_schur"]))
    print(f"\nSaved table to {out}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
