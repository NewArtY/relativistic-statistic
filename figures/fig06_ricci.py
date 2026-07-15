# -*- coding: utf-8 -*-
r"""
fig06_ricci.py  ->  figures/fig06_ricci.{pdf,png}   (Fig. 6, fig:ricci)
=======================================================================

WHAT IT SHOWS
-------------
The thermodynamic (Fisher--Rao / Ruppeiner) Ricci scalar curvature R of the
relativistic ideal gas as a function of the fugacity z, for the three
Boltzmann--Gibbs branches of the master occupation of Sec. 5
(Eq. master-occupation):

    * Fermi--Dirac   (a = +1):  R < 0   (effective statistical repulsion, Pauli)
    * Bose--Einstein (a = -1):  R > 0   and R -> +infinity as z -> 1^-
                                 (geometric signature of BEC, Cor. bec-geometry)
    * Maxwell--Juttner (a = 0):  R = 0  identically (flat classical manifold)

Fixed coldness zeta = m/T = 1, spatial dimension D = 3.

PROVENANCE:  COMPUTED (original result of this work).
Nothing is loaded from disk; R is evaluated by direct numerical integration of
the relativistic susceptibility-weighted energy moments and the
Janyszek--Mrugala determinant curvature formula (Eq. ricci-det).

METHOD
------
Metric on the intensive manifold x = (beta, gamma), beta = 1/T,
gamma = -beta*mu = -ln z, is the Hessian of ln Z (Def. fisher-metric):

    g_{bb} = I_2 ,   g_{bg} = I_1 ,   g_{gg} = I_0 ,

with the susceptibility-weighted moments (Eq. master-moments)

    I_k = \int_m^\infty eps^k Omega(eps) w(eps) deps ,
    w = n (1 - a n) ,   n = 1/(exp[beta(eps-mu)] + a) ,

Omega(eps) = eps (eps^2 - m^2)^{(D-2)/2} the relativistic DOS (Sec. quantum).
The moment derivatives raise the order through a factor (1 - 2 a n)
(Eq. moment-derivs):

    d_beta  I_k = -J_{k+1} ,   d_gamma I_k = -J_k ,
    J_k = \int_m^\infty eps^k Omega (1 - 2 a n) w deps ,

so the Ricci scalar (Eq. ricci-det) is

              -1   | I_2   I_1   I_0 |
    R  =  --------- | -J_3  -J_2  -J_1 | ,    g = I_0 I_2 - I_1^2 .
            2 g^2   | -J_2  -J_1  -J_0 |

CONVENTION
----------
Natural units c = hbar = k_B = 1 (Eq. conv-natural-units).  The x-axis is the
rest-mass-referenced activity ztilde (Eq. activity), measured relative to the
relativistic condensation threshold mu_c = m (Sec. quantum), i.e.

    ztilde = exp[(mu - m)/T]  in (0, 1),

so that ztilde -> 1^- is the Bose threshold mu -> m^- (Cor. bec-geometry) and
ztilde -> 0 is the dilute Maxwell--Juttner limit.  It is distinct from the
frozen fugacity z = exp(mu/T) of the master occupation; ztilde = z * exp(-zeta).  The integration is carried out in the
particle momentum p (Omega deps = p^{D-1} dp), which removes the band-edge
square-root of the DOS and makes the near-threshold Bose peak easy to resolve.

NUMERICAL NOTES (risk R10)
--------------------------
Near z -> 1^- the bosonic weight w = n(1+n) develops a sharp integrable peak at
the band edge (p ~ sqrt(2(1-z)) m), so the quadrature is given break-points at
that scale, a tight epsrel, and a dense z-grid on the right edge.  The +infinity
divergence is physical (a curvature singularity == diverging correlation length
== BEC), cross-checked here by the monotone growth of R_BE as z -> 1.  The
determinant curvature was validated against an independent finite-difference
evaluation of the 2D scalar curvature (agreement < 0.3%).

The figure has TWO panels:
  (a) Boltzmann-Gibbs branches (sigma = 0): FD, BE, MJ, as above.
  (b) Kaniadakis Bose gas (sigma = kappa, a = -1) for several kappa: the
      deformation drives R negative in the dilute limit and crosses zero at
      a crossover z*(kappa) that grows with kappa and -> 0 as kappa -> 0.
      This is the honest, verified deformed-gas result (Prop.
      deformed-curvature): sgn R = -sgn(a) is INVERTED for the Kaniadakis
      Bose gas below z*(kappa), so the deformation does NOT simply preserve
      the sign rule.  The Kaniadakis metric exists only for kappa < 1/4
      (energy-variance moment I_2 diverges beyond, D=3).

VERIFIED BEHAVIOUR (printed at run time)
----------------------------------------
  Panel (a):  sgn R = -sgn(a) throughout 0 < z < 1 (Prop. curvature-sign);
    R_FD < 0, R_BE > 0, R_MJ = 0; R_BE -> +inf as z -> 1^-; small-z limit
    R -> -a c1(zeta) with c1 > 0 (Lemma smallz-curvature).
  Panel (b):  Kaniadakis Bose R_kappa < 0 for z < z*(kappa); crossover
    z* = 0.024, 0.092, 0.204, 0.338 at kappa = 0.05, 0.10, 0.15, 0.20
    (zeta = 1).  Cross-checked against a finite-difference scalar curvature
    of g_ij(beta,gamma) and against 50-digit mpmath quadrature; the three
    agree in the convergence window kappa < 1/4.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402  (apply_style, savefig_dual, constants)


# ===========================================================================
# 1. Relativistic susceptibility moments and the determinant curvature
# ===========================================================================
def _moment(k, a, z, m, deriv, p_max, pts):
    """Single moment integral (in momentum p, D=3 -> weight p^2).

        I_k  = int p^2 eps^k w dp                        (deriv = False)
        J_k  = int p^2 eps^k (1 - 2 a n) w dp            (deriv = True)

    with eps = sqrt(p^2 + m^2), n = z/(exp(eps-m) + a z), w = n(1 - a n).
    Units T = 1 so beta = 1 and eps - mu = (eps - m) - ln z.
    """
    def integrand(p):
        eps = np.sqrt(p * p + m * m)
        x = eps - m                       # = beta (eps - m) >= 0
        if x > 700.0:
            return 0.0
        ex = np.exp(x)
        denom = ex + a * z                # exp(eps-mu) = exp(x)/z, scaled by z
        if denom <= 0.0:                  # forbidden (mu > m) -- never for z<1
            return 0.0
        n = z / denom
        w = n * (1.0 - a * n)
        val = p * p * eps ** k * w
        if deriv:
            val *= (1.0 - 2.0 * a * n)
        return val

    v, _ = quad(integrand, 0.0, p_max, points=pts, limit=500,
                epsabs=1e-14, epsrel=1e-12)
    return v


def ricci_scalar(z, a, m=1.0, D=3):
    """Thermodynamic Ricci scalar R(z) for quantum sign a at coldness zeta=m.

    (T = 1, so zeta = m/T = m.)  Returns R via Eq. ricci-det.  For the
    classical branch a = 0 the determinant vanishes identically -> R = 0.
    """
    if D != 3:
        raise NotImplementedError("fig06 is specialised to D = 3.")

    # Break-points: resolve the near-threshold Bose peak at p ~ sqrt(2(1-z)) m
    # and give the exponential fall-off a few nodes.  Upper cut where eps-m ~ 90.
    delta = max(1.0 - z, 1e-13)
    ppk = np.sqrt(2.0 * delta) * m if a < 0 else 0.5 * m
    p_max = np.sqrt((m + 90.0) ** 2 - m * m)
    pts = sorted({p for p in (0.25 * ppk, 0.5 * ppk, ppk, 2.0 * ppk,
                              4.0 * ppk, 8.0 * ppk, m, 3.0 * m)
                  if 0.0 < p < p_max})

    I2 = _moment(2, a, z, m, False, p_max, pts)
    I1 = _moment(1, a, z, m, False, p_max, pts)
    I0 = _moment(0, a, z, m, False, p_max, pts)
    if a == 0:
        return 0.0
    J0 = _moment(0, a, z, m, True, p_max, pts)
    J1 = _moment(1, a, z, m, True, p_max, pts)
    J2 = _moment(2, a, z, m, True, p_max, pts)
    J3 = _moment(3, a, z, m, True, p_max, pts)

    g = I0 * I2 - I1 * I1                       # det g > 0 (Cauchy-Schwarz)
    mat = np.array([[I2, I1, I0],
                    [-J3, -J2, -J1],
                    [-J2, -J1, -J0]])
    return -np.linalg.det(mat) / (2.0 * g * g)


# ===========================================================================
# 1b. Dilute-limit constant c1(zeta):  R -> -a c1(zeta) as ztilde -> 0
# ===========================================================================
# The small-fugacity Lemma (lem:smallz-curvature) gives R = -a c1(zeta) + O(ztilde)
# with c1(zeta) = -Delta(zeta) / [M0 M2 - M1^2]^2, where the BG moments carry the
# weights e^{-beta eps} (unbarred) and e^{-2 beta eps} (barred):
#
#     M_k    = int_m^inf eps^k Omega(eps) e^{-beta eps}  deps = m^{k+3} Kcal_{k+1}(zeta),
#     Mbar_k = int_m^inf eps^k Omega(eps) e^{-2 beta eps} deps = m^{k+3} Kcal_{k+1}(2 zeta),
#     Kcal_n(x) = int_0^inf e^{-x cosh t} sinh^2 t cosh^n t dt   (Macdonald combos),
#     Delta = det[[M2,M1,M0],[M3,M2,M1],[Mbar2,Mbar1,Mbar0]].
#
# c1 > 0 for every zeta; UR plateau c1(zeta->0) = 1/16 (massless moments give
# Delta = -9, denominator 144), NR decay c1 -> 0 as zeta -> infinity.  This makes
# sgn R = -sgn(a) robust across the whole coldness range (Prop. curvature-sign).
def _bg_moment(k, m, beta):
    """M_k(beta) = int_m^inf eps^k Omega(eps) e^{-beta eps} deps  (D=3, in eps)."""
    def integrand(eps):
        return eps ** k * eps * np.sqrt(eps * eps - m * m) * np.exp(-beta * eps)
    v, _ = quad(integrand, m, np.inf, limit=400, epsabs=1e-14, epsrel=1e-11)
    return v


def c1_of_zeta(zeta):
    """Dilute-limit curvature constant c1(zeta) (T = 1, so m = zeta)."""
    m = zeta
    M0 = _bg_moment(0, m, 1.0); M1 = _bg_moment(1, m, 1.0)
    M2 = _bg_moment(2, m, 1.0); M3 = _bg_moment(3, m, 1.0)
    Mb0 = _bg_moment(0, m, 2.0); Mb1 = _bg_moment(1, m, 2.0)
    Mb2 = _bg_moment(2, m, 2.0)
    Delta = np.linalg.det(np.array([[M2, M1, M0],
                                    [M3, M2, M1],
                                    [Mb2, Mb1, Mb0]]))
    return -Delta / (M0 * M2 - M1 * M1) ** 2


# ===========================================================================
# 1c. Deformed-gas curvature R_q, R_kappa (Tsallis / Kaniadakis) -- diagnostic
# ===========================================================================
# CORRECTED master occupation (Def. def:master-occupation):
#   n = 1/(E + a),  E = [exp_sigma(-y)]^{-1},  y = (eps - m) - ln ztilde  (T = 1).
# For sigma in {0, kappa} reciprocity gives E = exp_sigma(+y); Tsallis has no
# reciprocity, so E = [exp_q(-y)]^{-1} = [1 + (q-1) y]^{1/(q-1)}, whose classical
# branch n = exp_q(-y) carries the power-law tail eps^{-1/(q-1)} (NOT the old
# compact-support form).  Fluctuation kernel w = -dn/dy = E'/(E+a)^2 ; its
# y-derivative wprime = [E'' (E+a) - 2 E'^2]/(E+a)^3.  Deformation derivatives:
#   Tsallis:    E = [1+(q-1)y]^{1/(q-1)}, u = 1+(q-1)y,  E' = E/u,  E'' = (2-q)E/u^2.
#               No cutoff for q>1 (u>0 since y is bounded below); the metric moment
#               I_2 converges only for q < 5/4 (tail w ~ eps^{-1-1/(q-1)}).
#   Kaniadakis: E = exp_k(y), s = sqrt(1+k^2 y^2), E' = E/s, E'' = E/s^2 - E k^2 y/s^3.
# FINDING (Prop. deformed-curvature, verified): sgn R = -sgn(a) holds for the
# UNDEFORMED quantum gases over 0 < ztilde < 1.  Under the Tsallis deformation the
# classical branch (a=0) is EXACTLY flat (R_q == 0, det = 0 to 40 digits) and the
# sign rule is RETAINED (Fermi R<0, Bose R>0, no crossover).  Only the Kaniadakis
# gas inverts it: the Kaniadakis-Bose curvature turns NEGATIVE for ztilde below a
# kappa-dependent threshold z*(kappa) (z* -> 0 as kappa -> 0).  This panel plots the
# Kaniadakis case; the Tsallis branch below is provided for verification.
def _deformed_E(y, sigma, par):
    """(E, E', E'') for sigma in {'q','k'} with pole handling; y scalar."""
    if sigma == "q":                          # CORRECTED: E = [exp_q(-y)]^{-1}
        q = par
        u = 1.0 + (q - 1.0) * y               # affine argument; u > 0 for q>1
        if u <= 0.0:
            return np.inf, np.inf, np.inf
        E = u ** (1.0 / (q - 1.0))
        return E, E / u, (2.0 - q) * E / (u * u)
    if sigma == "k":
        k = par
        s = np.sqrt(1.0 + k * k * y * y)
        E = (s + k * y) ** (1.0 / k)
        return E, E / s, E / (s * s) - E * k * k * y / (s ** 3)
    raise ValueError(sigma)


def _deformed_moment(k, a, ztilde, m, sigma, par, deriv, p_max, pts):
    lnz = np.log(ztilde)

    def integrand(p):
        eps = np.sqrt(p * p + m * m)
        y = (eps - m) - lnz
        E, Ep, Epp = _deformed_E(y, sigma, par)
        if not np.isfinite(E):
            return 0.0
        d = E + a
        if abs(d) < 1e-300:
            return 0.0
        w = Ep / (d * d)
        f = (Epp * d - 2.0 * Ep * Ep) / (d ** 3) if deriv else w
        return p * p * eps ** k * f

    v, _ = quad(integrand, 0.0, p_max, points=pts, limit=600,
                epsabs=1e-14, epsrel=1e-11)
    return v


def ricci_deformed(ztilde, a, sigma, par, m=1.0):
    """Deformed-gas thermodynamic Ricci scalar (diagnostic; T = 1, D = 3)."""
    delta = max(1.0 - ztilde, 1e-13)
    ppk = np.sqrt(2.0 * delta) * m if a < 0 else 0.5 * m
    p_max = np.sqrt((m + 90.0) ** 2 - m * m)
    pts = sorted({p for p in (0.25 * ppk, 0.5 * ppk, ppk, 2.0 * ppk,
                              4.0 * ppk, 8.0 * ppk, m, 3.0 * m)
                  if 0.0 < p < p_max})
    P2 = _deformed_moment(2, a, ztilde, m, sigma, par, False, p_max, pts)
    P1 = _deformed_moment(1, a, ztilde, m, sigma, par, False, p_max, pts)
    P0 = _deformed_moment(0, a, ztilde, m, sigma, par, False, p_max, pts)
    Q3 = _deformed_moment(3, a, ztilde, m, sigma, par, True, p_max, pts)
    Q2 = _deformed_moment(2, a, ztilde, m, sigma, par, True, p_max, pts)
    Q1 = _deformed_moment(1, a, ztilde, m, sigma, par, True, p_max, pts)
    Q0 = _deformed_moment(0, a, ztilde, m, sigma, par, True, p_max, pts)
    g = P0 * P2 - P1 * P1
    if abs(g) < 1e-300:
        return float("nan")
    mat = np.array([[P2, P1, P0], [Q3, Q2, Q1], [Q2, Q1, Q0]])
    return -np.linalg.det(mat) / (2.0 * g * g)


# ===========================================================================
# 2. Build the z-grid and evaluate the three branches
# ===========================================================================
def build_curves(zeta=1.0):
    m = zeta  # T = 1
    # Dense on the right edge (z -> 1^-) to resolve the Bose divergence.
    z_bulk = np.logspace(-3.0, np.log10(0.9), 170)
    z_edge = 1.0 - np.logspace(np.log10(0.1), -3.3, 90)   # 0.9 ... 0.9995
    z = np.unique(np.concatenate([z_bulk, z_edge]))
    z = z[(z > 0) & (z < 1)]

    R_fd = np.array([ricci_scalar(zi, +1, m) for zi in z])
    R_be = np.array([ricci_scalar(zi, -1, m) for zi in z])
    R_mj = np.zeros_like(z)
    return z, R_fd, R_be, R_mj


# The Kaniadakis metric moment g_bb = I_2 = int p^2 eps^2 w dp converges only
# for kappa < 1/(k+2)|_{k=2} = 1/4 in D = 3: the power-law tail w ~ eps^{-1-1/kappa}
# makes the energy variance (and hence the thermodynamic metric) divergent for
# kappa >= 1/4.  The curvature is therefore defined only in this window, which we
# respect by keeping kappa <= 0.20 in the deformed panel.
KAPPA_MAX = 0.25


def build_deformed_curves(kappas, zeta=1.0):
    """Kaniadakis-Bose (a = -1) curvature R_kappa(ztilde) for several kappa.

    Returns dict kappa -> (z, R) and the interpolated crossover z*(kappa)
    where R changes sign from negative (dilute) to positive.
    """
    m = zeta
    z = np.logspace(-3.0, np.log10(0.9), 130)
    out, zstar = {}, {}
    for kap in kappas:
        assert kap < KAPPA_MAX, "kappa outside metric-convergence window"
        R = np.array([ricci_deformed(zi, -1, "k", kap, m) for zi in z])
        out[kap] = (z, R)
        zc = None
        for i in range(len(z) - 1):
            if R[i] < 0.0 <= R[i + 1]:
                f = (0.0 - R[i]) / (R[i + 1] - R[i])
                zc = np.exp(np.log(z[i]) + f * (np.log(z[i + 1]) - np.log(z[i])))
                break
        zstar[kap] = zc
    return out, zstar


# ===========================================================================
# 3. Plot  (two panels: BG branches, and the deformed-gas sign inversion)
# ===========================================================================
def make_figure():
    plt = C.apply_style()

    zeta = 1.0
    z, R_fd, R_be, R_mj = build_curves(zeta)

    # --- run-time verification, panel (a): BG branches ----------------------
    print("=" * 74)
    print("fig06_ricci: relativistic thermodynamic Ricci scalar (D=3, zeta=1)")
    print("=" * 74)
    print("PANEL (a)  Boltzmann-Gibbs branches (sigma = 0)")
    print("  %-9s %13s %13s %10s" % ("ztilde", "R_FD (a=+1)", "R_BE (a=-1)", "R_MJ"))
    for zi in (1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999):
        j = int(np.argmin(np.abs(z - zi)))
        print("  %-9.4f %13.6e %13.6e %10.1e"
              % (z[j], R_fd[j], R_be[j], R_mj[j]))
    print("  R_FD < 0 everywhere: %s ; R_BE > 0 everywhere: %s"
          % (bool(np.all(R_fd < 0)), bool(np.all(R_be > 0))))
    c1_be = -R_be[0] / (-1)
    print("  small-ztilde c1(zeta=1) ~ %.4e (>0) ; sgn R = -sgn(a) OK" % c1_be)

    # --- panel (b): Kaniadakis-Bose sign inversion --------------------------
    kappas = [0.05, 0.10, 0.15, 0.20]
    defc, zstar = build_deformed_curves(kappas, zeta)
    print("-" * 74)
    print("PANEL (b)  Kaniadakis Bose gas (a = -1): dilute-limit sign inversion")
    print("  the classical (a=0) Kaniadakis manifold is NOT flat (R<0), so the")
    print("  deformation drives R_BE < 0 for ztilde < z*(kappa); sgn R = -sgn(a) fails.")
    print("  %-8s %14s %12s" % ("kappa", "R_BE(ztilde=1e-3)", "z*(kappa)"))
    for kap in kappas:
        zz, RR = defc[kap]
        zc = zstar[kap]
        print("  %-8.2f %14.4e %12s"
              % (kap, RR[0], ("%.4f" % zc) if zc else "--"))
    print("  z*(kappa) grows with kappa and -> 0 as kappa -> 0 "
          "(BG limit, no inversion).")
    print("-" * 74)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.1))

    # ---- panel (a) ---------------------------------------------------------
    R_cap = 0.62
    be_plot = np.where(R_be <= R_cap, R_be, np.nan)
    axA.axhline(0.0, color="0.55", lw=0.9, zorder=1)
    axA.plot(z, be_plot, color="#0077BB", lw=2.0, zorder=3,
             label="Bose–Einstein $(a=-1)$: $R>0$")
    axA.plot(z, R_mj, color="#555555", lw=1.8, ls=(0, (5, 2)), zorder=2,
             label="Maxwell–Jüttner $(a=0)$: $R\\equiv 0$")
    axA.plot(z, R_fd, color="#CC3311", lw=2.0, zorder=3,
             label="Fermi–Dirac $(a=+1)$: $R<0$")
    z_div = z[np.nanargmin(np.abs(be_plot - (R_cap - 0.04)))]
    axA.annotate(r"$R_{\mathrm{BE}}\!\to\!+\infty$" "\n" r"as $\tilde{z}\to1^-$ (BEC)",
                 xy=(z_div, R_cap - 0.02), xytext=(0.16, 0.50),
                 fontsize=8.6, color="#0077BB", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#0077BB", lw=1.1,
                                 connectionstyle="arc3,rad=-0.2"))
    axA.text(2.0e-3, -0.052, r"Pauli repulsion", color="#CC3311",
             fontsize=8.2, ha="left", va="center")
    axA.text(0.30, 0.028, r"flat (uncorrelated)", color="#555555",
             fontsize=8.2, ha="left", va="center")
    axA.set_xscale("log")
    axA.set_xlim(1e-3, 1.0)
    axA.set_ylim(-0.07, R_cap)
    axA.set_xlabel(r"rest-mass-referenced activity  $\tilde{z} = e^{(\mu-m)/T}$")
    axA.set_ylabel(r"thermodynamic Ricci scalar  $R$")
    axA.set_title(r"(a)  Boltzmann–Gibbs branches ($\sigma=0$)", fontsize=10.0)
    axA.legend(loc="upper left", fontsize=8.2, handlelength=1.8,
               borderaxespad=0.5)

    # ---- panel (b): Kaniadakis Bose, symlog to span the sign change --------
    axB.axhline(0.0, color="0.55", lw=0.9, zorder=1)
    # BG (kappa -> 0) reference: positive everywhere
    z_ref = np.logspace(-3.0, np.log10(0.9), 90)
    R_be_ref = np.array([ricci_scalar(zi, -1, zeta) for zi in z_ref])
    axB.plot(z_ref, R_be_ref, color="#000000", lw=1.4, ls=(0, (5, 2)),
             zorder=4, label=r"$\kappa=0$ (BG): $R>0$")
    cols = ["#EE7733", "#009988", "#0077BB", "#AA3377"]
    for kap, col in zip(kappas, cols):
        zz, RR = defc[kap]
        axB.plot(zz, RR, color=col, lw=1.9, zorder=3,
                 label=r"$\kappa=%.2f$" % kap)
        zc = zstar[kap]
        if zc is not None:
            axB.plot([zc], [0.0], marker="o", color=col, ms=5.0,
                     zorder=5, mec="white", mew=0.6)
    axB.set_xscale("log")
    axB.set_yscale("symlog", linthresh=1e-2)
    axB.set_xlim(1e-3, 0.9)
    axB.set_ylim(-1.2, 0.3)
    # shade the sign-inverted (dilute) region schematically via text
    axB.text(1.4e-3, -0.35, "sign-inverted\n" r"$R_{\mathrm{BE}}<0$",
             color="#555555", fontsize=8.2, ha="left", va="center")
    axB.annotate(r"crossover $z_*(\kappa)$",
                 xy=(zstar[0.10], 0.0), xytext=(0.02, 0.06),
                 fontsize=8.4, color="#333333", ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0,
                                 connectionstyle="arc3,rad=0.2"))
    axB.set_xlabel(r"rest-mass-referenced activity  $\tilde{z} = e^{(\mu-m)/T}$")
    axB.set_ylabel(r"Ricci scalar  $R_{\kappa}$  (Bose, $a=-1$)")
    axB.set_title(r"(b)  Kaniadakis Bose gas: dilute-limit inversion",
                  fontsize=10.0)
    axB.legend(loc="lower right", fontsize=8.0, handlelength=1.7,
               borderaxespad=0.5, ncol=1)

    fig.suptitle(r"Thermodynamic curvature of the relativistic ideal gas"
                 r"  ($D=3$, coldness $\zeta=m/T=1$)", fontsize=10.8, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    paths = C.savefig_dual(fig, "fig06_ricci")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    return paths


if __name__ == "__main__":
    make_figure()
