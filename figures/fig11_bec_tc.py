# -*- coding: utf-8 -*-
r"""
fig11_bec_tc.py  ->  figures/fig11_bec_tc.{pdf,png}   (Fig. 11, fig:bec)
========================================================================

WHAT IT SHOWS
-------------
The Bose--Einstein condensation critical temperature T_c of a relativistic ideal
*charged* Bose gas (Haber--Weldon) at fixed **net charge density**
n = n_particles - n_antiparticles = 1 fm^-3, as a function of the boson mass m:

    * exact relativistic curve   T_c(m)          (full Bose integrals, mu_c = m)
    * non-relativistic asymptote T_c^{NR} ~ m^{-1}   (diverges fastest as m->0)
    * ultra-relativistic asymptote T_c^{UR} ~ m^{-1/2}

Log--log axes.  The crossover between the two regimes sits at m ~ T_c
(~ 282 MeV for n = 1 fm^-3), where the exact curve bends from the m^{-1} branch
onto the m^{-1/2} branch.

KEY MESSAGES (caption-relevant)
-------------------------------
  * The exact T_c lies BELOW T_c^{NR} everywhere: relativity + antiparticles
    (pair fluctuations that carry the conserved charge) SUPPRESS T_c.
  * T_c DIVERGES as m -> 0, going like m^{-1/2} (the UR asymptote), NOT to a
    mass-independent plateau.  There is NO "398 MeV plateau": 398 MeV is merely
    the value of the mass-dependent UR formula near the pion mass, not a limit.

PROVENANCE:  COMPUTED.  Arrays are loaded from code/data/bec_tc.npz, produced by
code/calculations/bec_critical_temperature.py (full relativistic Bose integrals,
Brent root-find of the net-charge constraint at mu = m).  If the .npz is absent,
this script runs that calculation first.

CONVENTION
----------
Natural units c = hbar = k_B = 1 (Eq. conv-natural-units); the condensation
threshold is the relativistic value mu_c = m (Sec. quantum), not the
non-relativistic mu_c = 0.  Masses and temperatures are shown in MeV
(1 GeV = 1000 MeV); the charge density is quoted in fm^-3.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402

DATA = os.path.join(CODE_DIR, "data", "bec_tc.npz")
CALC = os.path.join(CODE_DIR, "calculations", "bec_critical_temperature.py")


def load_data():
    """Load bec_tc.npz, regenerating it from the calculation if necessary."""
    if not os.path.exists(DATA):
        print("bec_tc.npz not found -- running the calculation first ...")
        subprocess.run([sys.executable, CALC], cwd=CODE_DIR, check=True)
    d = np.load(DATA)
    return d


def make_figure():
    plt = C.apply_style()
    d = load_data()

    m = d["m_grid_GeV"]           # GeV
    Tc = d["Tc_exact_GeV"]        # GeV
    Tnr = d["Tc_NR_GeV"]
    Tur = d["Tc_UR_GeV"]
    n_fm3 = float(d["n_fm3"])

    # to MeV for display
    m_MeV = m * 1e3
    Tc_MeV, Tnr_MeV, Tur_MeV = Tc * 1e3, Tnr * 1e3, Tur * 1e3

    # crossover mass where m = T_c(m) (log-interpolate)
    f = np.log(Tc) - np.log(m)            # = 0 at crossover
    s = np.where(np.diff(np.sign(f)))[0]
    if len(s):
        i = s[0]
        w = f[i] / (f[i] - f[i + 1])
        m_cross_MeV = np.exp(np.log(m_MeV[i]) + w * (np.log(m_MeV[i + 1]) -
                                                     np.log(m_MeV[i])))
        Tc_cross_MeV = m_cross_MeV        # by definition m = T_c there
    else:
        m_cross_MeV = Tc_cross_MeV = np.nan

    # --- run-time verification ---------------------------------------------
    print("=" * 70)
    print("fig11_bec_tc: relativistic charged-Bose T_c(m), n = %g fm^-3" % n_fm3)
    print("=" * 70)
    print("  exact T_c <= T_c^NR everywhere: %s (max T_c/T_NR = %.4f)"
          % (bool(np.all(Tc <= Tnr * (1 + 1e-9))), (Tc / Tnr).max()))
    print("  m -> 0 edge: m=%.2f MeV  T_c=%.1f MeV  T_c*sqrt(m)=%.4f (UR const=%.4f)"
          % (m_MeV[0], Tc_MeV[0], Tc[0] * np.sqrt(m[0]),
             np.sqrt(3.0 * float(d["n_GeV3"]) / float(d["g"]))))
    print("  crossover m = T_c at m ~ %.1f MeV (T_c ~ %.1f MeV)"
          % (m_cross_MeV, Tc_cross_MeV))
    print("-" * 70)

    fig, ax = plt.subplots(figsize=(5.4, 4.1))

    ax.plot(m_MeV, Tnr_MeV, color="#EE7733", lw=1.8, ls=(0, (5, 2)),
            label=r"non-relativistic $T_c^{\mathrm{NR}}\propto m^{-1}$")
    ax.plot(m_MeV, Tur_MeV, color="#009988", lw=1.8, ls=(0, (1, 1.4)),
            label=r"ultra-relativistic $T_c^{\mathrm{UR}}\propto m^{-1/2}$")
    ax.plot(m_MeV, Tc_MeV, color="#0077BB", lw=2.3, zorder=4,
            label=r"exact relativistic $T_c$")

    # crossover marker
    if np.isfinite(m_cross_MeV):
        ax.plot([m_cross_MeV], [Tc_cross_MeV], "o", color="#CC3311",
                ms=6, zorder=6)
        ax.annotate(r"crossover $m\simeq T_c\approx%.0f$ MeV" % m_cross_MeV,
                    xy=(m_cross_MeV, Tc_cross_MeV),
                    xytext=(m_cross_MeV * 2.3, Tc_cross_MeV * 2.6),
                    fontsize=8.6, color="#CC3311", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color="#CC3311", lw=1.0,
                                    connectionstyle="arc3,rad=0.2"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"boson mass  $m$  [MeV]")
    ax.set_ylabel(r"critical temperature  $T_c$  [MeV]")
    ax.set_title(r"Relativistic BEC of a charged Bose gas"
                 "\n" r"(fixed net charge density $n=%g\ \mathrm{fm}^{-3}$)"
                 % n_fm3, fontsize=10.5)

    ax.set_xlim(m_MeV.min(), m_MeV.max())
    ax.legend(loc="upper right", fontsize=8.8, handlelength=2.1)

    # annotate the two divergence rates and the suppression
    ax.text(1.6, 6.0e4, r"$T_c\to\infty$ as $m\to0$" "\n" r"(no plateau)",
            fontsize=8.4, color="#0077BB", ha="left", va="top")
    ax.text(2.6e3, 40.0, r"exact $<T_c^{\mathrm{NR}}$" "\n"
            r"(antiparticles suppress $T_c$)",
            fontsize=8.2, color="#333333", ha="center", va="center")

    paths = C.savefig_dual(fig, "fig11_bec_tc")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    return paths


if __name__ == "__main__":
    make_figure()
