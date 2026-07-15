# -*- coding: utf-8 -*-
r"""
fig08_etas.py  ->  figures/fig08_etas.{pdf,png}   (Fig. 8, fig:etas)
====================================================================

  *** REPRODUCTION OF PUBLISHED POSTERIORS -- NOT AN ORIGINAL COMPUTATION ***

WHAT IT SHOWS
-------------
The temperature dependence of the specific shear viscosity eta/s(T) of QCD
matter: a median curve with a 90% credible band, together with the
Kovtun-Son-Starinets (KSS) lower bound 1/(4 pi) ~ 0.0796.

CRITICAL PROVENANCE STATEMENT (risk R04 / verification O8)
----------------------------------------------------------
This figure is a REPRODUCTION of the *published* Bayesian posterior for eta/s(T)
from the heavy-ion global-analysis literature -- principally

    J. E. Bernhard, J. S. Moreland and S. A. Bass,
    "Bayesian estimation of the specific shear and bulk viscosity of
     quark-gluon plasma", Nature Physics 15, 1113 (2019),

with the shape/uncertainty cross-checked against the JETSCAPE (Everett et al.,
PRC 103, 054904 (2021)) and TRAJECTUM (Nijs et al., PRL 126, 202301 (2021))
analyses.  It is shown here ONLY to provide physical context for the Bayesian
methodology demonstrated in this work (the BGBW fit of Fig. 7, which IS our own
computation).  We do NOT perform a viscous-hydrodynamic + emulator + MCMC global
analysis; NONE of the numbers in this figure are an original result of this
paper.  In particular the well-known minimum

    (eta/s)_min ~ 0.08-0.10  near T_c

is a *reproduced literature value*, not a measurement of ours.

The band here is a smooth PARAMETRIC stand-in with the published qualitative
features: a shallow minimum near the pseudocritical temperature T_c ~ 0.154 GeV
close to the KSS bound, and a rise on both the hadronic (T < T_c) and QGP
(T > T_c) sides, with a credible band of published width (~+-0.03 near the
minimum, widening at high T where data constrain eta/s weakly).  The exact
parametrisation is ours only as a drawing device; the physics content is cited.

PROVENANCE TAG:  REPRODUCED-LITERATURE (Bernhard-Moreland-Bass 2019 et al.).
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402

# --- Physical anchors (all cited, none computed here) ----------------------
T_C = 0.154            # GeV, pseudocritical temperature (lattice, HotQCD/Wuppertal)
KSS = 1.0 / (4.0 * np.pi)   # ~ 0.0796, Kovtun-Son-Starinets bound
# Reproduced minimum (Bernhard-Moreland-Bass 2019): (eta/s)_min = 0.085 (+0.026 / -0.025)
ETAS_MIN = 0.085
ETAS_MIN_ERR = (0.025, 0.026)   # (lower, upper) 90% credible


def etas_median(T):
    r"""Parametric median eta/s(T) reproducing the published shape.

    A two-sided (hadronic / QGP) linear-in-T rise off a minimum at T_c, of the
    kind used to parametrise the published posteriors (e.g. the JETSCAPE/Duke
    'kinked-line' prior).  This is a DRAWING parametrisation only; the numbers it
    reproduces (minimum near KSS at T_c, positive QGP slope ~2 /GeV) are cited
    literature values, not fitted here.
    """
    T = np.asarray(T, float)
    slope_lo = 1.10       # /GeV, hadronic-side slope (T < T_c)
    slope_hi = 1.95       # /GeV, QGP-side slope       (T > T_c), Bernhard 2019
    below = ETAS_MIN + slope_lo * np.maximum(T_C - T, 0.0)
    above = ETAS_MIN + slope_hi * np.maximum(T - T_C, 0.0)
    return np.where(T < T_C, below, above)


def etas_band(T):
    r"""90% credible band (lower, upper) reproducing published widths.

    Narrow (~+-0.025) near the minimum where data constrain eta/s tightly,
    widening on the QGP side (weak high-T constraint) -- as in Bernhard 2019 /
    JETSCAPE 2021.  Purely illustrative widths consistent with the cited work.
    """
    T = np.asarray(T, float)
    med = etas_median(T)
    # width grows away from T_c, asymmetric and larger on the QGP side
    w_lo = ETAS_MIN_ERR[0] + 0.16 * np.maximum(T - T_C, 0.0) \
        + 0.10 * np.maximum(T_C - T, 0.0)
    w_hi = ETAS_MIN_ERR[1] + 0.55 * np.maximum(T - T_C, 0.0) \
        + 0.18 * np.maximum(T_C - T, 0.0)
    lower = np.maximum(med - w_lo, KSS * 0.98)   # essentially bounded by KSS
    upper = med + w_hi
    return lower, upper


def make_figure():
    plt = C.apply_style()

    T = np.linspace(0.145, 0.400, 400)          # GeV (145 - 400 MeV)
    med = etas_median(T)
    lo, hi = etas_band(T)

    print("=" * 70)
    print("fig08_etas: eta/s(T) posterior band")
    print("  *** REPRODUCED-LITERATURE (Bernhard-Moreland-Bass 2019 et al.) ***")
    print("  *** NOT an original computation of this work.               ***")
    print("=" * 70)
    print(f"  KSS bound 1/(4 pi)          = {KSS:.4f}")
    print(f"  reproduced (eta/s)_min      = {ETAS_MIN:.3f} "
          f"(+{ETAS_MIN_ERR[1]:.3f} / -{ETAS_MIN_ERR[0]:.3f})  [Bernhard 2019]")
    imin = int(np.argmin(med))
    print(f"  median minimum on curve     = {med[imin]:.4f} at T = "
          f"{T[imin]*1e3:.0f} MeV (~ T_c = {T_C*1e3:.0f} MeV)")
    print(f"  median at T = 400 MeV       = {med[-1]:.4f} "
          f"[{lo[-1]:.4f}, {hi[-1]:.4f}]")
    print("-" * 70)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    # 90% credible band + median
    ax.fill_between(T * 1e3, lo, hi, color="#4477AA", alpha=0.30, lw=0,
                    label="90% credible band (reproduced)")
    ax.plot(T * 1e3, med, color="#22447A", lw=2.2,
            label=r"median $\eta/s(T)$ (Bernhard $et\,al.$ 2019)")

    # KSS bound
    ax.axhline(KSS, color="#CC3311", lw=1.6, ls=(0, (5, 2)),
               label=r"KSS bound $1/4\pi\approx0.080$")

    # pseudocritical temperature marker
    ax.axvline(T_C * 1e3, color="0.55", lw=1.0, ls=":")
    ax.text(T_C * 1e3 + 3, 0.44, r"$T_c\simeq154$ MeV", color="0.4",
            fontsize=8.5, rotation=90, va="top", ha="left")

    # reproduced minimum with its published error bar
    ax.errorbar([T_C * 1e3], [ETAS_MIN],
                yerr=[[ETAS_MIN_ERR[0]], [ETAS_MIN_ERR[1]]],
                fmt="o", ms=5, color="#EE7733", capsize=3, lw=1.4,
                zorder=5,
                label=r"$(\eta/s)_{\min}=0.085^{+0.026}_{-0.025}$ [Bernhard 2019]")

    ax.set_xlim(145, 400)
    ax.set_ylim(0.0, 0.62)
    ax.set_xlabel(r"temperature  $T$  [MeV]")
    ax.set_ylabel(r"specific shear viscosity  $\eta/s$")
    ax.set_title("Specific shear viscosity of the QGP\n"
                 "(REPRODUCED from Bernhard-Moreland-Bass 2019; not this work)",
                 fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.8, handlelength=1.9,
              borderaxespad=0.5, framealpha=0.9)

    # explicit on-canvas provenance stamp
    ax.text(0.985, 0.03, "REPRODUCED-LITERATURE", transform=ax.transAxes,
            fontsize=7.5, color="#AA3322", ha="right", va="bottom",
            style="italic", alpha=0.85)

    paths = C.savefig_dual(fig, "fig08_etas")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    return paths


if __name__ == "__main__":
    make_figure()
