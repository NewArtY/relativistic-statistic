# -*- coding: utf-8 -*-
"""
fig00_overview.py  ->  fig00_overview.{pdf,png}

WHAT IT SHOWS
-------------
A single-panel *bare phase-space occupation* overview that places all six
statistical families used in the paper on one log-log axis versus the
dimensionless variable

        x = (eps - mu) / T ,

so that the qualitative contrast between EXPONENTIAL tails (the classical
and quantum ideal gases) and POWER-LAW tails (the non-extensive
generalizations) is immediately visible.  This is the figure referenced as
``fig:distributions`` in the Notation and Conventions section.

Curves (bare mean occupation, a function of x only):
    * Fermi-Dirac      n = 1/(e^x + 1)          -> exponential tail, ceiling 1
    * Bose-Einstein    n = 1/(e^x - 1)          -> exponential tail, diverges x->0+
    * Maxwell-Boltzmann n = e^{-x}              -> classical exponential
    * Maxwell-Juttner  (bare)  = e^{-x}          -> COINCIDES with MB at the level
                                                    of the bare occupation; the two
                                                    differ only once the relativistic
                                                    phase-space measure is included
                                                    (see fig01_maxwell_juttner).
    * Tsallis  q>1     n = exp_q(-x)            -> BARE power law  ~ x^{-1/(q-1)}
    * Kaniadakis kappa>0  n = exp_k(-x)         -> BARE power law  ~ x^{-1/kappa}

Thin reference lines mark the BARE asymptotic slopes.

CONVENTION (see 00_conventions.tex)
-----------------------------------
Bare-occupation tail indices are used throughout this figure:
    Tsallis     f_q     ~ eps^{-1/(q-1)}        (NOT -q/(q-1), which is the
                                                 Lorentz-invariant yield index)
    Kaniadakis  f_kappa ~ eps^{-1/kappa}
Coldness is denoted zeta = m/T (not shown here; this figure depends on x only).
The invariant-yield index -q/(q-1) is reserved for the p_T-spectrum figures.

PROVENANCE
----------
COMPUTED (pure analytic evaluation of the (deformed) exponentials; no data).

Run:  python fig00_overview.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402

import _common as C  # noqa: E402
from _common import apply_style, savefig_dual, exp_q, exp_k  # noqa: E402


def main():
    plt = apply_style()

    # Deformation parameters chosen so the two power laws have DISTINCT slopes
    # (an overview of the family); the matched case kappa = q-1 is fig05.
    q_val = 1.2      # Tsallis   -> bare slope -1/(q-1) = -5
    k_val = 0.3      # Kaniadakis-> bare slope -1/kappa = -3.333...

    x = np.logspace(np.log10(0.03), np.log10(60.0), 1000)

    n_fd = 1.0 / (np.exp(x) + 1.0)
    n_be = 1.0 / (np.exp(x) - 1.0)
    n_mb = np.exp(-x)
    n_mj = np.exp(-x)                 # bare Maxwell-Juttner == Maxwell-Boltzmann
    n_ts = exp_q(-x, q_val)
    n_ka = exp_k(-x, k_val)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    # --- quantum + classical (exponential-tail) family ---------------------
    ax.plot(x, n_fd, ls="-", color="#0077BB", lw=2.0, label="Fermi-Dirac")
    ax.plot(x, n_be, ls="-", color="#009988", lw=2.0, label="Bose-Einstein")
    ax.plot(x, n_mb, ls="--", color="#555555", lw=2.0,
            label="Maxwell-Boltzmann")
    ax.plot(x, n_mj, ls=":", color="#EE7733", lw=2.2,
            label="Maxwell-Juttner (bare)")

    # --- non-extensive (power-law-tail) family -----------------------------
    ax.plot(x, n_ts, ls="-.", color="#CC3311", lw=2.0,
            label=rf"Tsallis $q={q_val}$")
    ax.plot(x, n_ka, ls="-.", color="#AA3377", lw=2.0,
            label=rf"Kaniadakis $\kappa={k_val}$")

    # --- bare power-law reference slopes -----------------------------------
    s_ts = -1.0 / (q_val - 1.0)       # -5.0
    s_ka = -1.0 / k_val               # -3.333...
    xr = np.array([6.0, 55.0])
    # anchor each reference line onto its curve at x = 6
    a_ts = exp_q(-6.0, q_val) * (xr / 6.0) ** s_ts
    a_ka = exp_k(-6.0, k_val) * (xr / 6.0) ** s_ka
    ax.plot(xr, a_ts, ls=(0, (1, 1)), color="#CC3311", lw=1.0, alpha=0.6)
    ax.plot(xr, a_ka, ls=(0, (1, 1)), color="#AA3377", lw=1.0, alpha=0.6)
    ax.annotate(rf"$\propto x^{{{s_ts:.0f}}}$", xy=(45, a_ts[-1] * 1.6),
                color="#CC3311", fontsize=8, ha="center")
    ax.annotate(rf"$\propto x^{{{s_ka:.2f}}}$", xy=(45, a_ka[-1] * 1.8),
                color="#AA3377", fontsize=8, ha="center")

    # note that bare MJ coincides with MB
    ax.annotate("MB $\\equiv$ MJ (bare)\n$e^{-x}$", xy=(2.0, np.exp(-2.0)),
                xytext=(0.06, 3e-3), fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 60.0)
    ax.set_ylim(1e-7, 40.0)
    ax.set_xlabel(r"$x = (\varepsilon - \mu)/T$")
    ax.set_ylabel(r"bare occupation $f(x)$")
    ax.set_title("Distribution overview: exponential vs power-law tails")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, ncol=1)

    paths = savefig_dual(fig, "fig00_overview")
    for p in paths:
        print("wrote", p)
    print(f"Tsallis q={q_val} -> bare slope {s_ts:.4f} ; "
          f"Kaniadakis kappa={k_val} -> bare slope {s_ka:.4f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
