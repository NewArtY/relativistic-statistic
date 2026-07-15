# -*- coding: utf-8 -*-
"""
fig05_tails.py  ->  fig05_tails.{pdf,png}

WHAT IT SHOWS
-------------
A log-log comparison of the high-energy tails of the three occupations used
in the paper, all normalized to f = 1 at the reference momentum p = 0.1 GeV
(with m = 0, T = 0.1 GeV this is x = eps/T = 1):

    * Maxwell-Juttner  f = e^{-x}          -> EXPONENTIAL: curves steadily
                                              downward on a log-log axis
                                              (faster than any power law).
    * Tsallis   f = exp_q(-x),  q = 1.15   -> POWER LAW, slope -1/(q-1).
    * Kaniadakis f = exp_kappa(-x), kappa = 0.15 -> POWER LAW, slope -1/kappa.

The Tsallis and Kaniadakis parameters are chosen through the TAIL-INDEX
matching relation

        kappa = q - 1              (regime (i) of 00_conventions.tex),

so that both non-extensive tails share the SINGLE bare asymptotic slope

        -1/(q-1) = -1/kappa = -1/0.15 = -6.667 .

At high eps the Tsallis and Kaniadakis curves become PARALLEL straight lines
of this common slope (verified numerically below and annotated on the plot),
while the Maxwell-Juttner exponential falls away below both.

CONVENTION / RISK (O1, see 00_conventions.tex)
----------------------------------------------
This figure uses regime (i), kappa = q - 1, with the SINGLE bare tail index
-1/(q-1) = -1/kappa for BOTH curves.  It does NOT use the weak-deformation
relation q - 1 ~ kappa^2/2 (which would give kappa ~ 0.55 for q = 1.15, a
factor ~13 different) and does NOT mix in the invariant-yield index -q/(q-1).
Parameters: T = 0.1 GeV, mu = 0, m = 0.

PROVENANCE
----------
COMPUTED (pure evaluation of the (deformed) exponentials; no data).

Run:  python fig05_tails.py
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

    # --- tail-index matching: kappa = q - 1 (regime (i)) -------------------
    q_val = 1.15
    k_val = q_val - 1.0          # = 0.15
    shared_slope = -1.0 / (q_val - 1.0)
    assert abs(shared_slope - (-1.0 / k_val)) < 1e-12
    print(f"q = {q_val},  kappa = q-1 = {k_val:.3f}")
    print(f"shared bare tail slope  -1/(q-1) = -1/kappa = {shared_slope:.4f}")

    T = 0.1                       # GeV
    p_ref = 0.1                   # GeV  ->  x_ref = p_ref/T = 1 (m = 0)
    x_ref = p_ref / T

    x = np.logspace(0.0, 4.0, 1600)     # x = eps/T, from the reference upward

    # normalize each curve to 1 at x_ref
    f_mj = np.exp(-x) / np.exp(-x_ref)
    f_ts = exp_q(-x, q_val) / exp_q(-x_ref, q_val)
    f_ka = exp_k(-x, k_val) / exp_k(-x_ref, k_val)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))

    ax.plot(x, f_mj, ls="-", color="#0077BB", lw=2.2,
            label=r"Maxwell-Juttner  $e^{-x}$ (exponential)")
    ax.plot(x, f_ts, ls="--", color="#CC3311", lw=2.2,
            label=rf"Tsallis  $q={q_val}$")
    ax.plot(x, f_ka, ls="-.", color="#AA3377", lw=2.2,
            label=rf"Kaniadakis  $\kappa={k_val:g}$")

    # shared power-law reference line, slope -1/(q-1) = -1/kappa
    x0 = 30.0
    anchor = exp_q(-x0, q_val) / exp_q(-x_ref, q_val)
    ref = anchor * (x / x0) ** shared_slope
    mask = x > 12.0
    ax.plot(x[mask], ref[mask], ls=(0, (1, 1)), color="#555555", lw=1.2,
            alpha=0.8, label=rf"shared slope ${shared_slope:.2f}$")

    ax.annotate(rf"$-1/(q-1) = -1/\kappa = {shared_slope:.2f}$",
                xy=(400.0, ref[np.argmin(np.abs(x - 400.0))]),
                xytext=(1.5, 1e-6), fontsize=8.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 1e4)
    ax.set_ylim(1e-13, 2.0)
    ax.set_xlabel(r"$x = \varepsilon/T$   (normalized at $p=0.1$ GeV)")
    ax.set_ylabel(r"bare occupation  $f(x)/f(x_{\mathrm{ref}})$")
    ax.set_title("Power-law tail comparison: MJ / Tsallis / Kaniadakis")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    paths = savefig_dual(fig, "fig05_tails")
    for p in paths:
        print("wrote", p)

    # --- verify the two non-extensive slopes coincide at high x ------------
    xa, xb = 1e3, 1e4
    s_ts = (np.log(exp_q(-xb, q_val)) - np.log(exp_q(-xa, q_val))) / \
           (np.log(xb) - np.log(xa))
    s_ka = (np.log(exp_k(-xb, k_val)) - np.log(exp_k(-xa, k_val))) / \
           (np.log(xb) - np.log(xa))
    print("VERIFIED asymptotic slopes (finite difference over [1e3, 1e4]):")
    print(f"  Tsallis    q={q_val}   slope = {s_ts:.4f}")
    print(f"  Kaniadakis kappa={k_val:g}  slope = {s_ka:.4f}")
    print(f"  analytic shared slope        = {shared_slope:.4f}")
    print(f"  |Tsallis - Kaniadakis| = {abs(s_ts - s_ka):.2e} "
          f"({100*abs(s_ts - s_ka)/abs(shared_slope):.3f}% of shared slope)")
    plt.close(fig)


if __name__ == "__main__":
    main()
