# -*- coding: utf-8 -*-
"""
fig03_tsallis.py  ->  fig03_tsallis.{pdf,png}

WHAT IT SHOWS
-------------
The Tsallis q-exponential occupation

        f_q(x) = exp_q(-x) = [1 + (q-1) x]^{-1/(q-1)} ,
        x = (eps - mu) / T ,

for q = 1.0, 1.1, 1.2, 1.5.  Two panels:
    (a) LINEAR scale, showing how the near-thermal region fattens as q grows;
    (b) LOG-LOG scale, showing the BARE power-law asymptote

            f_q(x) ~ x^{-1/(q-1)}      (q > 1),

        with thin reference lines of slope -1/(q-1):
            q = 1.1 -> -10,   q = 1.2 -> -5,   q = 1.5 -> -2.
    The q = 1.0 curve is the ordinary exponential e^{-x} (no power-law tail).

CONVENTION / RISK (R08 / O5, see 00_conventions.tex)
----------------------------------------------------
The plotted quantity is the BARE phase-space occupation, whose tail index is

        -1/(q-1)      (NOT -q/(q-1)).

The index -q/(q-1) = 1 + 1/(q-1) belongs to the Lorentz-invariant p_T yield
dN/(p_T dp_T dy), which carries one extra power of momentum from the
phase-space measure and is used only in the spectra-fit figures.  The
reference-line slopes and this docstring both use -1/(q-1) so that
code, caption, and log-log fit slope agree.

Physical parameters (for the eps <-> x mapping):
        mu = 0,  T = 0.1 GeV,  m = m_pi = 0.13957 GeV,
so the comoving energy is eps = x*T and the kinematic threshold is
x_th = m/T = zeta = 1.3957.

PROVENANCE
----------
COMPUTED (pure evaluation of exp_q; no data).

Run:  python fig03_tsallis.py
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
from _common import apply_style, savefig_dual, exp_q, coldness  # noqa: E402


def main():
    plt = apply_style()

    T = 0.1                       # GeV
    m = C.M_PION                  # 0.13957 GeV
    mu = 0.0
    zeta = coldness(m, T)         # threshold x_th = m/T
    print(f"params: mu={mu}, T={T} GeV, m=m_pi={m:.5f} GeV, "
          f"threshold x_th = m/T = {zeta:.4f}")

    qs = [1.0, 1.1, 1.2, 1.5]
    styles = ["-", "--", "-.", ":"]
    colors = ["#555555", "#0077BB", "#009988", "#CC3311"]

    x_lin = np.linspace(0.0, 20.0, 1200)
    x_log = np.logspace(np.log10(zeta), 4.0, 1400)   # from threshold upward

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # --- panel (a): linear -------------------------------------------------
    for q, ls, col in zip(qs, styles, colors):
        axa.plot(x_lin, exp_q(-x_lin, q), ls=ls, color=col, lw=2.0,
                 label=rf"$q={q:g}$")
    axa.set_xlim(0.0, 20.0)
    axa.set_ylim(0.0, 1.02)
    axa.set_xlabel(r"$x = (\varepsilon-\mu)/T$")
    axa.set_ylabel(r"$f_q(x) = \exp_q(-x)$")
    axa.set_title("(a) linear scale")
    axa.grid(True, which="major", alpha=0.25)
    axa.legend(loc="upper right", fontsize=8)

    # --- panel (b): log-log with bare -1/(q-1) reference lines -------------
    for q, ls, col in zip(qs, styles, colors):
        fq = exp_q(-x_log, q)
        axb.plot(x_log, fq, ls=ls, color=col, lw=2.0, label=rf"$q={q:g}$")
        if q > 1.0:
            slope = -1.0 / (q - 1.0)          # BARE index
            x0 = 300.0
            ref = exp_q(-x0, q) * (x_log / x0) ** slope
            mask = x_log > 30.0
            axb.plot(x_log[mask], ref[mask], ls=(0, (1, 1)),
                     color=col, lw=1.0, alpha=0.6)
            # slope annotation near the tail
            axb.annotate(rf"$x^{{{slope:.0f}}}$",
                         xy=(2.5e3, exp_q(-2.5e3, q) * 3),
                         color=col, fontsize=8, ha="center")

    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(zeta, 1e4)
    axb.set_ylim(1e-12, 1.5)
    axb.set_xlabel(r"$x = (\varepsilon-\mu)/T$")
    axb.set_ylabel(r"$f_q(x)$")
    axb.set_title(r"(b) log-log: bare tail $\sim x^{-1/(q-1)}$")
    axb.grid(True, which="major", alpha=0.25)
    axb.legend(loc="lower left", fontsize=8)

    fig.suptitle("Tsallis $q$-exponential occupation", fontsize=12)

    paths = savefig_dual(fig, "fig03_tsallis")
    for p in paths:
        print("wrote", p)

    # numeric slope verification (acceptance: matches -1/(q-1) to <2%)
    print("log-log tail slope check (large-x finite difference):")
    for q in qs[1:]:
        xa, xb = 1e3, 1e4
        s = (np.log(exp_q(-xb, q)) - np.log(exp_q(-xa, q))) / \
            (np.log(xb) - np.log(xa))
        print(f"  q={q:g}: measured {s:.3f}  vs  -1/(q-1) = "
              f"{-1.0/(q-1.0):.3f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
