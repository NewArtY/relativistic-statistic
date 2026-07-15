# -*- coding: utf-8 -*-
"""
fig04_kaniadakis.py  ->  fig04_kaniadakis.{pdf,png}

WHAT IT SHOWS
-------------
The Kaniadakis kappa-exponential occupation

        f_kappa(x) = exp_kappa(-x)
                   = ( sqrt(1 + kappa^2 x^2) - kappa x )^{1/kappa} ,
        x = (eps - mu) / T ,

for kappa = 0, 0.1, 0.3, 0.5 on a log-log axis.  Each deformed curve follows
the ordinary exponential e^{-x} up to a crossover at x_tr ~ 1/kappa and then
bends into the BARE power-law tail

        f_kappa(x) ~ (2 kappa x)^{-1/kappa} ~ x^{-1/kappa} .

Thin reference lines mark the slope -1/kappa:
        kappa = 0.1 -> -10,   kappa = 0.3 -> -3.33,   kappa = 0.5 -> -2.
Faint vertical dotted lines mark the crossover x_tr = 1/kappa.  The
kappa = 0 curve is the ordinary exponential (no power-law tail).

CONVENTION (see 00_conventions.tex)
-----------------------------------
kappa is DIMENSIONLESS (|kappa| < 1); the argument x = beta(eps - mu) is
itself dimensionless, so the tail index -1/kappa is well defined and
self-consistent (code slope == caption slope).  This is the BARE-occupation
index; the invariant-yield figures use their own measure separately.

Physical parameters (eps <-> x mapping):
        mu = 0,  T = 0.1 GeV,  m = 0  (massless: threshold at x = 0),
so eps = x*T.

PROVENANCE
----------
COMPUTED (pure evaluation of exp_kappa; no data).

Run:  python fig04_kaniadakis.py
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
from _common import apply_style, savefig_dual, exp_k  # noqa: E402


def main():
    plt = apply_style()

    T = 0.1                       # GeV (massless case: eps = x*T)
    mu = 0.0
    m = 0.0
    print(f"params: mu={mu}, T={T} GeV, m={m} (massless), x = eps/T")

    kappas = [0.0, 0.1, 0.3, 0.5]
    styles = ["-", "--", "-.", ":"]
    colors = ["#555555", "#0077BB", "#009988", "#CC3311"]

    x = np.logspace(-1.0, 4.0, 1600)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))

    for kap, ls, col in zip(kappas, styles, colors):
        fk = exp_k(-x, kap)
        ax.plot(x, fk, ls=ls, color=col, lw=2.0,
                label=rf"$\kappa={kap:g}$")
        if kap > 0.0:
            slope = -1.0 / kap
            x0 = 300.0
            ref = exp_k(-x0, kap) * (x / x0) ** slope
            mask = x > 40.0
            ax.plot(x[mask], ref[mask], ls=(0, (1, 1)), color=col,
                    lw=1.0, alpha=0.6)
            # crossover marker x_tr = 1/kappa
            x_tr = 1.0 / kap
            ax.axvline(x_tr, color=col, ls=(0, (1, 3)), lw=0.8, alpha=0.5)
            ax.annotate(rf"$x^{{{slope:.2f}}}$",
                        xy=(3e3, exp_k(-3e3, kap) * 3),
                        color=col, fontsize=8, ha="center")

    ax.annotate(r"crossover $x_{\mathrm{tr}}=1/\kappa$",
                xy=(1.0 / 0.3, exp_k(-1.0 / 0.3, 0.3)),
                xytext=(0.13, 3e-4), fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e-11, 1.5)
    ax.set_xlabel(r"$x = (\varepsilon-\mu)/T$")
    ax.set_ylabel(r"$f_\kappa(x) = \exp_\kappa(-x)$")
    ax.set_title(r"Kaniadakis $\kappa$-distribution: bare tail $\sim x^{-1/\kappa}$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    paths = savefig_dual(fig, "fig04_kaniadakis")
    for p in paths:
        print("wrote", p)

    print("log-log tail slope check (large-x finite difference):")
    for kap in kappas[1:]:
        xa, xb = 1e3, 1e4
        s = (np.log(exp_k(-xb, kap)) - np.log(exp_k(-xa, kap))) / \
            (np.log(xb) - np.log(xa))
        print(f"  kappa={kap:g}: measured {s:.3f}  vs  -1/kappa = "
              f"{-1.0/kap:.3f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
