# -*- coding: utf-8 -*-
"""
fig02_occupation.py  ->  fig02_occupation.{pdf,png}

WHAT IT SHOWS
-------------
The mean occupation number of a single-particle state as a function of the
dimensionless variable

        x = (eps - mu) / T ,

for the three ideal-gas statistics:

        Fermi-Dirac      n_FD = 1/(e^x + 1)   -> bounded by the Pauli ceiling 1
        Bose-Einstein    n_BE = 1/(e^x - 1)   -> diverges as x -> 0^+ (onset of
                                                 Bose-Einstein condensation)
        Maxwell-Boltzmann n_MB = e^{-x}       -> the common classical limit,
                                                 approached by both quantum
                                                 statistics for x >> 1.

The quantum-degeneracy zone |x| < 1, where the three curves differ
significantly, is shaded.

IMPORTANT (risk R19)
--------------------
These curves depend ONLY on the dimensionless x = (eps - mu)/T.  There is NO
"mc^2 = 0.5 T" label anywhere: the occupation numbers carry no separate
dependence on the rest energy, so quoting such a value would be misleading.
This figure is a pure illustration of occupation statistics.

CONVENTION (see 00_conventions.tex)
-----------------------------------
n_BE is physical only for x > 0 (mu < eps); it is drawn only there.  n_FD and
n_MB are drawn over the full x range.  Natural units c = hbar = k_B = 1.

PROVENANCE
----------
COMPUTED (pure analytic evaluation; no data, no free parameters).

Run:  python fig02_occupation.py
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
from _common import apply_style, savefig_dual  # noqa: E402


def main():
    plt = apply_style()

    x = np.linspace(-3.5, 6.0, 1400)
    x_pos = x[x > 0.02]                      # BE only for x > 0

    n_fd = 1.0 / (np.exp(x) + 1.0)
    n_mb = np.exp(-x)
    n_be = 1.0 / (np.exp(x_pos) - 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    # degeneracy zone |x| < 1
    ax.axvspan(-1.0, 1.0, color="#DDAA33", alpha=0.12, lw=0,
               label="degeneracy zone $|x|<1$")

    ax.plot(x, n_fd, ls="-", color="#0077BB", lw=2.2,
            label=r"Fermi-Dirac  $1/(e^{x}+1)$")
    ax.plot(x_pos, n_be, ls="--", color="#009988", lw=2.2,
            label=r"Bose-Einstein  $1/(e^{x}-1)$")
    ax.plot(x, n_mb, ls="-.", color="#CC3311", lw=2.2,
            label=r"Maxwell-Boltzmann  $e^{-x}$")

    # Pauli ceiling and x=0 guide
    ax.axhline(1.0, color="#555555", lw=0.8, ls=(0, (1, 1)), alpha=0.7)
    ax.annotate("Pauli ceiling  $n_{FD}\\to1$", xy=(-3.3, 1.0),
                xytext=(-3.3, 1.35), fontsize=8, color="#555555")
    ax.axvline(0.0, color="#555555", lw=0.6, alpha=0.4)
    ax.annotate("BE diverges\n$x\\to0^{+}$ (BEC)", xy=(0.05, 8.0),
                xytext=(0.7, 6.6), fontsize=8, color="#009988",
                arrowprops=dict(arrowstyle="->", color="#009988", lw=0.8))

    ax.set_xlim(-3.5, 6.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel(r"$x = (\varepsilon - \mu)/T$")
    ax.set_ylabel(r"mean occupation  $n(x)$")
    ax.set_title("FD / BE / MB mean occupation numbers")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    paths = savefig_dual(fig, "fig02_occupation")
    for p in paths:
        print("wrote", p)
    print("checks: n_FD(0)=%.4f (=1/2), n_MB(0)=%.4f (=1), "
          "n_FD/n_MB/n_BE -> e^-x for x>>1"
          % (1.0 / (np.exp(0) + 1.0), np.exp(0)))
    plt.close(fig)


if __name__ == "__main__":
    main()
