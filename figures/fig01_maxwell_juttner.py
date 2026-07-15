# -*- coding: utf-8 -*-
"""
fig01_maxwell_juttner.py  ->  fig01_maxwell_juttner.{pdf,png}

WHAT IT SHOWS
-------------
The normalized Maxwell-Juttner momentum distribution

        P(u) du = u^2 exp[-zeta * sqrt(1 + u^2)] / N(zeta)  du ,
        u = p / m ,   zeta = m / T ,   N(zeta) = K_2(zeta) / zeta ,

for three coldness values that span the relativistic regimes:
        zeta = 0.5  (ultra-relativistic / hot),
        zeta = 1    (mildly relativistic),
        zeta = 5    (non-relativistic / cold).
The distribution is the radial (momentum-magnitude) density 4*pi*u^2 f_MJ,
i.e. the probability that |p|/m lies in [u, u+du].  As zeta decreases the
peak shifts to larger u and a long relativistic tail develops.

For comparison the NON-RELATIVISTIC Maxwell-Boltzmann momentum distribution
at the SAME temperature (same zeta) is shown as a grey dash-dot curve for the
cold case zeta = 5:

        P_MB(u) du = u^2 exp[-(zeta/2) u^2] / N_MB  du ,

obtained from the kinetic energy p^2/(2m).  At zeta = 5 the Maxwell-Juttner
and Maxwell-Boltzmann curves nearly coincide, confirming the correct
non-relativistic limit (and the K_2(zeta) normalization).

CONVENTION (see 00_conventions.tex)
-----------------------------------
Coldness is labelled zeta = m/T (NOT z; z is reserved for the fugacity).
zeta >> 1: non-relativistic; zeta ~ 1: mildly relativistic; zeta << 1:
ultra-relativistic.  The exact normalization uses the Macdonald function
N(zeta) = K_2(zeta)/zeta, verified numerically to <1e-6 below.

PROVENANCE
----------
COMPUTED (pure computation; scipy.special modified Bessel + numerical
normalization check).

Run:  python fig01_maxwell_juttner.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

import _common as C  # noqa: E402
from _common import apply_style, savefig_dual, bessel_k  # noqa: E402


def mj_pdf(u, zeta):
    """Normalized Maxwell-Juttner momentum-magnitude pdf, N = K_2(zeta)/zeta."""
    norm = bessel_k(2, zeta) / zeta
    return u ** 2 * np.exp(-zeta * np.sqrt(1.0 + u ** 2)) / norm


def mb_pdf(u, zeta):
    """Non-relativistic Maxwell-Boltzmann momentum-magnitude pdf (same T)."""
    # N_MB = int_0^inf u^2 exp(-(zeta/2)u^2) du = (sqrt(pi)/4)(zeta/2)^{-3/2}
    norm = (np.sqrt(np.pi) / 4.0) * (zeta / 2.0) ** (-1.5)
    return u ** 2 * np.exp(-(zeta / 2.0) * u ** 2) / norm


def main():
    plt = apply_style()

    zetas = [0.5, 1.0, 5.0]
    styles = ["-", "--", "-."]
    colors = ["#CC3311", "#0077BB", "#009988"]

    u = np.linspace(0.0, 8.0, 1600)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    # --- normalization self-check (acceptance: |int - 1| < 1e-3) -----------
    print("Maxwell-Juttner normalization check (int P(u) du should be 1):")
    for zeta in zetas:
        val, _ = integrate.quad(mj_pdf, 0.0, np.inf, args=(zeta,), limit=200)
        print(f"  zeta={zeta:>4}:  integral = {val:.8f}   "
              f"(err = {abs(val - 1.0):.2e})")

    peaks = []
    for zeta, ls, col in zip(zetas, styles, colors):
        p = mj_pdf(u, zeta)
        ax.plot(u, p, ls=ls, color=col, lw=2.0,
                label=rf"MJ  $\zeta={zeta:g}$")
        peaks.append(u[np.argmax(p)])

    # non-relativistic MB at the cold reference zeta = 5 (should overlap MJ)
    zeta_ref = 5.0
    ax.plot(u, mb_pdf(u, zeta_ref), ls=(0, (3, 1, 1, 1)), color="#555555",
            lw=1.8, label=rf"MB (non-rel.)  $\zeta={zeta_ref:g}$")

    # annotate the peak shift
    for zeta, up in zip(zetas, peaks):
        yp = mj_pdf(np.array([up]), zeta)[0]
        ax.plot([up], [yp], marker="o", ms=4, color="k", zorder=5)
    ax.annotate("peak shifts to\nlarger $u$ as $\\zeta\\downarrow$",
                xy=(peaks[0], mj_pdf(np.array([peaks[0]]), zetas[0])[0]),
                xytext=(4.0, 0.55), fontsize=8, color="k",
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8))

    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"$u = p/m$")
    ax.set_ylabel(r"$4\pi u^2 f_{\mathrm{MJ}}(u)$   (normalized)")
    ax.set_title(r"Maxwell-Juttner distribution vs coldness $\zeta=m/T$")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    paths = savefig_dual(fig, "fig01_maxwell_juttner")
    for pth in paths:
        print("wrote", pth)
    print("MJ peak positions u* =", [f"{p:.3f}" for p in peaks],
          "for zeta =", zetas)
    plt.close(fig)


if __name__ == "__main__":
    main()
