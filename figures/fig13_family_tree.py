# -*- coding: utf-8 -*-
"""
fig13_family_tree.py  ->  fig13_family_tree.{pdf,png}

WHAT IT SHOWS
-------------
A conceptual "family map" of the master occupation

        n(eps) = 1 / ( [exp_sigma(-beta (eps - mu))]^{-1} + a )

of Definition def:master-occupation and Table tab:master.  The reciprocal
of exp_sigma(-y) is used (not exp_sigma(+y)) so that the classical branch is
exp_sigma(-y) for every sigma: for sigma in {0, kappa} the two forms agree by
reciprocity, while for Tsallis (no reciprocity) only this form gives the
classical branch the power-law tail eps^{-1/(q-1)}.  The six members
are addressed by two discrete labels: the deformation label
sigma in {0, q, kappa} (the three rows) and the quantum sign
a in {0, +1, -1} (the three columns of the Boltzmann-Gibbs row).  The six
labelled nodes are

    Maxwell-Boltzmann  (non-relativistic baseline),
    Maxwell-Juttner    (sigma = 0, a = 0),
    Fermi-Dirac        (sigma = 0, a = +1),
    Bose-Einstein      (sigma = 0, a = -1),
    Tsallis            (sigma = q),
    Kaniadakis         (sigma = kappa).

Each node carries its occupation form and its tail behaviour (exponential for
the Boltzmann-Gibbs members, power law eps^{-1/(q-1)} for Tsallis and
eps^{-1/kappa} for Kaniadakis).  Converging arrows encode the universal
classical limit (Theorem thm:universal-limit): every member collapses onto
Maxwell-Juttner as the gas becomes dilute (z -> 0) and the deformation is
switched off (q -> 1, kappa -> 0).  The thermodynamic curvature sign is coded
on each member (R < 0 Fermi / effective repulsion, R > 0 Bose / effective
attraction, R = 0 Maxwell-Juttner / flat), with the honest note that the
Kaniadakis gas inverts the dilute-limit sign of the bosonic branch.

PROVENANCE
----------
SCHEMATIC / CONCEPTUAL.  No data and no numerical evaluation; this is a
labelled diagram of the algebraic structure of Section unified.

Run:  python fig13_family_tree.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import _common as C  # noqa: E402
from _common import apply_style, savefig_dual  # noqa: E402


# --- palette (CB-safe tokens from style.mplstyle) ---------------------------
COL = {
    "MB": "#555555",   # grey   -- non-relativistic baseline
    "MJ": "#EE7733",   # orange -- universal classical limit (highlight)
    "FD": "#0077BB",   # blue   -- Fermi, R < 0
    "BE": "#009988",   # teal   -- Bose,  R > 0
    "TS": "#CC3311",   # red    -- Tsallis
    "KA": "#AA3377",   # purple -- Kaniadakis
    "neg": "#0077BB",  # R < 0
    "pos": "#009988",  # R > 0
    "zero": "#555555",  # R = 0
}


def _tint(hexcol, frac=0.14):
    """Return a light tint of ``hexcol`` (fraction ``frac`` toward white)."""
    r, g, b = mcolors.to_rgb(hexcol)
    return (1 - frac * (1 - r), 1 - frac * (1 - g), 1 - frac * (1 - b))


def _cell(ax, cx, cy, w, h, edge, title, lines, title_size=10.5,
          body_size=8.6, lw=1.6, fill=None, z=2):
    """Draw a rounded node centred at (cx, cy) with a bold title and body."""
    if fill is None:
        fill = _tint(edge)
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=2.2",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        mutation_aspect=1.0, zorder=z,
    )
    ax.add_patch(box)
    ax.text(cx, cy + h / 2 - 0.30 * h, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=edge, zorder=z + 1)
    ax.text(cx, cy - 0.10 * h, "\n".join(lines), ha="center", va="center",
            fontsize=body_size, color="#222222", zorder=z + 1, linespacing=1.35)
    return box


def _badge(ax, x, y, text, color, size=8.0, z=6):
    """Small pill carrying a curvature sign."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight="bold", color="white", zorder=z + 1,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                      edgecolor="none"))


def _arrow(ax, xy0, xy1, color="#333333", lw=1.7, style="arc3,rad=0.0",
           z=4, alpha=1.0, ls="-"):
    a = FancyArrowPatch(
        xy0, xy1, connectionstyle=style, arrowstyle="-|>",
        mutation_scale=13, linewidth=lw, color=color, zorder=z,
        alpha=alpha, linestyle=ls, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a


def main():
    plt = apply_style()

    fig, ax = plt.subplots(figsize=(7.4, 7.05))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    ax.set_aspect("auto")
    # Reserve a clear top band for the (tall) master-occupation title so it
    # cannot touch the Maxwell-Boltzmann baseline node in the top-left corner.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.905, bottom=0.02)

    # ---- column (quantum-sign) headers -----------------------------------
    cx0, cx1, cx2 = 30.0, 56.0, 82.0   # a = 0, +1, -1 column centres
    head_y = 85.5
    for cx, sym, sub in ((cx0, r"$a=0$", "(classical)"),
                         (cx1, r"$a=+1$", "(fermions)"),
                         (cx2, r"$a=-1$", "(bosons)")):
        ax.text(cx, head_y, sym, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#111111")
        ax.text(cx, head_y - 3.0, sub, ha="center", va="center",
                fontsize=8.2, color="#444444")

    # ---- row (deformation-label) tags ------------------------------------
    ax.text(3.0, 68.5, r"$\sigma=0$", ha="center", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color="#111111")
    ax.text(3.0, 41.0, r"$\sigma=q$", ha="center", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color=COL["TS"])
    ax.text(3.0, 18.0, r"$\sigma=\kappa$", ha="center", va="center",
            rotation=90, fontsize=9.5, fontweight="bold", color=COL["KA"])

    # =====================================================================
    #  Boltzmann-Gibbs row (sigma = 0): three named cells
    # =====================================================================
    cw, ch = 24.0, 19.0
    bg_y = 68.5

    _cell(ax, cx0, bg_y, cw, ch, COL["MJ"], "Maxwell-Juttner",
          [r"$n=e^{-\beta(\varepsilon-\mu)}$",
           r"exponential tail", r"$(\sigma,a)=(0,0)$"],
          fill=_tint(COL["MJ"], 0.24), lw=2.2)
    _cell(ax, cx1, bg_y, cw, ch, COL["FD"], "Fermi-Dirac",
          [r"$n=\left(e^{\beta(\varepsilon-\mu)}+1\right)^{-1}$",
           r"exponential tail"])
    _cell(ax, cx2, bg_y, cw, ch, COL["BE"], "Bose-Einstein",
          [r"$n=\left(e^{\beta(\varepsilon-\mu)}-1\right)^{-1}$",
           r"exponential tail"])

    _badge(ax, cx0 + cw / 2 - 3.4, bg_y - ch / 2 + 2.4, r"$R=0$", COL["zero"])
    _badge(ax, cx1 + cw / 2 - 3.4, bg_y - ch / 2 + 2.4, r"$R<0$", COL["neg"])
    _badge(ax, cx2 + cw / 2 - 3.4, bg_y - ch / 2 + 2.4, r"$R>0$", COL["pos"])

    # =====================================================================
    #  Tsallis row (sigma = q): one wide node spanning the three columns
    # =====================================================================
    wide_w = (cx2 + cw / 2) - (cx0 - cw / 2)
    wide_cx = (cx0 - cw / 2 + cx2 + cw / 2) / 2.0
    ts_y = 41.0
    _cell(ax, wide_cx, ts_y, wide_w, 16.5, COL["TS"], "Tsallis  (non-extensive)",
          [r"$n=\left([\exp_{q}(-\beta(\varepsilon-\mu))]^{-1}+a\right)^{-1}$"
           r"$\qquad$ power-law tail $\;\sim\varepsilon^{-1/(q-1)}$",
           r"exchange sign $a$ selects the classical / Fermi / Bose "
           r"sub-branch $\;\Rightarrow\;\mathrm{sgn}\,R=-\,\mathrm{sgn}\,a$"],
          body_size=8.8)

    # =====================================================================
    #  Kaniadakis row (sigma = kappa): one wide node
    # =====================================================================
    ka_y = 18.0
    _cell(ax, wide_cx, ka_y, wide_w, 16.5, COL["KA"], "Kaniadakis  (non-extensive)",
          [r"$n=\left(\exp_{\kappa}[\beta(\varepsilon-\mu)]+a\right)^{-1}$"
           r"$\qquad$ power-law tail $\;\sim\varepsilon^{-1/\kappa}$",
           r"classical branch $R<0$; dilute-limit sign inversion of the "
           r"Bose branch $\;(\,|\kappa|<\frac{1}{4}\,)$"],
          body_size=8.8)

    # =====================================================================
    #  Maxwell-Boltzmann baseline node (top-left corner)
    # =====================================================================
    _cell(ax, 15.5, 93.5, 27.0, 9.5, COL["MB"], "",
          ["Maxwell-Boltzmann", "non-relativistic baseline"],
          body_size=8.2, title_size=1.0, lw=1.4,
          fill=_tint(COL["MB"], 0.10))

    # =====================================================================
    #  Convergence arrows -> universal classical limit (Maxwell-Juttner)
    # =====================================================================
    # MB baseline  ->  Maxwell-Juttner  (relativistic lift; NR limit zeta->inf)
    _arrow(ax, (17.5, 88.6), (26.5, 78.4), color=COL["MB"], lw=1.6,
           style="arc3,rad=-0.15")
    ax.text(10.8, 82.5, "relativistic\n" r"$\zeta=m/T$",
            ha="center", va="center", fontsize=7.6, color=COL["MB"])

    # Fermi-Dirac  ->  Maxwell-Juttner   (dilute limit z -> 0)
    _arrow(ax, (cx1 - cw / 2, bg_y - 1.0), (cx0 + cw / 2, bg_y - 1.0),
           color=COL["FD"], lw=1.7)
    # Bose-Einstein -> Maxwell-Juttner   (arc bowing above the row)
    _arrow(ax, (cx2 - cw / 2 + 1.0, bg_y + ch / 2),
           (cx0 + cw / 2 - 1.0, bg_y + ch / 2),
           color=COL["BE"], lw=1.7, style="arc3,rad=0.22")
    ax.text((cx1 + cx2) / 2.0, bg_y + ch / 2 + 4.0, r"dilute limit $\;z\to0$",
            ha="center", va="center", fontsize=8.0, color="#333333")

    # Tsallis   ->  Maxwell-Juttner      (weak deformation q -> 1)
    _arrow(ax, (cx0, ts_y + 16.5 / 2), (cx0, bg_y - ch / 2),
           color=COL["TS"], lw=1.7)
    ax.text(cx0 + 6.0, (ts_y + bg_y) / 2.0 + 1.0, r"$q\to1$",
            ha="left", va="center", fontsize=8.4, color=COL["TS"])

    # Kaniadakis ->  Maxwell-Juttner     (weak deformation kappa -> 0)
    _arrow(ax, (cx0 - cw / 2 - 1.0, ka_y),
           (cx0 - cw / 2 - 1.0, bg_y - ch / 2 + 1.0),
           color=COL["KA"], lw=1.7, style="arc3,rad=-0.28")
    ax.text(9.5, (ka_y + ts_y) / 2.0 + 2.0, r"$\kappa\to0$",
            ha="center", va="center", fontsize=8.4, color=COL["KA"])

    # highlight tag on the universal limit
    ax.text(cx0, bg_y + ch / 2 + 2.2, "universal classical limit",
            ha="center", va="center", fontsize=8.2, style="italic",
            color=COL["MJ"])

    # =====================================================================
    #  Curvature key (bottom strip)
    # =====================================================================
    key_y = 3.0
    ax.text(6.0, key_y, "curvature:", ha="left", va="center", fontsize=8.2,
            color="#222222", fontweight="bold")
    _badge(ax, 27.0, key_y, r"$R<0$", COL["neg"], size=7.6)
    ax.text(30.5, key_y, "Fermi (repulsion)", ha="left", va="center",
            fontsize=7.8, color="#333333")
    _badge(ax, 55.0, key_y, r"$R>0$", COL["pos"], size=7.6)
    ax.text(58.5, key_y, "Bose (attraction)", ha="left", va="center",
            fontsize=7.8, color="#333333")
    _badge(ax, 81.0, key_y, r"$R=0$", COL["zero"], size=7.6)
    ax.text(84.5, key_y, "flat", ha="left", va="center",
            fontsize=7.8, color="#333333")

    fig.suptitle(
        r"One master occupation "
        r"$n=1/([\exp_{\sigma}(-\beta(\varepsilon-\mu))]^{-1}+a)$"
        r"$\;\to\;$ six members by two labels $(\sigma,a)$",
        fontsize=10.5, y=0.975)

    paths = savefig_dual(fig, "fig13_family_tree")
    for p in paths:
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
