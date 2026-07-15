# -*- coding: utf-8 -*-
"""
fig14_bayesian_workflow.py  ->  fig14_bayesian_workflow.{pdf,png}

WHAT IT SHOWS
-------------
A conceptual flow diagram of the Bayesian inference workflow of Section
bayes, the inverse map that complements the forward "family tree" of
Section unified.  Reading the loop:

    parameters  theta = {T_kin, <beta_T>, n}  (and, in general, T, mu, q, kappa)
        --> forward model  (Boltzmann-Gibbs blast-wave spectrum, Def. bgbw)
        --> predicted spectrum  m(theta);

    data (ALICE identified-particle p_T spectra)  together with the
    likelihood (Gaussian in the log-yield, carrying the Gaussian-process
    model-discrepancy covariance) and the prior (with the hard physical
    boundaries: subluminal beta_s < 1 and the KSS bound eta/s >= 1/4pi)
        --> posterior  p(theta | x);

    the affine-invariant ensemble sampler (emcee)
        --> posterior with credible intervals and a posterior-predictive check,

    and the sampler proposes new parameters, closing the loop.

Two features are singled out as the methodological distinctives: the
correlated model-discrepancy term inside the likelihood and the hard physical
boundaries inside the prior.

PROVENANCE
----------
SCHEMATIC / CONCEPTUAL.  No data and no numerical evaluation; a labelled
diagram of the inference pipeline described in Section bayes.

Run:  python fig14_bayesian_workflow.py
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
    "param": "#0077BB",   # blue    -- parameters / sampler (inference side)
    "model": "#EE7733",   # orange  -- forward model / prediction
    "data": "#009988",    # teal    -- measured data
    "like": "#CC3311",    # red     -- likelihood
    "prior": "#AA3377",   # purple  -- prior
    "post": "#333333",    # dark    -- posterior
    "out": "#009988",     # teal    -- output / results
    "flag": "#CC3311",    # callout accent
}


def _tint(hexcol, frac=0.15):
    r, g, b = mcolors.to_rgb(hexcol)
    return (1 - frac * (1 - r), 1 - frac * (1 - g), 1 - frac * (1 - b))


def _box(ax, cx, cy, w, h, edge, title, lines, title_size=9.8,
         body_size=8.2, lw=1.7, fill=None, z=2):
    if fill is None:
        fill = _tint(edge)
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.5,rounding_size=2.0",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        mutation_aspect=1.0, zorder=z))
    ax.text(cx, cy + h / 2 - 0.26 * h, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=edge, zorder=z + 1)
    ax.text(cx, cy - 0.12 * h, "\n".join(lines), ha="center", va="center",
            fontsize=body_size, color="#222222", zorder=z + 1, linespacing=1.35)


def _arrow(ax, xy0, xy1, color="#333333", lw=1.8, style="arc3,rad=0.0",
           z=3, ls="-", ms=13):
    ax.add_patch(FancyArrowPatch(
        xy0, xy1, connectionstyle=style, arrowstyle="-|>",
        mutation_scale=ms, linewidth=lw, color=color, zorder=z,
        linestyle=ls, shrinkA=3, shrinkB=3))


def _callout(ax, x, y, text, color, w=27.0, h=9.0, size=7.8):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.6",
        linewidth=1.2, edgecolor=color, facecolor="white",
        linestyle=(0, (3, 2)), zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=color, zorder=6, linespacing=1.3, fontweight="bold")


def main():
    plt = apply_style()

    fig, ax = plt.subplots(figsize=(7.5, 5.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    # =====================================================================
    #  Top row: theta -> forward model -> predicted spectrum
    # =====================================================================
    y_top = 84.0
    _box(ax, 16, y_top, 27, 15, COL["param"], r"Parameters  $\theta$",
         [r"$\{T_{\mathrm{kin}},\ \langle\beta_{T}\rangle,\ n\}$",
          r"(general: $T,\mu,q,\kappa$)"])
    _box(ax, 50, y_top, 30, 15, COL["model"], "Forward model",
         [r"BGBW spectrum", r"$\frac{dN}{p_{T}\,dp_{T}}(\theta)$"],
         body_size=8.0)
    _box(ax, 84, y_top, 26, 15, COL["model"], "Predicted",
         [r"spectrum  $m(\theta)$"], fill=_tint(COL["model"], 0.08))

    _arrow(ax, (29.5, y_top), (35.0, y_top), color=COL["param"])
    _arrow(ax, (65.0, y_top), (71.0, y_top), color=COL["model"])

    # =====================================================================
    #  Middle row: data + likelihood
    # =====================================================================
    y_mid = 56.0
    _box(ax, 84, y_mid, 26, 16, COL["data"], "Data",
         [r"ALICE $\pi,K,p$", r"$p_{T}$ spectra",
          r"(Pb+Pb 5.02 TeV)"], body_size=7.8)
    _box(ax, 47, y_mid, 34, 16, COL["like"], r"Likelihood  $\mathcal{L}(\theta)$",
         [r"$\propto\exp[-\frac{1}{2}\chi^{2}]$,  Gaussian in log-yield",
          r"$\Sigma=\varepsilon_i^2\delta_{ij}"
          r"+\eta_{\mathrm{md}}^2\,e^{-(p_{T,i}-p_{T,j})^2/2\ell^2}$"],
         body_size=8.0)

    # predicted spectrum + data  ->  likelihood  (compare model to data)
    _arrow(ax, (80.0, y_top - 7.6), (58.0, y_mid + 8.3), color=COL["model"],
           style="arc3,rad=0.05")
    ax.text(70.5, 70.0, "compare", ha="center", va="center",
            fontsize=7.6, color="#555555")
    _arrow(ax, (71.0, y_mid), (64.2, y_mid), color=COL["data"])

    # callout: model-discrepancy term (distinctive feature #1)
    #   -> points to the covariance Sigma inside the LIKELIHOOD box.
    _callout(ax, 27.0, 70.0,
             "model-discrepancy term\n(GP covariance sizes the misfit)",
             COL["flag"], w=35.0, h=10.5)
    ax.annotate(
        "", xy=(45.0, 60.3), xytext=(33.0, 64.6),
        arrowprops=dict(arrowstyle="-|>", color=COL["flag"], lw=1.9,
                        mutation_scale=16, shrinkA=1, shrinkB=1,
                        connectionstyle="arc3,rad=0.12"), zorder=6)

    # =====================================================================
    #  Prior + posterior
    # =====================================================================
    y_low = 30.0
    _box(ax, 16, y_mid, 26, 16, COL["prior"], r"Prior  $\pi(\theta)$",
         [r"uniform on physical", r"ranges; hard bounds"], body_size=8.0)
    _box(ax, 44, y_low, 32, 14, COL["post"], "Posterior",
         [r"$p(\theta\mid x)\propto\mathcal{L}(\theta)\,\pi(\theta)$",
          r"(Bayes' theorem)"], body_size=8.2)

    # callout: hard physical boundaries (distinctive feature #2)
    #   -> lowered to open a clear corridor, then a bold arrow up into the
    #      PRIOR box makes the source->target unambiguous.
    _callout(ax, 15, 38.0,
             "hard boundaries:\n" r"$\beta_s<1$ (subluminal)" "\n"
             r"$\eta/s\geq 1/4\pi$ (KSS)",
             COL["flag"], w=25.0, h=12.0, size=7.6)
    ax.annotate(
        "", xy=(15.0, 48.3), xytext=(15.0, 44.2),
        arrowprops=dict(arrowstyle="-|>", color=COL["flag"], lw=1.9,
                        mutation_scale=16, shrinkA=1, shrinkB=1),
        zorder=6)

    # likelihood -> posterior ; prior -> posterior  (Bayes' theorem)
    _arrow(ax, (44.0, y_mid - 8.2), (46.0, y_low + 7.2), color=COL["like"],
           style="arc3,rad=0.08")
    _arrow(ax, (29.0, 52.0), (35.0, y_low + 6.8), color=COL["prior"],
           style="arc3,rad=-0.12")

    # =====================================================================
    #  MCMC -> output
    # =====================================================================
    _box(ax, 80, y_low, 30, 14, COL["param"], "MCMC sampler",
         [r"affine-invariant", r"ensemble (emcee,",
          r"100 walkers)"], body_size=7.8)
    _box(ax, 47, 9.0, 54, 12, COL["out"], "",
         [r"$\mathbf{Posterior}$: credible intervals  "
          r"$+$  posterior-predictive check"],
         title_size=1.0, body_size=8.8, fill=_tint(COL["out"], 0.12))

    _arrow(ax, (60.0, y_low), (65.0, y_low), color=COL["post"])
    _arrow(ax, (74.0, y_low - 7.2), (55.0, 15.5), color=COL["param"],
           style="arc3,rad=0.20")

    # feedback lane (dashed): the sampler proposes new theta, closing the loop.
    # Routed up the free corridor between likelihood and data, then across the
    # top, so it crosses no box.
    ax.plot([65.0, 65.0, 19.0], [y_low + 7.0, 94.0, 94.0],
            ls=(0, (5, 3)), color=COL["param"], lw=1.5, zorder=1)
    _arrow(ax, (19.0, 94.0), (16.0, y_top + 7.6), color=COL["param"],
           ls=(0, (5, 3)), lw=1.5)
    ax.text(66.5, 50.0, r"propose $\theta'$" "\n" "(MCMC step)", ha="left",
            va="center", fontsize=7.8, style="italic", color=COL["param"])

    fig.suptitle("Bayesian inference workflow: from parameters to a "
                 "constrained posterior", fontsize=11.0, y=0.995)

    paths = savefig_dual(fig, "fig14_bayesian_workflow")
    for p in paths:
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
