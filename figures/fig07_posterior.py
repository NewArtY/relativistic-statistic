# -*- coding: utf-8 -*-
r"""
fig07_posterior.py  ->  figures/fig07_posterior.{pdf,png}   (Fig. 7, fig:posterior)
===================================================================================

WHAT IT SHOWS
-------------
The joint Bayesian posterior of the Boltzmann-Gibbs blast-wave (BGBW) parameters

    theta = ( T_kin , <beta_T> , n )

from a REAL affine-invariant MCMC fit (emcee) of identified-particle p_T spectra
(pi+-, K+-, p+pbar) -- the flagship original computation of Sec. 6.  Rendered as
a corner plot (corner package) with 68% and 95% credible contours, posterior
medians, and -- for a synthetic closure test -- the injected truth marked.

This script LOADS the chain produced by
    code/bayesian/run_emcee_bgbw.py  ->  code/data/bgbw_chain.npz
and never re-samples or fabricates: the provenance flag stored in the chain file
(REAL ALICE data vs SYNTHETIC CLOSURE TEST) is read back and printed into the
figure title, so the caption cannot misrepresent the source (risks R03/R28).

The expected physical signature -- a NEGATIVE <beta_T>-n correlation induced by
the hard subluminal cut beta_s = <beta_T>(n+2)/2 < 1 -- is measured from the
chain and printed.

PROVENANCE:  read from bgbw_chain.npz (REAL DATA or SYNTHETIC CLOSURE TEST).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(CODE_DIR, "data")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402
import corner        # noqa: E402

CHAIN_PATH = os.path.join(DATA_DIR, "bgbw_chain.npz")


def make_figure():
    plt = C.apply_style()

    if not os.path.exists(CHAIN_PATH):
        raise FileNotFoundError(
            f"{CHAIN_PATH} not found -- run code/bayesian/run_emcee_bgbw.py first.")

    d = np.load(CHAIN_PATH, allow_pickle=True)
    # The chain samples (T_kin, <beta_T>, n, log10 eta); eta is a marginalized
    # model-discrepancy hyperparameter, so the corner plot shows only the three
    # PHYSICS parameters (eta is integrated over by simply dropping its column).
    n_phys = 3
    flat = d["flat_chain"][:, :n_phys]
    labels = [str(x) for x in d["param_labels"]][:n_phys]
    names = [str(x) for x in d["param_names"]][:n_phys]
    synthetic = bool(d["SYNTHETIC_CLOSURE_TEST"])
    prov = json.loads(str(d["provenance"]))
    med = d["median"][:n_phys]
    lo, hi = d["ci68_lo"][:n_phys], d["ci68_hi"][:n_phys]
    rhat = d["rhat"]                     # reported over all sampled parameters
    truth = d["truth"] if ("truth" in d.files) else None

    print("=" * 72)
    print("fig07_posterior: BGBW posterior corner plot")
    print("=" * 72)
    tag = "SYNTHETIC CLOSURE TEST" if synthetic else "REAL EXPERIMENTAL DATA"
    print(f"  PROVENANCE : {tag}")
    print(f"  source     : {prov['source']}")
    print(f"  samples    : {flat.shape[0]}   max Rhat = {np.max(rhat):.4f}")
    for i, nm in enumerate(names):
        print(f"    {nm:10s} = {med[i]:.4f}  [68%: {lo[i]:.4f}, {hi[i]:.4f}]")
    rho_bn = np.corrcoef(flat[:, 1], flat[:, 2])[0, 1]
    print(f"  corr(<beta_T>, n) = {rho_bn:+.3f}  (anti-correlation from beta_s<1)")
    print("-" * 72)

    # --- corner plot --------------------------------------------------------
    truths = list(truth) if (truth is not None and synthetic) else None
    fig = corner.corner(
        flat,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.68, 0.95),                 # 68% and 95% credible contours
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        truths=truths,
        truth_color="#EE7733",
        color="#22447A",
        hist_kwargs={"color": "#22447A"},
        plot_datapoints=False,
        fill_contours=True,
        smooth=1.0,
        contour_kwargs={"linewidths": 1.0},
    )

    # single-line top title (kept clear of the T_kin column title below it)
    fig.suptitle(
        "BGBW posterior from a real emcee run  "
        f"({flat.shape[0]} samples, max $\\hat{{R}}$={np.max(rhat):.3f})",
        fontsize=11, x=0.5, y=1.045)

    # provenance + diagnostics block in the empty upper-right corner panel
    axes = np.array(fig.axes).reshape((len(names), len(names)))
    corner_ax = axes[0, -1]
    stamp = "SYNTHETIC / CLOSURE TEST" if synthetic else "REAL ALICE DATA"
    stamp_color = "#AA3322" if synthetic else "#227722"
    corner_ax.text(0.98, 0.98, stamp, transform=corner_ax.transAxes,
                   ha="right", va="top", fontsize=11, style="italic",
                   weight="bold", color=stamp_color)
    src_lines = ("BGBW closure test\ninjected truth in orange"
                 if synthetic else
                 "ALICE Pb-Pb $\\sqrt{s_{NN}}=5.02$ TeV\n"
                 "0-5% centrality\nHEPData ins1759506")
    corner_ax.text(0.98, 0.80, src_lines, transform=corner_ax.transAxes,
                   ha="right", va="top", fontsize=8.5, color="0.25")
    corner_ax.text(0.98, 0.34,
                   f"corr($\\langle\\beta_T\\rangle,n$) = {rho_bn:+.2f}\n"
                   f"$\\beta_s^{{\\rm surf}}$ < 1 (subluminal)",
                   transform=corner_ax.transAxes, ha="right", va="top",
                   fontsize=8.5, color="0.25")

    paths = C.savefig_dual(fig, "fig07_posterior", bbox_inches="tight")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    return paths


if __name__ == "__main__":
    make_figure()
