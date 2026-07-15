# -*- coding: utf-8 -*-
r"""
fig07b_ppc.py  ->  figures/fig07b_ppc.{pdf,png}   (Fig. 7b, fig:ppc)
====================================================================

POSTERIOR-PREDICTIVE CHECK of the BGBW blast-wave fit of Sec. 6.3.

WHAT IT SHOWS
-------------
The fitted pion, kaon and proton transverse-momentum spectra overlaid on the
ALICE data, with a lower panel of normalized residuals (pulls).  This is the
honest companion to the corner plot (Fig. fig:posterior): it shows the ACTUAL
quality of the fit, including the residual point-to-point SHAPE misfit of the
idealized blast-wave that motivates the p_T-correlated model-discrepancy term.

Upper panel:
  * points  = measured ALICE spectra (scaled per species for legibility);
  * solid   = ideal blast-wave BEST FIT (diagonal likelihood; the maximum the
              three-parameter model can achieve).  The posterior median is
              statistically indistinguishable from it on this scale;
  * dark band  = 68% credible band of the mean model from the GP-augmented
                 chain (parametric uncertainty);
  * light band = +-eta_md model-discrepancy envelope of Eq. eq:gp-cov (the
                 coherent shape freedom the data allow the model).
Lower panel:
  * pulls (ln Y_data - ln Y_bestfit)/eps in units of the QUOTED point-to-point
    error.  They scatter well beyond +-1: the ideal blast-wave does not describe
    the spectra within the measurement errors (raw chi^2/dof ~ 4.6).
  * the per-point light band is the +-sqrt(1+(eta_md/eps)^2) total-uncertainty
    envelope (measurement (+) discrepancy): the pulls lie within it, which is why
    the augmented chi^2/dof ~ 1.

Two goodness-of-fit numbers are annotated: the ideal-model best fit (~4.6, an
honest statement of the misfit) and the augmented fit (~1, with the marginalized
GP discrepancy).

This script LOADS the chain produced by
    code/bayesian/run_emcee_bgbw.py  ->  code/data/bgbw_chain.npz
and never re-samples or fabricates.  The provenance flag stored in the chain
file (REAL ALICE data vs SYNTHETIC CLOSURE TEST) is read back and printed/stamped
so the caption cannot misrepresent the source.

Run:  python fig07b_ppc.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(CODE_DIR, "data")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402
from bayesian.bgbw_model import spectrum, beta_s_from_mean  # noqa: E402

CHAIN_PATH = os.path.join(DATA_DIR, "bgbw_chain.npz")

# name, latex, marker, colour, upper-panel scale factor (for visual separation)
SPECIES = [
    ("pi", r"$\pi^{+}+\pi^{-}$", "o", "#0077BB", 1.0e2),
    ("K",  r"$K^{+}+K^{-}$",     "s", "#EE7733", 1.0e1),
    ("p",  r"$p+\bar{p}$",       "^", "#009988", 1.0e0),
]


def _species_pack(d, name):
    """Rebuild the cached GP pieces for one species from the stored chain file."""
    pT = d[f"data_{name}_pT"]
    Y = d[f"data_{name}_Y"]
    sigma = d[f"data_{name}_sigma"]
    mass = float(d[f"data_{name}_mass"])
    ell = float(d["gp_ell_frac"]) * (pT.max() - pT.min())
    dx = pT[:, None] - pT[None, :]
    rbf = np.exp(-0.5 * (dx / ell) ** 2)
    return dict(pT=pT, Y=Y, sigma=sigma, mass=mass, ell=ell, rbf=rbf,
                eps=sigma / Y, ln_data=np.log(Y))


def _lnA_diag(sp, ln_model):
    """Inverse-variance (diagonal) profiled log-amplitude -- matches the raw fit."""
    w = 1.0 / sp["eps"] ** 2
    d = sp["ln_data"] - ln_model
    return np.sum(w * d) / np.sum(w)


def _lnA_gls(sp, ln_model, eta2):
    """GLS-profiled log-amplitude under the full GP covariance (matches the run)."""
    d = sp["ln_data"] - ln_model
    one = np.ones_like(d)
    Cf = np.diag(sp["eps"] ** 2) + eta2 * sp["rbf"]
    cf = cho_factor(Cf, lower=True, check_finite=False)
    a = float(one @ cho_solve(cf, one, check_finite=False))
    b = float(one @ cho_solve(cf, d, check_finite=False))
    return b / a


def make_figure():
    plt = C.apply_style()

    if not os.path.exists(CHAIN_PATH):
        raise FileNotFoundError(
            f"{CHAIN_PATH} not found -- run code/bayesian/run_emcee_bgbw.py first.")

    d = np.load(CHAIN_PATH, allow_pickle=True)
    flat = d["flat_chain"]                       # (N, 4): T, <beta_T>, n, log10 eta
    synthetic = bool(d["SYNTHETIC_CLOSURE_TEST"])
    prov = json.loads(str(d["provenance"]))
    med = d["median"]
    bestfit = d["bestfit"]                        # ideal blast-wave diagonal MLE
    chi2dof_raw = float(d["chi2dof_raw"])
    chi2dof_aug = float(d["chi2dof_aug"])
    eta_med = float(d["eta_median"])

    Tb, mbb, nb = bestfit[0], bestfit[1], bestfit[2]
    bsb = beta_s_from_mean(mbb, nb)

    print("=" * 74)
    print("fig07b_ppc: BGBW posterior-predictive check")
    print("=" * 74)
    tag = "SYNTHETIC CLOSURE TEST" if synthetic else "REAL EXPERIMENTAL DATA"
    print(f"  PROVENANCE : {tag}")
    print(f"  source     : {prov['source']}")
    print(f"  ideal best fit : T_kin={Tb:.4f} GeV  <beta_T>={mbb:.4f}  n={nb:.4f}")
    print(f"  posterior median: T_kin={med[0]:.4f}  <beta_T>={med[1]:.4f}  "
          f"n={med[2]:.4f}  eta={eta_med:.4f}")
    print(f"  ideal blast-wave chi^2/dof = {chi2dof_raw:.3f}  (measurement errors)")
    print(f"  augmented        chi^2/dof = {chi2dof_aug:.3f}  (with GP discrepancy)")
    print("-" * 74)

    rng = np.random.default_rng(20260714)
    n_draw = min(400, flat.shape[0])
    idx = rng.choice(flat.shape[0], n_draw, replace=False)

    fig, (axS, axR) = plt.subplots(
        2, 1, figsize=(7.0, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25], "hspace": 0.07})

    all_pulls = []
    chi2_check = 0.0
    npts = 0
    for name, tex, mk, col, scale in SPECIES:
        sp = _species_pack(d, name)
        pT = sp["pT"]
        pgrid = np.linspace(pT.min(), pT.max(), 220)

        # ideal blast-wave best-fit curve (diagonal-profiled amplitude)
        ln_model_bf = np.log(pT * spectrum(pT, sp["mass"], Tb, bsb, nb, n_r=81))
        lnA_bf = _lnA_diag(sp, ln_model_bf)
        y_bf = np.exp(lnA_bf) * pgrid * spectrum(pgrid, sp["mass"], Tb, bsb, nb,
                                                 n_r=81)

        # 68% parametric credible band of the mean model from the chain
        band = np.empty((n_draw, pgrid.size))
        for j, k in enumerate(idx):
            Tj, mbj, nj, lej = flat[k]
            bsj = beta_s_from_mean(mbj, nj)
            etaj2 = (10.0 ** lej) ** 2
            ln_model_j = np.log(pT * spectrum(pT, sp["mass"], Tj, bsj, nj, n_r=41))
            lnAj = _lnA_gls(sp, ln_model_j, etaj2)
            band[j] = np.exp(lnAj) * pgrid * spectrum(pgrid, sp["mass"], Tj, bsj,
                                                      nj, n_r=41)
        b_lo = np.percentile(band, 16, axis=0)
        b_hi = np.percentile(band, 84, axis=0)

        # ---- upper panel: spectra (scaled) --------------------------------
        axS.fill_between(pgrid, scale * y_bf * np.exp(-eta_med),
                         scale * y_bf * np.exp(eta_med),
                         color=col, alpha=0.13, lw=0, zorder=1)   # +-eta envelope
        axS.fill_between(pgrid, scale * b_lo, scale * b_hi,
                         color=col, alpha=0.40, lw=0, zorder=2)   # 68% credible
        axS.plot(pgrid, scale * y_bf, "-", color=col, lw=1.4, zorder=3)
        axS.errorbar(pT, scale * sp["Y"], yerr=scale * sp["sigma"], ls="none",
                     marker=mk, ms=3.6, mfc="white", mec=col, ecolor=col,
                     elinewidth=0.7, capsize=0, zorder=4,
                     label=rf"{tex} ($\times{scale:.0f}$)")

        # ---- lower panel: pulls vs best fit, in units of measurement error -
        pulls = (sp["ln_data"] - (ln_model_bf + lnA_bf)) / sp["eps"]
        chi2_check += float(np.sum(pulls ** 2))
        npts += pT.size

        # per-point total-uncertainty envelope +- sqrt(1 + (eta/eps)^2)
        env = np.sqrt(1.0 + (eta_med / sp["eps"]) ** 2)
        order = np.argsort(pT)
        axR.fill_between(pT[order], -env[order], env[order], color=col,
                         alpha=0.12, lw=0, zorder=1)
        axR.plot(pT, pulls, ls="none", marker=mk, ms=3.6, mfc="white",
                 mec=col, mew=0.9, zorder=3)
        all_pulls.append(pulls)

    axR.axhline(0.0, color="0.35", lw=0.8, zorder=2)
    for lev in (-1.0, 1.0):
        axR.axhline(lev, color="0.6", lw=0.7, zorder=2)
    for lev in (-2.0, 2.0):
        axR.axhline(lev, color="0.7", lw=0.7, ls=(0, (4, 2)), zorder=2)

    axS.set_yscale("log")
    axS.set_ylabel(r"$(1/N_{\rm ev})\,\mathrm{d}^2N/(\mathrm{d}p_T\,"
                   r"\mathrm{d}y)\times$ scale  [(GeV/$c$)$^{-1}$]", fontsize=8.6)
    axS.legend(loc="upper right", fontsize=8.2, handletextpad=0.3,
               labelspacing=0.25)

    gof = (rf"ideal blast-wave (best fit): $\chi^2/{{\rm dof}}={chi2dof_raw:.1f}$"
           "\n"
           rf"with GP discrepancy: $\chi^2/{{\rm dof}}={chi2dof_aug:.2f}$"
           "\n"
           rf"$\eta_{{\rm md}}={eta_med:.3f}$ (coherent shape freedom)")
    axS.text(0.02, 0.03, gof, transform=axS.transAxes, ha="left", va="bottom",
             fontsize=8.4, color="0.12",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                       alpha=0.92))

    axR.set_ylabel(r"pull $=(\ln Y_{\rm dat}-\ln Y_{\rm fit})/\varepsilon$",
                   fontsize=8.6)
    axR.set_xlabel(r"$p_{T}$ [GeV/$c$]")
    ally = np.concatenate(all_pulls)
    ylim = max(3.3, 1.12 * np.max(np.abs(ally)))
    axR.set_ylim(-ylim, ylim)
    axR.set_xlim(0.15, 3.05)

    banner = ("SYNTHETIC CLOSURE TEST - NOT measured data"
              if synthetic else
              "REAL DATA: ALICE Pb-Pb $\\sqrt{s_{NN}}=5.02$ TeV, 0-5%"
              " (HEPData ins1759506)")
    bcol = "#AA3322" if synthetic else "#227722"
    fig.suptitle("BGBW posterior-predictive check\n" + banner, fontsize=10,
                 color=bcol)

    paths = C.savefig_dual(fig, "fig07b_ppc", bbox_inches="tight")
    print(f"  pull-panel check: chi^2 = {chi2_check:.1f} over N={npts} points "
          f"-> chi^2/(N-6) = {chi2_check/(npts-6):.3f} (should match ideal "
          f"{chi2dof_raw:.3f})")
    print("  per-species pull RMS:")
    for pulls, (name, *_1) in zip(all_pulls, SPECIES):
        print(f"    {name:3s}: RMS(pull) = {np.sqrt(np.mean(pulls**2)):.2f}  "
              f"(N={pulls.size})")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    print("=" * 74)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    make_figure()
