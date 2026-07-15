# -*- coding: utf-8 -*-
r"""
fig10_kaniadakis_neutrino.py  ->  figures/fig10_kaniadakis_neutrino.{pdf,png}
=============================================================================

WHAT IT SHOWS
-------------
The high-energy astrophysical neutrino flux (per flavour, E^2 Phi vs E over
25 TeV - 2.8 PeV) compared with three functional forms:

  (i)   an unbroken power law   Phi ~ E^{-Gamma},  Gamma = 2.5      (dashed),
  (ii)  a Kaniadakis kappa-distribution  Phi ~ exp_kappa(-E/E0),
        whose high-energy bare tail is E^{-1/kappa}                (solid),
  (iii) a Maxwell-Juttner exponential  Phi ~ exp(-E/T_nu)          (dotted),
        which cannot describe the measured power-law tail and collapses
        above ~100 TeV.

The Kaniadakis kappa is fitted so that its asymptotic index -1/kappa
reproduces the neutrino spectral index; the fitted kappa is printed and
annotated.

CONVENTION (see 00_conventions.tex)
-----------------------------------
kappa is the DIMENSIONLESS Kaniadakis deformation parameter (|kappa|<1); the
argument E/E0 is dimensionless.  The bare-occupation tail index is -1/kappa
(distinct from the invariant-yield index used in Fig. 9).

DATA PROVENANCE  --  *** SYNTHETIC (clearly labelled) ***
---------------------------------------------------------
The individual flux points plotted here are SYNTHETIC: they are drawn from
the published single-power-law description of the IceCube astrophysical
neutrino flux (per-flavour spectral index Gamma ~= 2.5, normalization at
100 TeV of order 10^{-8} GeV cm^-2 s^-1 sr^-1), namely

    Aartsen et al. [IceCube], Astrophys. J. 809 (2015) 98
    ("A combined maximum-likelihood analysis of the high-energy
     astrophysical neutrino flux ..."),

with realistic log-normal scatter and ~35% error bars added, plus two
upper limits at the highest energies (down arrows), consistent with the
low PeV-scale event statistics.  ONLY the power-law SHAPE/INDEX and the
approximate normalization are taken from the citation; the per-point values
are NOT the published measurement and MUST NOT be presented as measured
data.  The SYNTHETIC flag is stamped on the figure and printed.

(We do not fabricate the exact published differential points, whose precise
binning and asymmetric errors we cannot reproduce verbatim; honest synthetic
data grounded in the published index is used instead -- risk R11.)

Run:  python fig10_kaniadakis_neutrino.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402
from _common import exp_k  # noqa: E402

# ---------------------------------------------------------------------------
# Published power-law description (Aartsen et al. 2015, ApJ 809:98).
# Per-flavour E^2 Phi at the pivot E_ref = 100 TeV, and the spectral index.
# ---------------------------------------------------------------------------
E_REF = 1.0e5                    # 100 TeV in GeV
GAMMA = 2.5                      # per-flavour astrophysical spectral index
E2PHI_REF = 3.0e-8              # GeV cm^-2 s^-1 sr^-1  (per flavour, ~ published)
E_MIN, E_MAX = 2.5e4, 2.8e6     # 25 TeV .. 2.8 PeV  in GeV

T_NU = 1.5e4                     # 15 TeV: Maxwell-Juttner / Kaniadakis scale E0
SYN_SEED = 20260715
SYN_SCATTER = 0.28               # log-normal scatter of the synthetic points
SYN_ERR = 0.35                   # ~35% relative error bars


def powerlaw_E2phi(E):
    """Published unbroken power law in E^2 Phi representation."""
    return E2PHI_REF * (E / E_REF) ** (2.0 - GAMMA)


def make_synthetic_data():
    """Draw SYNTHETIC flux points from the published power law (labelled)."""
    rng = np.random.default_rng(SYN_SEED)
    # 8 measured points + 2 upper limits at the top, log-spaced.
    E = np.logspace(np.log10(E_MIN), np.log10(E_MAX), 10)
    E_meas, E_ulim = E[:8], E[8:]
    true = powerlaw_E2phi(E_meas)
    y = true * rng.lognormal(0.0, SYN_SCATTER, size=E_meas.size)
    yerr = SYN_ERR * y
    # upper limits: place arrows at ~1.6x the power-law expectation
    ulim = 1.6 * powerlaw_E2phi(E_ulim)
    return dict(E=E_meas, y=y, yerr=yerr, E_ulim=E_ulim, ulim=ulim)


# ===========================================================================
# Kaniadakis and Maxwell-Juttner overlays
# ===========================================================================
def kaniadakis_E2phi(E, A, kappa, E0):
    """E^2 Phi for a Kaniadakis flux Phi ~ exp_kappa(-E/E0)."""
    return A * E ** 2 * exp_k(-E / E0, kappa)


def fit_kaniadakis(data):
    """Fit (A, kappa, E0) of the Kaniadakis form to the synthetic points."""
    E, y, yerr = data["E"], data["y"], data["yerr"]
    ln_sig = yerr / y

    def log_model(E_, lnA, kappa, lnE0):
        return np.log(kaniadakis_E2phi(E_, np.exp(lnA), kappa, np.exp(lnE0)))

    # asymptotic E^{2-1/kappa} = E^{-0.5}  ->  kappa ~ 0.4 initial guess
    p0 = [np.log(1e-18), 0.40, np.log(T_NU)]
    bounds = ([np.log(1e-30), 0.20, np.log(2e3)],
              [np.log(1e-6), 0.90, np.log(2e5)])
    popt, pcov = curve_fit(log_model, E, np.log(y), p0=p0, sigma=ln_sig,
                           absolute_sigma=True, bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    A, kappa, E0 = np.exp(popt[0]), popt[1], np.exp(popt[2])
    resid = (np.log(y) - log_model(E, *popt)) / ln_sig
    chi2 = float(np.sum(resid ** 2))
    dof = E.size - 3
    return dict(A=A, kappa=kappa, E0=E0, kappa_err=perr[1],
                chi2=chi2, dof=dof, chi2_dof=chi2 / dof)


def maxwell_juttner_E2phi(E, data):
    """Exponential (MJ) form normalized to the lowest-energy synthetic point."""
    E0_low = data["E"][0]
    norm = data["y"][0] / (E0_low ** 2 * np.exp(-E0_low / T_NU))
    return norm * E ** 2 * np.exp(-E / T_NU)


# ===========================================================================
# Figure
# ===========================================================================
def make_figure():
    plt = C.apply_style()
    data = make_synthetic_data()
    kfit = fit_kaniadakis(data)

    print("=" * 74)
    print("fig10_kaniadakis_neutrino: IceCube nu flux + Kaniadakis kappa-fit")
    print("=" * 74)
    print("  PROVENANCE : SYNTHETIC (labelled) -- NOT the published points.")
    print("  shape from : Aartsen et al. [IceCube], ApJ 809 (2015) 98 "
          f"(Gamma~{GAMMA}, per-flavour)")
    print(f"  energy range: {E_MIN/1e3:.0f} TeV .. {E_MAX/1e6:.2f} PeV")
    print("-" * 74)
    print("  OVERLAYS")
    print(f"    (i)   power law    : Gamma = {GAMMA}  (E^2 Phi ~ "
          f"E^{2.0-GAMMA:+.1f})")
    print(f"    (ii)  Kaniadakis   : kappa = {kfit['kappa']:.3f} "
          f"+/- {kfit['kappa_err']:.3f},  E0 = {kfit['E0']/1e3:.1f} TeV,  "
          f"asymptotic index -1/kappa = {-1.0/kfit['kappa']:.2f}")
    print(f"           chi2/dof (Kaniadakis vs synthetic) = "
          f"{kfit['chi2_dof']:.2f}")
    print(f"    (iii) Maxwell-Juttner exp : T_nu = {T_NU/1e3:.0f} TeV "
          "(collapses at high E)")
    print("-" * 74)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    Eg = np.logspace(np.log10(E_MIN * 0.9), np.log10(E_MAX * 1.05), 400)

    # (iii) Maxwell-Juttner exponential (dotted, fails)
    ax.plot(Eg, maxwell_juttner_E2phi(Eg, data), ls=":", color="#CC3311",
            lw=2.0, zorder=2,
            label=("Maxwell–Jüttner "
                   rf"$\propto e^{{-E/T_\nu}}$ ($T_\nu={T_NU/1e3:.0f}$ TeV)"))

    # (i) unbroken power law (dashed)
    ax.plot(Eg, powerlaw_E2phi(Eg), ls="--", color="#555555", lw=1.8,
            zorder=3, label=rf"power law $E^{{-\Gamma}}$, $\Gamma={GAMMA}$")

    # (ii) Kaniadakis (solid)
    ax.plot(Eg, kaniadakis_E2phi(Eg, kfit["A"], kfit["kappa"], kfit["E0"]),
            ls="-", color="#0077BB", lw=2.2, zorder=4,
            label=(rf"Kaniadakis $\exp_\kappa(-E/E_0)$, "
                   rf"$\kappa={kfit['kappa']:.2f}$ ($-1/\kappa="
                   rf"{-1.0/kfit['kappa']:.2f}$)"))

    # SYNTHETIC data points
    ax.errorbar(data["E"], data["y"], yerr=data["yerr"], ls="none",
                marker="o", ms=5, mfc="white", mec="#222222",
                ecolor="#222222", elinewidth=1.0, capsize=2.5, zorder=6,
                label="SYNTHETIC flux points")

    # upper limits as downward arrows
    ax.errorbar(data["E_ulim"], data["ulim"],
                yerr=0.35 * data["ulim"], uplims=True, ls="none",
                marker="_", ms=8, color="#222222", ecolor="#222222",
                elinewidth=1.0, capsize=3, zorder=6,
                label="upper limits (90% C.L.)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(E_MIN * 0.8, E_MAX * 1.3)
    ax.set_ylim(1e-11, 2e-7)
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$E_\nu^{2}\,\Phi$  [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title(r"Astrophysical $\nu$ flux vs Kaniadakis $\kappa$-distribution")
    ax.grid(True, which="major", alpha=0.22)

    # secondary axis in TeV for readability
    secax = ax.secondary_xaxis(
        "top", functions=(lambda e: e / 1e3, lambda t: t * 1e3))
    secax.set_xlabel(r"$E_\nu$ [TeV]", fontsize=9)

    ax.legend(loc="lower left", fontsize=7.6, handletextpad=0.4,
              labelspacing=0.35)

    # SYNTHETIC banner
    ax.text(0.97, 0.95, "SYNTHETIC DATA\n(NOT measured IceCube points)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            style="italic", weight="bold", color="#AA3322",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AA3322",
                      lw=0.8, alpha=0.9))

    paths = C.savefig_dual(fig, "fig10_kaniadakis_neutrino", bbox_inches="tight")
    print("  PROVENANCE (for caption): SYNTHETIC -- points are not the "
          "published IceCube measurement.")
    print(f"  kappa used (fitted) = {kfit['kappa']:.3f}")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    print("=" * 74)
    plt.close(fig)
    return kfit, paths


if __name__ == "__main__":
    make_figure()
