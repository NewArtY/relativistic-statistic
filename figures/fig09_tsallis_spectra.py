# -*- coding: utf-8 -*-
r"""
fig09_tsallis_spectra.py  ->  figures/fig09_tsallis_spectra.{pdf,png}
=====================================================================

WHAT IT SHOWS
-------------
Transverse-momentum spectra of identified hadrons (pi+-, K+-, p+pbar),
measured at midrapidity, fitted by the thermodynamically consistent
Tsallis distribution.  Two panels:

  (a) the invariant p_T spectrum (1/N_ev) d^2N/(dp_T dy) over ~7 decades
      (log-log), data points with error bars + the Tsallis fit per species;
  (b) the phase-space-divided Tsallis function
          g(p_T) = [ (1/N_ev) d^2N/(dp_T dy) ] / (p_T m_T)
                 ~ [1 + (q-1) m_T/T]^{-q/(q-1)},
      which isolates the common power-law tail; a thin reference line marks
      the slope p_T^{-q/(q-1)} (invariant-yield convention).

TSALLIS FORM  (Cleymans & Worku, J. Phys. G 39 (2012) 025006)
-------------------------------------------------------------
The thermodynamically consistent (Lorentz-invariant) yield at y=0 is

    E d^3N/d^3p  =  (1/2pi p_T) d^2N/(dp_T dy)
                 =  C m_T [1 + (q-1)(m_T - mu)/T]^{-q/(q-1)},
    m_T = sqrt(p_T^2 + m^2),   mu = 0,

so the MEASURED observable (1/N_ev) d^2N/(dp_T dy) is fitted with

    Y_model(p_T) = A p_T m_T [1 + (q-1) m_T/T]^{-q/(q-1)}.

We fit ln Y (spectra span decades) for (A, q, T) per species with
scipy.optimize.curve_fit and report q, T and chi^2/dof.

CONVENTION (see 00_conventions.tex, sec. Power-Law Tail Conventions)
--------------------------------------------------------------------
For the INVARIANT p_T yield the correct Tsallis power-law index is
-q/(q-1) (the measure p_T m_T carries the extra powers of momentum); this
is the index shown in panel (b).  It differs by construction from the
bare-occupation index -1/(q-1) used in Fig. 3/Fig. 4.

DATA PROVENANCE
---------------
We FIRST attempt to download the REAL ALICE identified-particle p_T spectra
in pp collisions at sqrt(s) = 5.02 TeV from HEPData record ins1759506
(Acharya et al. [ALICE], Phys. Rev. C 101 (2020) 044907;
DOI 10.17182/hepdata.104923):
    Table 2 = pions (pp),  Table 4 = kaons (pp),  Table 6 = protons (pp),
observable (1/N_ev) d^2N/(dp_T dy) in (GeV/c)^-1 at |y|<0.5.

If (and only if) the download fails, we do NOT fabricate experimental data:
we fall back to a CLEARLY LABELLED SYNTHETIC spectrum drawn from the same
Tsallis form at a known (q, T) plus realistic log-normal noise, with the
SYNTHETIC flag propagated into the figure banner, legend and printout so no
caption can misrepresent the source (risk R05).

PROVENANCE:  printed and stamped on the figure as REAL (HEPData/ALICE) or
             SYNTHETIC.

Run:  python fig09_tsallis_spectra.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402

# ---------------------------------------------------------------------------
# HEPData record: ALICE Pb-Pb & pp id. spectra at 5.02 TeV (ins1759506).
# We use the pp tables (2/4/6) because a single Tsallis function describes
# minimum-bias pp spectra over the whole p_T range without the radial-flow
# distortion of central Pb-Pb.
# ---------------------------------------------------------------------------
HEPDATA_INS = "ins1759506"
HEPDATA_RECID = 104923
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:120.0) research-script"}

# name, latex, mass [GeV], HEPData table, marker, colour
SPECIES = [
    ("pi", r"$\pi^{+}+\pi^{-}$", C.M_PION,   "Table 2", "o", "#0077BB"),
    ("K",  r"$K^{+}+K^{-}$",     C.M_KAON,   "Table 4", "s", "#EE7733"),
    ("p",  r"$p+\bar{p}$",       C.M_PROTON, "Table 6", "^", "#009988"),
]

# fit window per species [GeV/c] (thermal region -> hard onset; avoid the
# lowest bins where resonance feed-down and the very sparse >~12 GeV tail).
FIT_RANGE = {"pi": (0.2, 12.0), "K": (0.25, 12.0), "p": (0.35, 12.0)}

# Synthetic-fallback injected truth (clearly labelled if ever used).
SYN_TRUTH = {"pi": (1.135, 0.075), "K": (1.125, 0.082), "p": (1.115, 0.090)}
SYN_AMP = {"pi": 25.0, "K": 8.0, "p": 4.0}
SYN_NOISE = 0.08


# ===========================================================================
# 1. Data acquisition (real ALICE -> synthetic fallback)
# ===========================================================================
def _fetch_table_json(table_name):
    rec_url = f"https://www.hepdata.net/record/{HEPDATA_INS}?format=json"
    with urllib.request.urlopen(
            urllib.request.Request(rec_url, headers=HEADERS), timeout=40) as r:
        rec = json.loads(r.read().decode("utf-8"))
    proc = table_name.replace(" ", "")
    tbl = next(t for t in rec["data_tables"]
               if t.get("processed_name") == proc or t.get("name") == table_name)
    data_url = (f"https://www.hepdata.net/record/data/"
                f"{HEPDATA_RECID}/{tbl['id']}/1")
    with urllib.request.urlopen(
            urllib.request.Request(data_url, headers=HEADERS), timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def _parse_table(tbl_json):
    """Extract (pT_centre, Y, sigma) with sigma = stat (+) syst in quadrature."""
    pT, Y, sig = [], [], []
    for row in tbl_json["values"]:
        xlo = float(row["x"][0]["low"])
        xhi = float(row["x"][0]["high"])
        xc = 0.5 * (xlo + xhi)
        ycell = row["y"][0]
        yval = float(ycell["value"])
        stat = syst = 0.0
        for e in ycell.get("errors", []):
            lab = e.get("label", "").lower()
            se = float(e.get("symerror", 0.0))
            if "stat" in lab:
                stat = se
            elif "sys" in lab:
                syst = se
        s = float(np.hypot(stat, syst))
        if yval > 0.0 and s > 0.0:
            pT.append(xc); Y.append(yval); sig.append(s)
    return np.array(pT), np.array(Y), np.array(sig)


def load_data():
    """Return (dataset, provenance). Try REAL ALICE pp; fall back to SYNTHETIC."""
    try:
        dataset = {}
        for name, tex, mass, table, mk, col in SPECIES:
            tj = _fetch_table_json(table)
            pT, Y, sig = _parse_table(tj)
            if pT.size < 5:
                raise RuntimeError(f"too few points for {name} ({pT.size})")
            dataset[name] = dict(mass=mass, pT=pT, Y=Y, sigma=sig,
                                 tex=tex, marker=mk, colour=col)
        prov = dict(
            SYNTHETIC=False,
            label="REAL (ALICE / HEPData)",
            source=("ALICE pp sqrt(s)=5.02 TeV, |y|<0.5, HEPData "
                    f"{HEPDATA_INS} (DOI 10.17182/hepdata.104923); "
                    "Acharya et al. [ALICE], Phys. Rev. C 101 (2020) 044907"),
            observable="(1/N_ev) d^2N/(dp_T dy) [(GeV/c)^-1]",
        )
        print(f"[data] REAL ALICE pp 5.02 TeV spectra downloaded (HEPData "
              f"{HEPDATA_INS}).")
        return dataset, prov
    except Exception as exc:  # noqa: BLE001
        print(f"[data] HEPData fetch FAILED ({type(exc).__name__}: {exc}).")
        print("[data] Falling back to a CLEARLY LABELLED SYNTHETIC spectrum "
              "(no fabrication of experimental data).")

    rng = np.random.default_rng(20260715)
    dataset = {}
    for name, tex, mass, table, mk, col in SPECIES:
        q0, T0 = SYN_TRUTH[name]
        pT = np.logspace(np.log10(0.12), np.log10(18.0), 48)
        model = tsallis_yield(pT, SYN_AMP[name], q0, T0, mass)
        noise = rng.lognormal(0.0, SYN_NOISE, size=pT.size)
        Y = model * noise
        sig = SYN_NOISE * Y
        dataset[name] = dict(mass=mass, pT=pT, Y=Y, sigma=sig, tex=tex,
                             marker=mk, colour=col, syn_truth=(q0, T0))
    prov = dict(
        SYNTHETIC=True,
        label="SYNTHETIC (Tsallis + noise; NOT measured data)",
        source=("SYNTHETIC -- thermodynamically consistent Tsallis form at "
                "injected (q,T) per species + "
                f"{int(SYN_NOISE*100)}% log-normal noise. NOT experimental."),
        observable="synthetic d^2N/(dp_T dy) [arb. units]",
    )
    return dataset, prov


# ===========================================================================
# 2. Thermodynamically consistent Tsallis model & fit
# ===========================================================================
def tsallis_yield(pT, A, q, T, m):
    """Measured observable model: A p_T m_T [1+(q-1) m_T/T]^{-q/(q-1)}."""
    mT = np.sqrt(pT ** 2 + m ** 2)
    base = 1.0 + (q - 1.0) * mT / T
    return A * pT * mT * np.power(base, -q / (q - 1.0))


def _fit_species(sp, name):
    """Fit (A, q, T) in log space. Returns dict of params + chi2/dof."""
    pmin, pmax = FIT_RANGE[name]
    m = sp["mass"]
    mask = (sp["pT"] >= pmin) & (sp["pT"] <= pmax)
    pT, Y, sig = sp["pT"][mask], sp["Y"][mask], sp["sigma"][mask]
    ln_sig = sig / Y  # fractional error = sigma of ln Y

    def log_model(pT_, lnA, q, T):
        return np.log(tsallis_yield(pT_, np.exp(lnA), q, T, m))

    # robust initial guess: amplitude from the lowest-pT point.
    A0 = Y[0] / (pT[0] * np.sqrt(pT[0] ** 2 + m ** 2))
    p0 = [np.log(A0), 1.12, 0.08]
    bounds = ([np.log(A0) - 15, 1.001, 0.02], [np.log(A0) + 15, 1.30, 0.30])
    popt, pcov = curve_fit(log_model, pT, np.log(Y), p0=p0, sigma=ln_sig,
                           absolute_sigma=True, bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.diag(pcov))
    lnA, q, T = popt
    resid = (np.log(Y) - log_model(pT, *popt)) / ln_sig
    dof = pT.size - 3
    chi2 = float(np.sum(resid ** 2))
    n_index = q / (q - 1.0)
    return dict(A=np.exp(lnA), q=q, T=T, q_err=perr[1], T_err=perr[2],
                chi2=chi2, dof=dof, chi2_dof=chi2 / dof, n_pts=pT.size,
                n_index=n_index, pmin=pmin, pmax=pmax)


# ===========================================================================
# 3. Figure
# ===========================================================================
def make_figure():
    plt = C.apply_style()
    dataset, prov = load_data()
    syn = prov["SYNTHETIC"]

    print("=" * 74)
    print("fig09_tsallis_spectra: identified-hadron p_T spectra + Tsallis fit")
    print("=" * 74)
    print(f"  PROVENANCE : {prov['label']}")
    print(f"  source     : {prov['source']}")
    print(f"  observable : {prov['observable']}")
    print("-" * 74)

    fits = {}
    for name, tex, mass, table, mk, col in SPECIES:
        fits[name] = _fit_species(dataset[name], name)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    pgrid = np.logspace(np.log10(0.1), np.log10(20.0), 400)

    # ---- panel (a): invariant spectrum + Tsallis fit --------------------
    for name, tex, mass, table, mk, col in SPECIES:
        sp = dataset[name]
        f = fits[name]
        axA.errorbar(sp["pT"], sp["Y"], yerr=sp["sigma"], ls="none",
                     marker=mk, ms=3.4, mfc="white", mec=col, ecolor=col,
                     elinewidth=0.7, capsize=0, label=tex, zorder=3)
        yfit = tsallis_yield(pgrid, f["A"], f["q"], f["T"], mass)
        axA.plot(pgrid, yfit, "-", color=col, lw=1.4, zorder=2, alpha=0.9)

    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(0.1, 20.0)
    axA.set_xlabel(r"$p_{T}$ [GeV/$c$]")
    ylab = (r"$(1/N_{\rm ev})\,\mathrm{d}^2N/(\mathrm{d}p_T\,\mathrm{d}y)$"
            r" [(GeV/$c$)$^{-1}$]")
    axA.set_ylabel(ylab, fontsize=8.5)
    axA.set_title("(a) invariant spectra + Tsallis fit", fontsize=9.5)
    axA.legend(loc="lower left", fontsize=8, handletextpad=0.3)
    axA.grid(True, which="major", alpha=0.22)

    # ---- panel (b): phase-space-divided Tsallis function + tail ---------
    for name, tex, mass, table, mk, col in SPECIES:
        sp = dataset[name]
        f = fits[name]
        mT = np.sqrt(sp["pT"] ** 2 + mass ** 2)
        g = sp["Y"] / (sp["pT"] * mT)            # ~ [1+(q-1)mT/T]^{-q/(q-1)}
        g_err = sp["sigma"] / (sp["pT"] * mT)
        axB.errorbar(sp["pT"], g, yerr=g_err, ls="none", marker=mk, ms=3.4,
                     mfc="white", mec=col, ecolor=col, elinewidth=0.7,
                     capsize=0, zorder=3,
                     label=rf"{tex}: $q={f['q']:.3f}$, $T={f['T']*1e3:.0f}$ MeV")
        mTg = np.sqrt(pgrid ** 2 + mass ** 2)
        gfit = f["A"] * np.power(1.0 + (f["q"] - 1.0) * mTg / f["T"],
                                 -f["q"] / (f["q"] - 1.0))
        axB.plot(pgrid, gfit, "-", color=col, lw=1.3, zorder=2, alpha=0.9)

    # reference power-law tail p_T^{-q/(q-1)} anchored to the pion fit
    fpi = fits["pi"]
    n_ref = fpi["n_index"]
    x0 = 3.0
    mT0 = np.sqrt(x0 ** 2 + C.M_PION ** 2)
    g0 = fpi["A"] * np.power(1.0 + (fpi["q"] - 1.0) * mT0 / fpi["T"],
                             -fpi["q"] / (fpi["q"] - 1.0))
    xr = np.logspace(np.log10(2.0), np.log10(18.0), 50)
    ref = g0 * (xr / x0) ** (-n_ref)
    axB.plot(xr, ref * 4.0, ls=(0, (4, 2)), color="#111111", lw=1.2,
             zorder=4, label=rf"$\propto p_T^{{-q/(q-1)}}={-n_ref:.2f}$")

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(0.1, 20.0)
    axB.set_xlabel(r"$p_{T}$ [GeV/$c$]")
    axB.set_ylabel(r"$g(p_T)=Y/(p_T m_T)\ \sim\ "
                   r"[1+(q{-}1)m_T/T]^{-q/(q-1)}$", fontsize=8.2)
    axB.set_title("(b) Tsallis function + power-law tail", fontsize=9.5)
    axB.legend(loc="lower left", fontsize=7.4, handletextpad=0.3)
    axB.grid(True, which="major", alpha=0.22)

    # ---- provenance banner --------------------------------------------------
    banner = ("SYNTHETIC DATA (Tsallis + noise) -- NOT measured"
              if syn else
              "REAL DATA: ALICE pp $\\sqrt{s}=5.02$ TeV (HEPData ins1759506)")
    bcol = "#AA3322" if syn else "#227722"
    fig.suptitle(banner, fontsize=10, color=bcol, style="italic",
                 weight="bold")

    paths = C.savefig_dual(fig, "fig09_tsallis_spectra", bbox_inches="tight")

    # ---- printout -----------------------------------------------------------
    print("  FITTED TSALLIS PARAMETERS (thermodynamically consistent, mu=0)")
    print(f"    {'species':8s} {'q':>8s} {'T[MeV]':>9s} "
          f"{'-q/(q-1)':>9s} {'chi2/dof':>9s} {'Npts':>5s}   fit range")
    for name, tex, mass, table, mk, col in SPECIES:
        f = fits[name]
        print(f"    {name:8s} {f['q']:8.4f} {f['T']*1e3:9.2f} "
              f"{-f['n_index']:9.2f} {f['chi2_dof']:9.2f} {f['n_pts']:5d}   "
              f"[{f['pmin']:.2f},{f['pmax']:.1f}] GeV/c")
    print("-" * 74)
    print(f"  PROVENANCE (for caption): {prov['label']}")
    if syn:
        print("    -> caption MUST state the points are SYNTHETIC, not ALICE.")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    print("=" * 74)
    plt.close(fig)
    return fits, prov, paths


if __name__ == "__main__":
    make_figure()
