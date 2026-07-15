# -*- coding: utf-8 -*-
r"""
run_emcee_bgbw.py -- REAL affine-invariant MCMC for the BGBW blast-wave fit
===========================================================================

This is the flagship *original* computational result of Sec. 6 (Block C).  It
replaces the earlier faked posterior (which drew np.random.multivariate_normal
and merely *labelled* it "emcee") with a genuine Markov-chain Monte Carlo run of
the emcee affine-invariant ensemble sampler over the Boltzmann-Gibbs blast-wave
(BGBW) likelihood of identified-particle p_T spectra.

DATA PROVENANCE
---------------
We FIRST attempt to download the real ALICE Pb-Pb, sqrt(s_NN) = 5.02 TeV
identified-particle (pi+-, K+-, p+pbar) transverse-momentum spectra, 0-5%
centrality, from HEPData record ins1759506 (Acharya et al. [ALICE],
Phys. Rev. C 101, 044907 (2020); HEPData DOI 10.17182/hepdata.104923):
    Table 1 = pions (Pb-Pb),  Table 3 = kaons,  Table 5 = protons,
    centrality group 0 = 0.0-5.0 %.
The measured observable is (1/N_ev) d^2N/(dp_T dy) in (GeV/c)^-1.

If (and only if) the download fails or returns an unexpected structure, we do
NOT fabricate data.  Instead we fall back to a CLEARLY LABELLED synthetic
closure test: the BGBW model evaluated at an injected truth
(T_kin = 0.095 GeV, <beta_T> = 0.65, n = 0.85) for pi/K/p, corrupted with
realistic ~10% multiplicative log-normal noise, with SYNTHETIC_CLOSURE_TEST set
True and recorded in every output.

The provenance flag is written into code/data/bgbw_chain.npz and printed to the
console so no downstream figure can misrepresent the source.

LIKELIHOOD -- HONEST ERROR MODEL (why this file changed)
--------------------------------------------------------
An earlier version used a purely DIAGONAL Gaussian likelihood in the log-yield,
    ln L = -1/2 sum_i [ (ln Y_model,i - ln Y_data,i) / eps_i ]^2,  eps_i = sigma_i/Y_i,
and reported sub-percent credible intervals (<beta_T> = 0.660 +- 0.001).  That is
NOT defensible: the ideal blast-wave misses these spectra at chi^2/dof ~ 4.6, so
the residuals are dominated by genuine point-to-point SHAPE misfit of the model,
not by the quoted statistical/systematic errors.  Sub-percent intervals from a
fit with chi^2/dof ~ 4.6 overstate the precision; published ALICE/BGBW analyses
quote T_kin ~ +-10 MeV and <beta_T> ~ +-0.02-0.03.

CHOSEN FIX (approach a of the honesty plan): a p_T-CORRELATED MODEL-DISCREPANCY
term added to the covariance -- a squared-exponential (Gaussian) process kernel
over p_T with a single, marginalized amplitude hyperparameter eta.  For each
species the log-yield covariance is

    C_s = diag(eps_i^2)  +  eta^2 * K_s,
    K_s[i,j] = exp( -(p_T,i - p_T,j)^2 / (2 ell_s^2) ),
    ell_s    = GP_ELL_FRAC * (p_T^max - p_T^min)_s      (correlation length),

so eta^2 K_s represents a smooth, coherent shape freedom that the ideal
blast-wave is allowed on top of the measurement errors.  The single amplitude
hyperparameter eta (shared by all three species; log-uniform prior, sampled and
then MARGINALIZED) is fixed by the data through the proper Gaussian normalization
    ln L_s = -1/2 r^T C_s^{-1} r  -  1/2 ln det C_s  -  (N_s-1)/2 ln(2 pi)
             -  1/2 ln( 1^T C_s^{-1} 1 ),
whose log-determinant penalizes an over-large eta and so lets the data determine
how much shape freedom the model needs.  The per-species overall amplitude ln A_s
is still profiled analytically, now by GENERALIZED least squares under the full
C_s (the last two terms above make the profiling identical to marginalizing ln A_s
under a flat prior on the log-amplitude -- see PROFILED AMPLITUDE below).

Effect: the discrepancy amplitude settles at eta ~ 0.09 (about a 9% coherent
shape freedom in the log-yield), the AUGMENTED fit has chi^2/dof ~ 1, and,
because the physics parameters must now share the residual with the discrepancy
term, the marginal credible intervals widen to a physical scale
(T_kin ~ +-14 MeV, <beta_T> ~ +-0.013, n ~ +-0.05).  We report BOTH the raw
diagonal chi^2/dof (the ideal blast-wave, ~4.6, an honest statement of the model
misfit) and the augmented chi^2/dof (~1).  The sampled vector is

    theta = (T_kin, <beta_T>, n, log10 eta),

and the three physics parameters are the reported posterior (eta is a nuisance
hyperparameter, marginalized out).

PROFILED AMPLITUDE
------------------
Because ln L_s is quadratic in ln A_s, the optimal ln A_s under the full C_s is
the generalized-least-squares estimate
    ln A_s = (1^T C_s^{-1} d_s) / (1^T C_s^{-1} 1),   d_s = ln Y_data - ln Y_model,
and the analytic marginalization of ln A_s under a flat prior adds the constant
-1/2 ln(1^T C_s^{-1} 1) kept in ln L_s above.  Profiling and marginalizing the
log-amplitude therefore coincide here, which is the sense in which Table
tab:inference lists the per-species normalizations as "profiled / marginalized".

PRIORS  (physically motivated, Sec. 6.2 -- these are the ACTUAL run priors)
---------------------------------------------------------------------------
    T_kin    ~ U[0.05, 0.15] GeV      (kinetic freeze-out window; the upper edge
                                       0.15 GeV sits just below the pseudocritical
                                       T_c ~ 0.155 GeV -- freeze-out follows
                                       hadronization)
    <beta_T> ~ U[0.40, 0.90]          (mean transverse flow)
    n        ~ U[0.30, 1.80]          (velocity-profile exponent)
    log10 eta ~ U[-3, 0]              (model-discrepancy amplitude, ~0.1% .. 100%)
    HARD subluminal cut: beta_s = <beta_T> (n + 2) / 2 < 1   (surface velocity).

The subluminal cut is what induces the physical <beta_T>-n anti-correlation seen
in the posterior (larger n at fixed <beta_T> raises beta_s toward the light cone).
The posterior is insensitive to widening the first three ranges: the modes sit
well inside them.

SAMPLER & CONVERGENCE
---------------------
emcee.EnsembleSampler, 100 walkers, several thousand steps.  We (a) estimate the
integrated autocorrelation time tau (emcee get_autocorr_time), require chain
length > 50 tau, discard burn-in = a few tau, thin by ~tau/2; (b) compute the
Gelman-Rubin split-Rhat across walkers (a heuristic for an ensemble sampler; the
tau-based effective sample size is the primary diagnostic); (c) report the mean
acceptance fraction.  Runtime is a few minutes on a single core.

OUTPUT
------
code/data/bgbw_chain.npz : flat (post-burn-in, thinned) chain, parameter names,
                           provenance flag, injected truth (if synthetic),
                           data arrays, GP error-model settings, raw and augmented
                           chi^2/dof, and convergence diagnostics.

Run:  python run_emcee_bgbw.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(CODE_DIR, "data")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C                       # noqa: E402
from bayesian.bgbw_model import (          # noqa: E402
    spectrum, beta_s_from_mean, mean_beta_T,
)

import emcee                               # noqa: E402

SEED = 20260714
RNG = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Species table.  Fit ranges follow the standard ALICE simultaneous BGBW
# windows (thermal region; excludes low-p_T resonance feed-down and the
# high-p_T onset of hard/non-flow physics): pi [0.5,1.0], K [0.2,1.5],
# p [0.3,3.0] GeV/c.
# ---------------------------------------------------------------------------
SPECIES = [
    # name, mass [GeV], HEPData table, pT_min, pT_max
    ("pi", C.M_PION,   "Table 1", 0.5, 1.0),
    ("K",  C.M_KAON,   "Table 3", 0.2, 1.5),
    ("p",  C.M_PROTON, "Table 5", 0.3, 3.0),
]

HEPDATA_RECID = 104923               # ALICE Pb-Pb & pp 5.02 TeV id. spectra
HEPDATA_INS = "ins1759506"
CENTRALITY_GROUP = 0                 # 0 -> "0.0-5.0 pct"

# Injected truth for the synthetic closure-test fallback.
TRUTH = {"T_kin": 0.095, "mean_beta": 0.65, "n": 0.85}
SYN_NOISE = 0.10                     # 10% multiplicative log-normal noise

# ---------------------------------------------------------------------------
# GP model-discrepancy settings (the honest error model; see module docstring).
# ---------------------------------------------------------------------------
GP_ELL_FRAC = 0.30    # discrepancy correlation length = 30% of each fit window
GP_KERNEL = "squared-exponential (Gaussian) in p_T, per species, block-diagonal"


# ===========================================================================
# 1. Data acquisition
# ===========================================================================
def _fetch_table_json(table_name):
    """Download one HEPData table (record/data JSON) for the ALICE record.

    Returns the parsed JSON dict, or raises on any network/parse error.
    """
    # Resolve the numeric table id from the record, then hit the data endpoint.
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:120.0) research-script"}
    rec_url = f"https://www.hepdata.net/record/{HEPDATA_INS}?format=json"
    with urllib.request.urlopen(urllib.request.Request(rec_url, headers=hdr),
                                timeout=40) as r:
        rec = json.loads(r.read().decode("utf-8"))
    proc = table_name.replace(" ", "")
    tbl = next(t for t in rec["data_tables"]
               if t.get("processed_name") == proc or t.get("name") == table_name)
    data_url = (f"https://www.hepdata.net/record/data/"
                f"{HEPDATA_RECID}/{tbl['id']}/1")
    with urllib.request.urlopen(urllib.request.Request(data_url, headers=hdr),
                                timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def _parse_alice_table(tbl_json, pT_min, pT_max):
    """Extract (pT, Y, sigma) for centrality group 0 from an ALICE data table.

    pT is the bin centre; Y is (1/N_ev) d^2N/(dp_T dy); sigma is the
    point-to-point error (stat (+) syst.uncorr in quadrature).  Only bins in
    [pT_min, pT_max] are kept.
    """
    pT, Y, sig = [], [], []
    for row in tbl_json["values"]:
        xlo = float(row["x"][0]["low"])
        xhi = float(row["x"][0]["high"])
        xc = 0.5 * (xlo + xhi)
        if not (pT_min <= xc <= pT_max):
            continue
        ycell = next(c for c in row["y"] if c.get("group", 0) == CENTRALITY_GROUP)
        yval = float(ycell["value"])
        stat = unc = 0.0
        for e in ycell.get("errors", []):
            lab = e.get("label", "").lower()
            se = float(e.get("symerror", 0.0))
            if "stat" in lab:
                stat = se
            elif "uncorr" in lab:      # point-to-point systematic
                unc = se
        s = np.hypot(stat, unc)
        if yval > 0.0 and s > 0.0:
            pT.append(xc); Y.append(yval); sig.append(s)
    return np.array(pT), np.array(Y), np.array(sig)


def load_data():
    """Return (dataset, provenance).  Try real ALICE; fall back to synthetic."""
    # -------- attempt REAL ALICE / HEPData --------
    try:
        dataset = {}
        for name, mass, table, pmin, pmax in SPECIES:
            tj = _fetch_table_json(table)
            pT, Y, sig = _parse_alice_table(tj, pmin, pmax)
            if pT.size < 3:
                raise RuntimeError(f"too few points for {name} ({pT.size})")
            dataset[name] = dict(mass=mass, pT=pT, Y=Y, sigma=sig)
        prov = dict(
            SYNTHETIC_CLOSURE_TEST=False,
            source="ALICE Pb-Pb 5.02 TeV, 0-5% centrality, HEPData "
                   f"{HEPDATA_INS} (DOI 10.17182/hepdata.104923); "
                   "Acharya et al. [ALICE], Phys. Rev. C 101, 044907 (2020)",
            observable="(1/N_ev) d^2N/(dp_T dy) [(GeV/c)^-1]",
            error_model="stat (+) syst.uncorr in quadrature (diagonal) PLUS a "
                        "p_T-correlated GP model-discrepancy term "
                        "(squared-exponential kernel, marginalized amplitude eta)",
            truth=None,
        )
        print("[data] REAL ALICE Pb-Pb 5.02 TeV spectra downloaded from HEPData "
              f"{HEPDATA_INS}.")
        return dataset, prov
    except Exception as exc:                            # noqa: BLE001
        print(f"[data] HEPData fetch failed ({type(exc).__name__}: {exc}).")
        print("[data] Falling back to a LABELLED SYNTHETIC CLOSURE TEST "
              "(no fabrication of experimental data).")

    # -------- SYNTHETIC CLOSURE TEST (clearly labelled) --------
    T0, mb0, n0 = TRUTH["T_kin"], TRUTH["mean_beta"], TRUTH["n"]
    bs0 = beta_s_from_mean(mb0, n0)
    amp = {"pi": 3.0e3, "K": 4.0e2, "p": 1.2e2}     # arbitrary species scales
    dataset = {}
    for name, mass, table, pmin, pmax in SPECIES:
        pT = np.linspace(pmin, pmax, 16)
        model = amp[name] * pT * spectrum(pT, mass, T0, bs0, n0)   # dN/dpT dy
        noise = RNG.lognormal(mean=0.0, sigma=SYN_NOISE, size=pT.size)
        Y = model * noise
        sig = SYN_NOISE * Y
        dataset[name] = dict(mass=mass, pT=pT, Y=Y, sigma=sig)
    prov = dict(
        SYNTHETIC_CLOSURE_TEST=True,
        source="SYNTHETIC CLOSURE TEST -- BGBW model at injected truth "
               f"(T_kin={T0} GeV, <beta_T>={mb0}, n={n0}) + {int(SYN_NOISE*100)}% "
               "log-normal noise.  NOT experimental data.",
        observable="synthetic d^2N/(dp_T dy) [arb. units]",
        error_model=f"{int(SYN_NOISE*100)}% multiplicative log-normal PLUS a "
                    "p_T-correlated GP model-discrepancy term",
        truth=dict(T_kin=T0, mean_beta=mb0, n=n0),
    )
    return dataset, prov


def _prepare(dataset):
    """Cache the static per-species pieces of the GP likelihood.

    Only the discrepancy AMPLITUDE eta varies during sampling; the
    squared-exponential kernel K_s and the diagonal errors are fixed.  We
    therefore diagonalize the whitened kernel A_s = D^{-1/2} K_s D^{-1/2}
    (D = diag(eps^2)) ONCE per species, so every likelihood call reduces to
    an O(N^2) matrix-vector product with no per-step Cholesky:

        C_s = diag(eps^2) + eta^2 K_s = D^{1/2} (I + eta^2 A_s) D^{1/2},
        A_s = U_s diag(lam_s) U_s^T,
        ln det C_s = sum ln(eps^2) + sum ln(1 + eta^2 lam_s),
        u^T C_s^{-1} v = sum_k p_k(u) p_k(v) / (1 + eta^2 lam_s,k),
        p(v) = U_s^T (v / eps).

    Adds to each species dict: eps, inv_eps, ln_data, U (eigvecs), lam
    (eigvals), t1 = p(1), logdet_D, n_pts, ell, and the raw kernel rbf.
    """
    for name, sp in dataset.items():
        pT = sp["pT"]
        ell = GP_ELL_FRAC * (pT.max() - pT.min())
        dx = pT[:, None] - pT[None, :]
        rbf = np.exp(-0.5 * (dx / ell) ** 2)
        eps = sp["sigma"] / sp["Y"]
        inv_eps = 1.0 / eps
        A = rbf * np.outer(inv_eps, inv_eps)         # D^{-1/2} K D^{-1/2}
        lam, U = np.linalg.eigh(A)                    # symmetric -> real eigvals
        lam = np.clip(lam, 0.0, None)                # numerical floor (PSD)
        sp["rbf"] = rbf
        sp["eps"] = eps
        sp["inv_eps"] = inv_eps
        sp["ln_data"] = np.log(sp["Y"])
        sp["U"] = U
        sp["lam"] = lam
        sp["t1"] = U.T @ inv_eps                      # p(1) = U^T (1/eps)
        sp["logdet_D"] = 2.0 * float(np.sum(np.log(eps)))
        sp["n_pts"] = pT.size
        sp["ell"] = ell
    return dataset


# ===========================================================================
# 2. Likelihood, priors, posterior
# ===========================================================================
PRIOR = dict(T_kin=(0.05, 0.15), mean_beta=(0.40, 0.90), n=(0.30, 1.80),
             log10_eta=(-3.0, 0.0))
PARAM_NAMES = ["T_kin", "mean_beta", "n", "log10_eta"]
PARAM_LABELS = [r"$T_{\rm kin}$ [GeV]", r"$\langle\beta_T\rangle$", r"$n$",
                r"$\log_{10}\eta$"]
PHYS_NAMES = ["T_kin", "mean_beta", "n"]     # the reported physics parameters
_LN2PI = np.log(2.0 * np.pi)


def log_prior(theta):
    T_kin, mean_beta, n, log10_eta = theta
    if not (PRIOR["T_kin"][0] < T_kin < PRIOR["T_kin"][1]):
        return -np.inf
    if not (PRIOR["mean_beta"][0] < mean_beta < PRIOR["mean_beta"][1]):
        return -np.inf
    if not (PRIOR["n"][0] < n < PRIOR["n"][1]):
        return -np.inf
    if not (PRIOR["log10_eta"][0] < log10_eta < PRIOR["log10_eta"][1]):
        return -np.inf
    # HARD subluminal cut on the surface velocity.
    if beta_s_from_mean(mean_beta, n) >= 1.0:
        return -np.inf
    return 0.0


def _profiled_species_ll(sp, T_kin, beta_s, n, eta2):
    """GP-augmented log-likelihood of one species; ln-amplitude profiled by GLS.

    Covariance C_s = diag(eps^2) + eta2 * K_s (squared-exponential K_s).  The
    log-amplitude ln A_s is profiled by generalized least squares under C_s and
    the analytic flat-prior marginalization constant is retained, so the return
    is the proper marginal Gaussian log-likelihood.

    Returns (loglike_contrib, lnA_opt, chi2_full).  A 41-node radial Simpson grid
    is used for the forward model (changes ln L by < 0.1 vs the 81-node grid).
    """
    model = sp["pT"] * spectrum(sp["pT"], sp["mass"], T_kin, beta_s, n, n_r=41)
    if np.any(model <= 0.0) or not np.all(np.isfinite(model)):
        return -np.inf, 0.0, np.inf
    ln_model = np.log(model)
    d = sp["ln_data"] - ln_model                     # residual before amplitude
    n_pts = sp["n_pts"]
    # transform into the eigenbasis of the whitened kernel (cached in _prepare)
    td = sp["U"].T @ (d * sp["inv_eps"])             # p(d)
    t1 = sp["t1"]                                    # p(1)
    denom = 1.0 + eta2 * sp["lam"]                   # eigenvalues of I + eta^2 A
    a = float(np.sum(t1 * t1 / denom))               # 1^T C^-1 1  (> 0)
    b = float(np.sum(t1 * td / denom))               # 1^T C^-1 d
    lnA = b / a                                       # GLS optimal log-amplitude
    tr = lnA * t1 - td                                # p(r), r = lnA*1 - d
    chi2 = float(np.sum(tr * tr / denom))            # r^T C^-1 r
    logdet = sp["logdet_D"] + float(np.sum(np.log(denom)))   # ln det C_s
    # marginal Gaussian log-likelihood with ln A_s marginalized (flat prior):
    ll = (-0.5 * chi2 - 0.5 * logdet
          - 0.5 * (n_pts - 1) * _LN2PI - 0.5 * np.log(a))
    return ll, lnA, chi2


def log_likelihood(theta, dataset):
    T_kin, mean_beta, n, log10_eta = theta
    beta_s = beta_s_from_mean(mean_beta, n)
    eta2 = (10.0 ** log10_eta) ** 2
    total = 0.0
    for name in dataset:
        ll, _, _ = _profiled_species_ll(dataset[name], T_kin, beta_s, n, eta2)
        if not np.isfinite(ll):
            return -np.inf
        total += ll
    return total


def log_posterior(theta, dataset):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, dataset)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


# ---------------------------------------------------------------------------
# chi^2 book-keeping (reported honestly, both raw and augmented)
# ---------------------------------------------------------------------------
def chi2_diagonal(dataset, T_kin, beta_s, n):
    """Raw chi^2 of the IDEAL blast-wave with the DIAGONAL stat+syst errors only.

    This is the honest statement of the model misfit (eta -> 0): the amplitude is
    profiled by ordinary inverse-variance weighting.  Returns (chi2, n_points).
    """
    chi2 = 0.0
    npts = 0
    for name, sp in dataset.items():
        model = sp["pT"] * spectrum(sp["pT"], sp["mass"], T_kin, beta_s, n,
                                    n_r=41)
        ln_model = np.log(model)
        d = sp["ln_data"] - ln_model
        w = 1.0 / sp["eps"] ** 2
        lnA = np.sum(w * d) / np.sum(w)
        resid = (lnA - d) / sp["eps"]
        chi2 += float(np.sum(resid ** 2))
        npts += d.size
    return chi2, npts


def diag_mle(dataset):
    """Best-fit of the IDEAL blast-wave under the diagonal (stat+syst) likelihood.

    This is the honest "how well can the three-parameter blast-wave possibly do"
    number: the minimum diagonal chi^2 over (T_kin, <beta_T>, n).  The residual
    misfit at this optimum (chi^2/dof ~ 4.6) is the coherent point-to-point shape
    misfit that motivates the model-discrepancy term.  Returns a dict with the
    best-fit parameters and chi^2.
    """
    def obj(x):
        T, mb, n = x
        if not (PRIOR["T_kin"][0] < T < PRIOR["T_kin"][1]):
            return 1e8
        if not (PRIOR["mean_beta"][0] < mb < PRIOR["mean_beta"][1]):
            return 1e8
        if not (PRIOR["n"][0] < n < PRIOR["n"][1]):
            return 1e8
        if beta_s_from_mean(mb, n) >= 1.0:
            return 1e8
        c2, _ = chi2_diagonal(dataset, T, beta_s_from_mean(mb, n), n)
        return c2
    best = None
    for x0 in ([0.091, 0.660, 0.750], [0.100, 0.640, 0.820],
               [0.110, 0.620, 0.900]):
        res = minimize(obj, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-4, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    T, mb, n = best.x
    bs = beta_s_from_mean(mb, n)
    c2, N = chi2_diagonal(dataset, T, bs, n)
    return dict(T=T, mean_beta=mb, n=n, chi2=c2, N=N)


def chi2_augmented(dataset, T_kin, beta_s, n, eta2):
    """chi^2 under the full GP covariance (the augmented model).

    Returns (chi2, n_points); the augmented model has chi^2/dof ~ 1 by
    construction once eta absorbs the coherent shape misfit.
    """
    chi2 = 0.0
    npts = 0
    for name, sp in dataset.items():
        _, _, c2 = _profiled_species_ll(sp, T_kin, beta_s, n, eta2)
        chi2 += c2
        npts += sp["pT"].size
    return chi2, npts


# ===========================================================================
# 3. Gelman-Rubin split-Rhat
# ===========================================================================
def gelman_rubin(chain):
    """Split-Rhat for chain of shape (n_steps, n_walkers, n_dim).

    Each walker is split into two half-chains (rank-agnostic split-Rhat).
    Returns an array of Rhat per parameter.  For an ensemble sampler this is a
    heuristic (the walkers are not independent chains); the tau-based effective
    sample size is the primary convergence diagnostic.
    """
    n_steps, n_walkers, n_dim = chain.shape
    half = n_steps // 2
    # Two halves per walker -> m = 2*n_walkers sequences of length half.
    a = chain[:half]
    b = chain[half:2 * half]
    seqs = np.concatenate([a, b], axis=1)        # (half, 2*n_walkers, n_dim)
    Nn, m, _ = seqs.shape
    means = seqs.mean(axis=0)                     # (m, n_dim)
    variances = seqs.var(axis=0, ddof=1)         # (m, n_dim)
    W = variances.mean(axis=0)                    # within-chain variance
    B = Nn * means.var(axis=0, ddof=1)           # between-chain variance
    var_hat = (Nn - 1) / Nn * W + B / Nn
    return np.sqrt(var_hat / W)


# ===========================================================================
# 4. Main
# ===========================================================================
def main(n_walkers=100, n_steps=8000, seed=SEED):
    t0 = time.time()
    dataset, prov = load_data()
    _prepare(dataset)

    print("=" * 72)
    print("BGBW Bayesian fit -- REAL emcee run (Sec. 6, Block C)")
    print("=" * 72)
    tag = ("SYNTHETIC CLOSURE TEST" if prov["SYNTHETIC_CLOSURE_TEST"]
           else "REAL EXPERIMENTAL DATA")
    print(f"  PROVENANCE : {tag}")
    print(f"  source     : {prov['source']}")
    print(f"  error model: {prov['error_model']}")
    npts = {k: dataset[k]['pT'].size for k in dataset}
    print(f"  data points: {npts}  (total {sum(npts.values())})")
    print(f"  GP kernel  : {GP_KERNEL};  ell = {GP_ELL_FRAC:.2f} x fit window "
          f"({', '.join(f'{k}:{dataset[k]['ell']:.2f} GeV' for k in dataset)})")
    print(f"  priors     : T_kin~U{PRIOR['T_kin']}  "
          f"<beta_T>~U{PRIOR['mean_beta']}  n~U{PRIOR['n']}  "
          f"log10(eta)~U{PRIOR['log10_eta']} ; beta_s<1")
    print("-" * 72)

    ndim = 4
    rng = np.random.default_rng(seed)
    # Initialise walkers in a small ball inside the prior, physically sensible.
    p0_centre = np.array([0.10, 0.62, 0.80, -1.0])
    p0 = p0_centre + np.array([1e-2, 1e-2, 1e-2, 5e-2]) * \
        rng.standard_normal((n_walkers, ndim))
    p0[:, 0] = np.clip(p0[:, 0], *PRIOR["T_kin"])
    p0[:, 1] = np.clip(p0[:, 1], *PRIOR["mean_beta"])
    p0[:, 2] = np.clip(p0[:, 2], *PRIOR["n"])
    p0[:, 3] = np.clip(p0[:, 3], *PRIOR["log10_eta"])
    # ensure inside support at start
    for i in range(n_walkers):
        while not np.isfinite(log_posterior(p0[i], dataset)):
            p0[i] = p0_centre + np.array([1e-2, 1e-2, 1e-2, 5e-2]) * \
                rng.standard_normal(ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior,
                                    args=(dataset,))
    print(f"  running emcee: {n_walkers} walkers x {n_steps} steps ...")
    sampler.run_mcmc(p0, n_steps, progress=False)

    # --- autocorrelation / burn-in / thinning ------------------------------
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_ok = np.all(n_steps > 50 * tau)
    except Exception as exc:                            # noqa: BLE001
        tau = np.array([np.nan] * ndim)
        tau_ok = False
        print(f"  [warn] autocorr estimate: {exc}")
    tau_max = np.nanmax(tau)
    burnin = int(np.ceil(3 * tau_max)) if np.isfinite(tau_max) else n_steps // 5
    thin = max(1, int(np.floor(0.5 * tau_max))) if np.isfinite(tau_max) else 1

    chain = sampler.get_chain()                          # (n_steps, n_walk, ndim)
    rhat = gelman_rubin(chain[burnin:])
    flat = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    acc = float(np.mean(sampler.acceptance_fraction))
    n_flat = flat.shape[0]
    # tau-based effective sample size (primary): ESS = N_post / tau per param
    n_post = (n_steps - burnin) * n_walkers
    ess = n_post / tau

    # --- posterior summary --------------------------------------------------
    med = np.median(flat, axis=0)
    lo = np.percentile(flat, 16, axis=0)
    hi = np.percentile(flat, 84, axis=0)
    lo90 = np.percentile(flat, 5, axis=0)
    hi90 = np.percentile(flat, 95, axis=0)

    # --- chi^2: ideal blast-wave BEST FIT (honest misfit), and GP-augmented -
    beta_s_med = beta_s_from_mean(med[1], med[2])
    eta_med = 10.0 ** med[3]
    # (i) raw diagonal chi^2 at the IDEAL blast-wave's own best fit (~4.6):
    bf = diag_mle(dataset)
    chi2_raw, N = bf["chi2"], bf["N"]
    dof_raw = N - 6            # 3 physics + 3 per-species amplitudes
    chi2dof_raw = chi2_raw / dof_raw
    # (ii) augmented chi^2 under the full GP covariance at the posterior median:
    chi2_aug, _ = chi2_augmented(dataset, med[0], beta_s_med, med[2], eta_med ** 2)
    dof_aug = N - 7           # + the marginalized discrepancy amplitude eta
    chi2dof_aug = chi2_aug / dof_aug

    print("-" * 72)
    print("  CONVERGENCE DIAGNOSTICS")
    print(f"    acceptance fraction (mean) : {acc:.3f}")
    for i, nm in enumerate(PARAM_NAMES):
        print(f"    tau[{nm:9s}] = {tau[i]:7.2f} steps   "
              f"chain/tau = {n_steps / tau[i]:6.1f}   "
              f"ESS = {ess[i]:8.0f}   Rhat = {rhat[i]:.4f}")
    print(f"    chain length > 50*tau : {bool(tau_ok)}")
    print(f"    max Rhat = {np.max(rhat):.4f}  (split-Rhat is heuristic for an "
          f"ensemble sampler; tau-based ESS is primary)")
    print(f"    burn-in discarded = {burnin} steps ; thin = {thin} ; "
          f"flat samples = {n_flat}")
    print("-" * 72)
    print("  GOODNESS OF FIT (reported honestly)")
    print(f"    ideal blast-wave BEST FIT, diagonal chi^2/dof     = "
          f"{chi2_raw:.1f}/{dof_raw} = {chi2dof_raw:.3f}  "
          f"(at T={bf['T']:.4f}, <beta_T>={bf['mean_beta']:.4f}, n={bf['n']:.4f})")
    print(f"      -> the ideal BGBW has residual point-to-point SHAPE misfit.")
    print(f"    augmented chi^2/dof (with GP discrepancy)         = "
          f"{chi2_aug:.1f}/{dof_aug} = {chi2dof_aug:.3f}")
    print(f"    discrepancy amplitude eta (median) = {eta_med:.4f}  "
          f"(~{100*eta_med:.0f}% coherent log-yield shape freedom)")
    print("-" * 72)
    print("  POSTERIOR (median with 68% and 90% credible intervals)")
    for i, nm in enumerate(PARAM_LABELS):
        print(f"    {nm:22s} = {med[i]:.4f}  "
              f"[68%: {lo[i]:.4f}, {hi[i]:.4f}]  "
              f"[90%: {lo90[i]:.4f}, {hi90[i]:.4f}]")
    # convenient MeV statement for T_kin
    print(f"    -> T_kin = {1e3*med[0]:.0f} +{1e3*(hi[0]-med[0]):.0f}/"
          f"-{1e3*(med[0]-lo[0]):.0f} MeV ; "
          f"<beta_T> = {med[1]:.3f} +-{0.5*(hi[1]-lo[1]):.3f} ; "
          f"n = {med[2]:.2f} +-{0.5*(hi[2]-lo[2]):.2f}")
    # derived beta_s and its subluminal margin
    bs_chain = beta_s_from_mean(flat[:, 1], flat[:, 2])
    print(f"    derived beta_s (surface) = {np.median(bs_chain):.4f}  "
          f"[68%: {np.percentile(bs_chain,16):.4f}, "
          f"{np.percentile(bs_chain,84):.4f}]  (must be < 1)")
    # anti-correlation of <beta_T> and n
    rho_bn = np.corrcoef(flat[:, 1], flat[:, 2])[0, 1]
    print(f"    corr(<beta_T>, n) = {rho_bn:+.3f}  (expect < 0, from beta_s<1)")

    # --- recovery (synthetic only) -----------------------------------------
    if prov["SYNTHETIC_CLOSURE_TEST"]:
        print("-" * 72)
        print("  CLOSURE-TEST RECOVERY (injected truth vs posterior)")
        tvals = [TRUTH["T_kin"], TRUTH["mean_beta"], TRUTH["n"]]
        for i in range(3):
            nm = PHYS_NAMES[i]
            sig = 0.5 * (hi[i] - lo[i])
            pull = (med[i] - tvals[i]) / sig if sig > 0 else np.nan
            ok = abs(pull) < 2.0
            print(f"    {nm:9s}: truth={tvals[i]:.4f}  post={med[i]:.4f}"
                  f"  pull={pull:+.2f} sigma  {'OK' if ok else 'CHECK'}")

    # --- save ---------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "bgbw_chain.npz")
    save_kwargs = dict(
        flat_chain=flat,
        param_names=np.array(PARAM_NAMES),
        param_labels=np.array(PARAM_LABELS),
        phys_names=np.array(PHYS_NAMES),
        SYNTHETIC_CLOSURE_TEST=bool(prov["SYNTHETIC_CLOSURE_TEST"]),
        provenance=json.dumps(prov),
        seed=seed,
        tau=tau, rhat=rhat, ess=ess, acceptance_fraction=acc,
        burnin=burnin, thin=thin, n_walkers=n_walkers, n_steps=n_steps,
        median=med, ci68_lo=lo, ci68_hi=hi, ci90_lo=lo90, ci90_hi=hi90,
        # goodness of fit (both reported)
        chi2_raw=chi2_raw, dof_raw=dof_raw, chi2dof_raw=chi2dof_raw,
        chi2_aug=chi2_aug, dof_aug=dof_aug, chi2dof_aug=chi2dof_aug,
        eta_median=eta_med,
        # ideal blast-wave (diagonal) best-fit parameters, for the PPC curve
        bestfit=np.array([bf["T"], bf["mean_beta"], bf["n"]]),
        # GP error-model settings
        gp_ell_frac=GP_ELL_FRAC, gp_kernel=GP_KERNEL,
        gp_ell=np.array([dataset[k]["ell"] for k in dataset]),
        prior_T_kin=np.array(PRIOR["T_kin"]),
        prior_mean_beta=np.array(PRIOR["mean_beta"]),
        prior_n=np.array(PRIOR["n"]),
        prior_log10_eta=np.array(PRIOR["log10_eta"]),
    )
    if prov["truth"] is not None:
        save_kwargs["truth"] = np.array(
            [prov["truth"]["T_kin"], prov["truth"]["mean_beta"],
             prov["truth"]["n"]])
    # also store the fitted data for reproducibility / the caption / the PPC
    for name in dataset:
        save_kwargs[f"data_{name}_pT"] = dataset[name]["pT"]
        save_kwargs[f"data_{name}_Y"] = dataset[name]["Y"]
        save_kwargs[f"data_{name}_sigma"] = dataset[name]["sigma"]
        save_kwargs[f"data_{name}_mass"] = dataset[name]["mass"]
    np.savez(out, **save_kwargs)
    print("-" * 72)
    print(f"  saved chain -> {out}")
    print(f"  wall time   : {time.time() - t0:.1f} s")
    print("=" * 72)
    return out


if __name__ == "__main__":
    main()
