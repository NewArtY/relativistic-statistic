# -*- coding: utf-8 -*-
r"""
pinn_tsallis_boltzmann.py
=========================

WHAT IT DOES
------------
A minimal but *genuinely runnable* physics-informed neural network (PINN) in
PyTorch for a 1D relaxation-type kinetic equation whose equilibrium is a Tsallis
q-exponential (q-Gaussian).  It solves the FORWARD problem (learn the field
f_theta(t, v)) and, simultaneously, an INVERSE problem: the Tsallis
non-extensivity index q is a single trainable scalar q_theta, recovered from
synthetic, noisy "observations" that were generated at a known q_true.

This is the supplementary Block-D prototype of the article (Sec. 8.3 outlook)
and is the real training run behind Fig. 12.  It is intentionally a *downscaled,
CPU-friendly* version of the plan's full 256/256/128 relativistic-Boltzmann
network: the physics (relaxation to a q-exponential + trainable q recovered by
enforcing the PDE on data-fitted field) is identical, but the geometry is
reduced to a spatially homogeneous velocity relaxation so the whole thing
trains to completion on a laptop CPU in a couple of minutes.

THE EQUATION SOLVED
-------------------
Spatially homogeneous BGK / relaxation-time kinetic equation for a 1D velocity
distribution f(t, v) on (t, v) in [0, t_max] x [-v_max, v_max]:

        d f / d t  =  - (1/tau) [ f(t, v) - f_eq(v; q) ] ,

whose stationary state is the (normalized) Tsallis q-Gaussian

        f_eq(v; q)  =  N(q) * exp_q( - v^2 / (2 T) ) ,
        exp_q(x)    =  [ 1 + (1 - q) x ]_+^{ 1 / (1 - q) }  ->  e^x  as q -> 1,

so f_eq(v; q) ~ |v|^{-2/(q-1)} is a heavy-tailed power law for q > 1 and reduces
to the Maxwellian Gaussian as q -> 1 (consistent with the bare-occupation tail
convention of 00_conventions.tex, f_q ~ eps^{-1/(q-1)}).  The initial condition
is a *Boltzmann-Gibbs* (q = 1) Gaussian,

        f(0, v)  =  f_0(v)  =  N_0 * exp( - v^2 / (2 T) ) ,

so the field physically relaxes from a Gaussian toward the q-Gaussian.  Both
f_0 and f_eq are normalized to unit integral over the velocity domain, so the
relaxation conserves particle number; the exact reference solution is

        f_star(t, v)  =  f_eq(v; q_true)
                         + [ f_0(v) - f_eq(v; q_true) ] * exp(-t/tau) .

LOSS
----
L = w_pde * L_PDE + w_ic * L_IC + w_cons * L_cons + w_data * L_data , with the
plan's PDE : BC : cons weighting 10 : 1 : 5 (plus a data-misfit term):

    L_PDE  = < ( d_t f_theta + (1/tau)(f_theta - f_eq(v; q_theta)) )^2 >   (collocation)
    L_IC   = < ( f_theta(0, v) - f_0(v) )^2 >                              (initial data)
    L_cons = < ( \int f_theta(t, v) dv - 1 )^2 >                           (number conservation)
    L_data = < ( f_theta(t_obs, v_obs) - f_obs )^2 >                       (noisy synthetic obs)

The trainable q_theta enters ONLY through f_eq in L_PDE.  The data pin the field
toward f_star(.; q_true); enforcing the PDE then drags q_theta -> q_true.  This
is the standard inverse-PINN mechanism (an unknown PDE parameter recovered by
requiring the fitted field to satisfy the physics).

PROVENANCE
----------
COMPUTED from a real PINN training run.  Running this script trains the network
with Adam on CPU (fixed seed) and writes the full training history
(loss components and q_theta per logged epoch, plus final field snapshots) to
    code/data/pinn_history.npz
which is consumed verbatim by code/figures/fig12_pinn.py.  Nothing is mocked.

ACCEPTANCE (Block D, printed at run time)
-----------------------------------------
  (1) L_total decreases by >= 3 orders of magnitude;
  (2) recovered q_theta matches q_true to < 5%;
  (3) the solution conserves particle number to numerical accuracy.

Run:  python pinn_tsallis_boltzmann.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(CODE_DIR, "data")
HISTORY_PATH = os.path.join(DATA_DIR, "pinn_history.npz")

# ---------------------------------------------------------------------------
# Reproducibility & problem constants
# ---------------------------------------------------------------------------
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)

T_TEMP = 1.0        # temperature (natural units, sets the Gaussian width)
TAU = 0.5           # relaxation time
T_MAX = 3.0         # time horizon ( >> tau : field nearly fully relaxed )
V_MAX = 6.0         # velocity half-window (q-Gaussian tail is negligible beyond)

Q_TRUE = 1.15       # ground-truth non-extensivity index (generates the data)
Q_INIT = 1.30       # deliberately wrong initial guess for the inverse problem

# Loss weights: plan's PDE : BC : cons = 10 : 1 : 5, plus a data term.
W_PDE, W_IC, W_CONS, W_DATA = 10.0, 1.0, 5.0, 15.0

# Training / sampling sizes (small: CPU, a few minutes).
N_COLLOC = 2500     # interior PDE collocation points
N_IC = 200          # initial-condition points (t = 0)
N_DATA = 300        # synthetic noisy observations
N_TCONS = 20        # time slices for the conservation integral
N_VQUAD = 161       # velocity nodes for trapezoidal quadratures
EPOCHS = 5000
LOG_EVERY = 25
OBS_NOISE = 0.02    # 2% relative Gaussian noise on the observations


# ===========================================================================
# 1. Tsallis q-Gaussian equilibrium (differentiable in q) and BG initial state
# ===========================================================================
# Fixed dense velocity grid used to normalize the (unnormalized) distributions
# to unit integral over the domain; trapezoidal weights precomputed once.
_V_NORM = torch.linspace(-V_MAX, V_MAX, 401)
_DV_NORM = _V_NORM[1] - _V_NORM[0]


def _trapz(y, dx):
    """Composite-trapezoid integral of y (last-axis samples) with spacing dx."""
    return dx * (y[..., 1:] + y[..., :-1]).sum(dim=-1) * 0.5


def exp_q_torch(x, q):
    """Tsallis q-exponential in torch (differentiable in x and q).

        exp_q(x) = [1 + (1 - q) x]_+^{1/(1-q)} .

    For q > 1 and x = -v^2/(2T) <= 0 the base 1 + (q-1) v^2/(2T) >= 1 is always
    positive, so the q-Gaussian is smooth and strictly positive; the clamp only
    guards against a transient excursion q <= 1 during optimization.
    """
    base = 1.0 + (1.0 - q) * x
    base = torch.clamp(base, min=1e-9)
    return base.pow(1.0 / (1.0 - q))


def f_eq_unnorm(v, q):
    """Unnormalized Tsallis q-Gaussian equilibrium exp_q(-v^2/(2T))."""
    return exp_q_torch(-(v * v) / (2.0 * T_TEMP), q)


def norm_const(q):
    r"""N(q) = 1 / \int_{-V_MAX}^{V_MAX} exp_q(-v^2/2T) dv  (differentiable in q)."""
    integ = _trapz(f_eq_unnorm(_V_NORM, q), _DV_NORM)
    return 1.0 / integ


def f_eq(v, q):
    """Normalized q-Gaussian equilibrium f_eq(v; q) = N(q) exp_q(-v^2/2T)."""
    return norm_const(q) * f_eq_unnorm(v, q)


# Boltzmann-Gibbs (q = 1) Gaussian initial condition, normalized to unit
# integral over the domain (a fixed numpy constant -> torch scalar).
_N0 = 1.0 / float(_trapz(torch.exp(-(_V_NORM ** 2) / (2.0 * T_TEMP)), _DV_NORM))


def f0(v):
    """Initial condition f_0(v) = N_0 exp(-v^2/2T)  (a Maxwellian Gaussian)."""
    return _N0 * torch.exp(-(v * v) / (2.0 * T_TEMP))


def f_star(t, v, q):
    """Exact reference solution of the relaxation equation at index q:

        f_star = f_eq(v; q) + (f_0(v) - f_eq(v; q)) exp(-t/tau).
    """
    fe = f_eq(v, q)
    return fe + (f0(v) - fe) * torch.exp(-t / TAU)


# ===========================================================================
# 2. The PINN: a small tanh MLP  (t, v) -> f_theta
# ===========================================================================
class PINN(nn.Module):
    """Small fully-connected tanh network with input normalization.

    Downscaled CPU configuration: 3 hidden layers of width 48 (~5k params),
    versus the plan's 256/256/128 GPU network.
    """

    def __init__(self, width=48, depth=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        # Glorot init for tanh nets.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, t, v):
        tn = 2.0 * t / T_MAX - 1.0          # -> [-1, 1]
        vn = v / V_MAX                       # -> [-1, 1]
        return self.net(torch.cat([tn, vn], dim=1))


# ===========================================================================
# 3. Synthetic observations (generated at q_true, with noise)
# ===========================================================================
def make_observations(rng):
    """N_DATA noisy observations of f_star(.; q_true) at random (t, v)."""
    t_obs = torch.from_numpy(rng.uniform(0.0, T_MAX, (N_DATA, 1)).astype(np.float32))
    v_obs = torch.from_numpy(rng.uniform(-V_MAX, V_MAX, (N_DATA, 1)).astype(np.float32))
    with torch.no_grad():
        clean = f_star(t_obs, v_obs, Q_TRUE)
    noise = torch.from_numpy(
        (rng.standard_normal((N_DATA, 1)) * OBS_NOISE).astype(np.float32))
    f_obs = clean * (1.0 + noise)            # multiplicative 2% noise
    return t_obs, v_obs, f_obs


# ===========================================================================
# 4. Training
# ===========================================================================
def train():
    rng = np.random.default_rng(SEED)
    device = torch.device("cpu")

    model = PINN().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Trainable inverse-problem scalar q_theta (initialized away from truth).
    q_theta = nn.Parameter(torch.tensor(Q_INIT))

    # --- fixed collocation / IC / conservation / data sets ------------------
    t_f = torch.from_numpy(rng.uniform(0.0, T_MAX, (N_COLLOC, 1)).astype(np.float32))
    v_f = torch.from_numpy(rng.uniform(-V_MAX, V_MAX, (N_COLLOC, 1)).astype(np.float32))
    t_f.requires_grad_(True)

    v_ic = torch.from_numpy(
        np.linspace(-V_MAX, V_MAX, N_IC).astype(np.float32)).reshape(-1, 1)
    t_ic = torch.zeros_like(v_ic)
    f_ic_target = f0(v_ic).detach()

    # Conservation: for each of N_TCONS times, integrate over a fixed v-grid.
    t_cons_1d = torch.from_numpy(
        np.linspace(0.0, T_MAX, N_TCONS).astype(np.float32))
    v_quad_1d = torch.linspace(-V_MAX, V_MAX, N_VQUAD)
    dv_quad = (v_quad_1d[1] - v_quad_1d[0]).item()
    TT, VV = torch.meshgrid(t_cons_1d, v_quad_1d, indexing="ij")
    t_cons = TT.reshape(-1, 1)
    v_cons = VV.reshape(-1, 1)

    t_obs, v_obs, f_obs = make_observations(rng)

    # --- optimizer: separate (faster) learning rate for the physics scalar --
    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1.0e-3},
        {"params": [q_theta], "lr": 3.0e-3},
    ])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)

    mse = nn.MSELoss()
    zero = torch.zeros(1)

    hist = {k: [] for k in ("epoch", "loss_total", "loss_pde", "loss_ic",
                            "loss_cons", "loss_data", "q")}

    print("=" * 72)
    print("PINN  -  Tsallis q-Gaussian relaxation kinetic equation (inverse q)")
    print("=" * 72)
    print("  equation : d_t f = -(f - f_eq(v;q))/tau ,  "
          "f_eq ~ exp_q(-v^2/2T)")
    print("  network  : MLP 2 -> [48 x 3, tanh] -> 1   (%d parameters)" % n_params)
    print("  weights  : PDE=%.0f  IC=%.0f  cons=%.0f  data=%.0f"
          % (W_PDE, W_IC, W_CONS, W_DATA))
    print("  q_true=%.4f   q_init=%.4f   T=%.1f  tau=%.1f  epochs=%d  seed=%d"
          % (Q_TRUE, Q_INIT, T_TEMP, TAU, EPOCHS, SEED))
    print("-" * 72)
    print("  %6s  %11s  %10s  %10s  %10s  %10s  %8s"
          % ("epoch", "L_total", "L_pde", "L_ic", "L_cons", "L_data", "q_theta"))

    t0 = time.perf_counter()
    for epoch in range(EPOCHS + 1):
        opt.zero_grad()

        # ---- PDE residual: d_t f + (f - f_eq(v;q_theta))/tau -------------
        f_col = model(t_f, v_f)
        df_dt = torch.autograd.grad(
            f_col, t_f, grad_outputs=torch.ones_like(f_col),
            create_graph=True)[0]
        res = df_dt + (f_col - f_eq(v_f, q_theta)) / TAU
        loss_pde = mse(res, torch.zeros_like(res))

        # ---- initial condition ------------------------------------------
        f_ic = model(t_ic, v_ic)
        loss_ic = mse(f_ic, f_ic_target)

        # ---- number conservation:  \int f(t,.) dv = 1  for each t -------
        f_cn = model(t_cons, v_cons).reshape(N_TCONS, N_VQUAD)
        mass = _trapz(f_cn, dv_quad)                  # (N_TCONS,)
        loss_cons = mse(mass, torch.ones_like(mass))

        # ---- data misfit -------------------------------------------------
        f_pred = model(t_obs, v_obs)
        loss_data = mse(f_pred, f_obs)

        loss = (W_PDE * loss_pde + W_IC * loss_ic
                + W_CONS * loss_cons + W_DATA * loss_data)
        loss.backward()
        opt.step()
        sched.step()

        if epoch % LOG_EVERY == 0:
            hist["epoch"].append(epoch)
            hist["loss_total"].append(loss.item())
            hist["loss_pde"].append(loss_pde.item())
            hist["loss_ic"].append(loss_ic.item())
            hist["loss_cons"].append(loss_cons.item())
            hist["loss_data"].append(loss_data.item())
            hist["q"].append(q_theta.item())
            if epoch % (LOG_EVERY * 20) == 0 or epoch == EPOCHS:
                print("  %6d  %11.4e  %10.3e  %10.3e  %10.3e  %10.3e  %8.5f"
                      % (epoch, float(loss), float(loss_pde), float(loss_ic),
                         float(loss_cons), float(loss_data), float(q_theta)))

    elapsed = time.perf_counter() - t0

    # ---- final field snapshots (for Fig. 12 context) -----------------------
    v_plot = torch.linspace(-V_MAX, V_MAX, 241).reshape(-1, 1)
    t_late = torch.full_like(v_plot, T_MAX)
    with torch.no_grad():
        feq_true = f_eq(v_plot, torch.tensor(Q_TRUE)).squeeze().numpy()
        feq_rec = f_eq(v_plot, q_theta.detach()).squeeze().numpy()
        f_net_late = model(t_late, v_plot).squeeze().numpy()
        f0_plot = f0(v_plot).squeeze().numpy()

    q_rec = float(q_theta)
    rel_err = abs(q_rec - Q_TRUE) / Q_TRUE
    L0 = hist["loss_total"][0]
    Lf = hist["loss_total"][-1]
    orders = np.log10(L0 / Lf)
    mass_err = float(np.max(np.abs(mass.detach().numpy() - 1.0)))

    print("-" * 72)
    print("  training time            : %.1f s  (%d epochs, CPU)" % (elapsed, EPOCHS))
    print("  final total loss         : %.6e   (start %.4e, %.2f orders down)"
          % (Lf, L0, orders))
    print("  final components         : PDE=%.3e IC=%.3e cons=%.3e data=%.3e"
          % (hist["loss_pde"][-1], hist["loss_ic"][-1],
             hist["loss_cons"][-1], hist["loss_data"][-1]))
    print("  q_true = %.5f  ->  q_recovered = %.5f   (rel. error %.3f%%)"
          % (Q_TRUE, q_rec, 100.0 * rel_err))
    print("  number conservation      : max |mass - 1| = %.3e" % mass_err)
    print("-" * 72)
    print("  ACCEPTANCE (Block D):")
    print("    (1) loss down >= 3 orders : %s  (%.2f orders)"
          % ("PASS" if orders >= 3.0 else "FAIL", orders))
    print("    (2) q recovered  < 5%%     : %s  (%.3f%%)"
          % ("PASS" if rel_err < 0.05 else "FAIL", 100.0 * rel_err))
    print("    (3) conservation held     : %s  (%.2e)"
          % ("PASS" if mass_err < 1e-2 else "FAIL", mass_err))
    print("=" * 72)

    # ---- persist the real history for the figure ---------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(
        HISTORY_PATH,
        epoch=np.array(hist["epoch"]),
        loss_total=np.array(hist["loss_total"]),
        loss_pde=np.array(hist["loss_pde"]),
        loss_ic=np.array(hist["loss_ic"]),
        loss_cons=np.array(hist["loss_cons"]),
        loss_data=np.array(hist["loss_data"]),
        q_hist=np.array(hist["q"]),
        q_true=Q_TRUE, q_init=Q_INIT, q_recovered=q_rec, q_rel_err=rel_err,
        w_pde=W_PDE, w_ic=W_IC, w_cons=W_CONS, w_data=W_DATA,
        T=T_TEMP, tau=TAU, t_max=T_MAX, v_max=V_MAX,
        n_params=n_params, epochs=EPOCHS, seed=SEED,
        train_seconds=elapsed, loss_orders=orders, mass_err=mass_err,
        arch="MLP 2 -> [48x3 tanh] -> 1",
        v_plot=v_plot.squeeze().numpy(),
        feq_true=feq_true, feq_recovered=feq_rec,
        f_net_late=f_net_late, f0=f0_plot,
    )
    print("  wrote history -> %s" % HISTORY_PATH)
    return HISTORY_PATH


if __name__ == "__main__":
    train()
