# -*- coding: utf-8 -*-
r"""
fig12_pinn.py  ->  figures/fig12_pinn.{pdf,png}   (Fig. 12, fig:pinn)
=====================================================================

WHAT IT SHOWS
-------------
Two panels documenting the supplementary Block-D physics-informed neural
network (PINN) prototype (Sec. 8.3 outlook):

  (a) A schematic of the PINN architecture: the inputs (t, v) feed a small
      fully-connected tanh MLP that outputs the field f_theta(t, v); the three
      physics losses L_PDE, L_BC (initial condition) and L_cons (number
      conservation) constrain it, and the Tsallis non-extensivity index q is a
      single TRAINABLE scalar q_theta -- the inverse-problem knob -- that enters
      the q-Gaussian equilibrium f_eq inside the PDE residual and is recovered
      from noisy data.

  (b) The ACTUAL training convergence of that run: the total loss and its four
      components versus epoch on a logarithmic axis, with a twin axis tracking
      the trainable q_theta as it is driven from its wrong initial guess toward
      the ground-truth q_true.  The recovered value is annotated.

THE EQUATION SOLVED (see pinn_tsallis_boltzmann.py)
---------------------------------------------------
    d_t f = -(1/tau)[ f(t,v) - f_eq(v; q) ] ,
    f_eq(v; q) = N(q) exp_q(-v^2/2T)  (Tsallis q-Gaussian, heavy-tailed q>1),
with a Boltzmann-Gibbs Gaussian initial condition; q recovered as an inverse
problem from synthetic observations generated at q_true = 1.15.

PROVENANCE:  COMPUTED from a real PINN training run.  Panel (b) is drawn
verbatim from code/data/pinn_history.npz, written by
code/pinn/pinn_tsallis_boltzmann.py (Adam, CPU, fixed seed).  If that file is
absent, this script runs the prototype first.  Panel (a) is a schematic
(matplotlib patches) that matches the trained architecture exactly.

CONVENTION
----------
Natural units c = hbar = k_B = 1 (Eq. conv-natural-units); q is the Tsallis
index with the bare-occupation tail f_q ~ eps^{-1/(q-1)} (Sec. conv-tails).

Run:  python fig12_pinn.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import _common as C  # noqa: E402

DATA = os.path.join(CODE_DIR, "data", "pinn_history.npz")
PROTO = os.path.join(CODE_DIR, "pinn", "pinn_tsallis_boltzmann.py")

# colour tokens (style.mplstyle CB-safe cycle)
CLR_TOTAL = "#000000"
CLR_PDE = "#0077BB"
CLR_IC = "#EE7733"
CLR_CONS = "#009988"
CLR_DATA = "#AA3377"
CLR_Q = "#CC3311"


def load_history():
    """Load pinn_history.npz, running the real training prototype if absent."""
    if not os.path.exists(DATA):
        print("pinn_history.npz not found -- running the PINN prototype first ...")
        subprocess.run([sys.executable, PROTO], cwd=CODE_DIR, check=True)
    return np.load(DATA, allow_pickle=True)


# ===========================================================================
# Panel (a): architecture schematic
# ===========================================================================
def draw_schematic(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

    # ---- node layers: 2 inputs, 3 hidden (drawn 4 nodes each), 1 output ----
    layer_x = [1.1, 3.1, 4.6, 6.1, 8.1]
    layer_n = [2, 4, 4, 4, 1]
    layer_col = ["#333333", CLR_PDE, CLR_PDE, CLR_PDE, "#333333"]
    node_r = 0.26

    def ys(n):
        if n == 1:
            return [5.6]
        span = 3.0
        return list(np.linspace(5.6 + span / 2, 5.6 - span / 2, n))

    coords = []
    for x, n, col in zip(layer_x, layer_n, layer_col):
        col_coords = []
        for y in ys(n):
            face = "white" if col == "#333333" else "#DCEAF5"
            ax.add_patch(Circle((x, y), node_r, facecolor=face,
                                edgecolor=col, lw=1.3, zorder=3))
            col_coords.append((x, y))
        coords.append(col_coords)

    # ---- thin connections between successive layers ------------------------
    for a, b in zip(coords[:-1], coords[1:]):
        for (xa, ya) in a:
            for (xb, yb) in b:
                ax.plot([xa + node_r, xb - node_r], [ya, yb],
                        color="#B8C4CE", lw=0.35, zorder=1)

    # ---- input / output labels --------------------------------------------
    for (x, y), lab in zip(coords[0], [r"$t$", r"$v$"]):
        ax.text(x, y, lab, ha="center", va="center", fontsize=10, zorder=4)
    ax.text(coords[-1][0][0], coords[-1][0][1] + 0.0, r"$\tilde f_\theta$",
            ha="center", va="center", fontsize=9, zorder=4)

    ax.text(layer_x[0], 8.7, "inputs", ha="center", fontsize=9, color="#333333")
    ax.text(layer_x[2], 8.7, r"hidden $\tanh$ (3$\times$48)", ha="center",
            fontsize=9, color=CLR_PDE)
    ax.text(layer_x[4], 8.7, "output", ha="center", fontsize=9, color="#333333")
    ax.text(layer_x[4], 6.5, r"$\tilde f_\theta(t,v)$", ha="center",
            fontsize=9.5, color="#333333")

    # ---- trainable q knob feeding the equilibrium into the PDE loss --------
    ax.add_patch(FancyBboxPatch((0.35, 1.7), 2.5, 1.15,
                                boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#FBEAE7", edgecolor=CLR_Q, lw=1.6,
                                zorder=3))
    ax.text(1.6, 2.5, r"trainable $q_\theta$", ha="center", va="center",
            fontsize=10, color=CLR_Q, zorder=4)
    ax.text(1.6, 2.02, r"$f_{\rm eq}\!\propto\!\exp_q(-v^2/2T)$", ha="center",
            va="center", fontsize=8.2, color=CLR_Q, zorder=4)
    ax.text(1.6, 1.15, "inverse-problem knob", ha="center", va="center",
            fontsize=8.0, color=CLR_Q, style="italic")

    # ---- the three physics-loss boxes + data box --------------------------
    def loss_box(x, y, w, h, label, formula, color):
        # Render the loss NAME (e.g. L_PDE) large/bold for legibility and the
        # defining expression smaller beneath it, on two clearly-spaced rows.
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.1",
                                    facecolor="white", edgecolor=color, lw=1.5,
                                    zorder=3))
        ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color, zorder=4)
        ax.text(x + w / 2, y + h * 0.24, formula, ha="center", va="center",
                fontsize=7.6, color=color, zorder=4)

    loss_box(4.05, 3.15, 1.75, 0.95, r"$\mathcal{L}_{\rm PDE}$",
             r"$\partial_t\tilde f+\frac{\tilde f-f_{\rm eq}}{\tau}$", CLR_PDE)
    loss_box(6.05, 3.15, 1.55, 0.95, r"$\mathcal{L}_{\rm BC}$",
             r"$\tilde f|_{t=0}=f_0$", CLR_IC)
    loss_box(4.05, 1.55, 1.75, 0.95, r"$\mathcal{L}_{\rm cons}$",
             r"$\int\tilde f\,dv=1$", CLR_CONS)
    loss_box(6.05, 1.55, 1.55, 0.95, r"$\mathcal{L}_{\rm data}$",
             r"$\tilde f\!\to\! f_{\rm obs}$", CLR_DATA)

    # ---- arrows: output -> losses; q -> PDE loss; losses summed ------------
    def arrow(p0, p1, color, rad=0.0, lw=1.3, style="-|>"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                     mutation_scale=11, color=color, lw=lw,
                                     connectionstyle="arc3,rad=%g" % rad,
                                     zorder=2))

    ax.add_patch(FancyArrowPatch((coords[3][0][0] + node_r, coords[3][0][1]),
                                 (layer_x[4] - node_r, coords[-1][0][1]),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#333333", lw=1.4, zorder=2))
    # field -> each loss
    arrow((7.4, 5.2), (5.3, 4.1), "#888888", rad=0.25, lw=1.0)
    arrow((7.6, 5.2), (6.9, 4.1), "#888888", rad=-0.1, lw=1.0)
    arrow((7.4, 5.2), (5.0, 2.5), "#888888", rad=0.28, lw=1.0)
    arrow((7.7, 5.2), (7.0, 2.5), "#888888", rad=-0.28, lw=1.0)
    # trainable q -> PDE residual (the inverse-problem coupling)
    arrow((2.85, 2.7), (4.05, 3.5), CLR_Q, rad=-0.15, lw=1.6)

    ax.text(5.0, 0.75, r"$\mathcal{L}=10\,\mathcal{L}_{\rm PDE}"
            r"+\mathcal{L}_{\rm BC}+5\,\mathcal{L}_{\rm cons}"
            r"+15\,\mathcal{L}_{\rm data}$",
            ha="center", va="center", fontsize=8.8, color="#222222")

    ax.set_title("(a) PINN architecture and inverse-problem setup",
                 fontsize=10.5)


# ===========================================================================
# Panel (b): real training convergence + q_theta twin axis
# ===========================================================================
def draw_convergence(ax, d):
    ep = d["epoch"]
    q_true = float(d["q_true"])
    q_rec = float(d["q_recovered"])
    q_init = float(d["q_init"])
    rel_err = float(d["q_rel_err"])

    ax.plot(ep, d["loss_total"], color=CLR_TOTAL, lw=2.2, zorder=6,
            label=r"$\mathcal{L}_{\rm total}$")
    ax.plot(ep, d["loss_pde"], color=CLR_PDE, lw=1.3, ls="-",
            label=r"$\mathcal{L}_{\rm PDE}$")
    ax.plot(ep, d["loss_ic"], color=CLR_IC, lw=1.3, ls="--",
            label=r"$\mathcal{L}_{\rm BC}$")
    ax.plot(ep, d["loss_cons"], color=CLR_CONS, lw=1.3, ls=":",
            label=r"$\mathcal{L}_{\rm cons}$")
    ax.plot(ep, d["loss_data"], color=CLR_DATA, lw=1.3, ls="-.",
            label=r"$\mathcal{L}_{\rm data}$")

    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(ep.min(), ep.max())
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.0, ncol=2, handlelength=1.8,
              columnspacing=1.1)

    # --- twin axis: trainable q_theta -> q_true ----------------------------
    axq = ax.twinx()
    axq.plot(ep, d["q_hist"], color=CLR_Q, lw=2.0, zorder=5)
    axq.axhline(q_true, color=CLR_Q, lw=1.0, ls=(0, (4, 2)), alpha=0.7)
    axq.set_ylabel(r"trainable index  $q_\theta$", color=CLR_Q)
    axq.tick_params(axis="y", colors=CLR_Q)
    axq.spines["right"].set_visible(True)
    axq.spines["right"].set_color(CLR_Q)
    qmax = max(float(np.max(d["q_hist"])), q_init) + 0.1
    axq.set_ylim(1.0, qmax)

    axq.annotate(r"$q_{\rm true}=%.2f$" % q_true,
                 xy=(ep.max(), q_true), xytext=(ep.max() * 0.60, q_true + 0.16),
                 fontsize=8.6, color=CLR_Q, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=CLR_Q, lw=1.0))
    axq.annotate(r"$q_\theta^{\rm init}=%.2f$" % q_init,
                 xy=(ep[0], q_init), xytext=(ep.max() * 0.06, q_init + 0.18),
                 fontsize=8.2, color=CLR_Q, ha="left", va="bottom")
    # Recovered-value box: placed in the clear upper-right corner (the q_theta
    # curve peaks early on the LEFT and decays, so this region carries no
    # curve), with an opaque white background so it is fully legible.
    axq.text(ep.max() * 0.985, 1.0 + 0.82 * (qmax - 1.0),
             r"recovered $q_\theta=%.3f$" "\n" r"(err $%.2f\%%$)"
             % (q_rec, 100.0 * rel_err),
             ha="right", va="top", fontsize=8.6, color=CLR_Q,
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=CLR_Q, lw=1.0, alpha=1.0), zorder=8)

    ax.set_title("(b) Real training convergence (Adam, CPU)", fontsize=10.5)


# ===========================================================================
def make_figure():
    plt = C.apply_style()
    d = load_history()

    orders = float(d["loss_orders"])
    print("=" * 70)
    print("fig12_pinn: PINN convergence + q recovery (REAL run)")
    print("=" * 70)
    print("  arch            : %s" % str(d["arch"]))
    print("  epochs / seed   : %d / %d" % (int(d["epochs"]), int(d["seed"])))
    print("  train time      : %.1f s" % float(d["train_seconds"]))
    print("  loss: %.4e -> %.4e  (%.2f orders)"
          % (float(d["loss_total"][0]), float(d["loss_total"][-1]), orders))
    print("  q_true = %.4f  ->  q_recovered = %.4f  (rel err %.3f%%)"
          % (float(d["q_true"]), float(d["q_recovered"]),
             100.0 * float(d["q_rel_err"])))
    print("  mass conservation error : %.3e" % float(d["mass_err"]))
    print("-" * 70)

    fig = plt.figure(figsize=(10.4, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    draw_schematic(ax_a)
    draw_convergence(ax_b, d)

    paths = C.savefig_dual(fig, "fig12_pinn")
    print("  wrote:")
    for p in paths:
        print("    " + p)
    plt.close(fig)
    return paths


# alias matching the deliverable name (fig12_pinn)
fig12_pinn = make_figure


if __name__ == "__main__":
    make_figure()
