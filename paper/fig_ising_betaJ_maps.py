# FROZEN (2026-11-07). This file reproduces stale route-A `D1` figures and is deliberately
# NOT ported to the route-B API: run it from `git checkout 378bf2a^`, the last commit whose
# `rdme` it matches. Against the current library it will not even import the names it wants.
# The `S1` maps replace these figures; `D1` stays in rdme.presets as the record of their
# parameters and has no live caller.

""" (beta, J) phase maps of activity, entropy production and the two path entropies for the
single-population Ising network -- figure of the SciPost paper.

Self-contained by design: every parameter of the run is in this file and the cache lives in
`papers/scipost/cache`, so the figure can be regenerated from a clean checkout without any of
the development scripts in `scripts/` having been run, and nothing here can silently pick up
a cache some other script wrote. Only the `rdme` package is shared.

The cache holds the *reduced* statistics rather than the trajectories themselves.

    python papers/scipost/fig_ising_betaJ_maps.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
from rdme.rdm_batch import RDMIsingModelBatch


# ── Paths & switches ──────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
CACHE_DIR = HERE / "cache"
FIG_DIR   = HERE / "figures"
STATS_FILE = CACHE_DIR / "ising_betaJ_maps_stats.pt"

run_sim   = True
run_plot  = True
overwrite = False
torch.set_default_dtype(torch.float64)


# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS = dict(
    J       = torch.linspace(0, 18.0, 101),
    E       = 0.9,
    beta    = torch.linspace(1.0, 60.0, 101),
    theta   = 1.0,
    tau_int = 20.0,
    tau_ref = 3.0,
    K_ref   = 0.0,
    dt0     = 0.5,
    dt      = 0.5,
    hazard  = "async",
    eps     = 0.01,
)
EQUI, STEPS, CHUNK = 10000, 10000, 1024


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not STATS_FILE.exists()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # J and beta are passed as vectors, so the batch is their outer product and the trajectory
    # comes back shaped (n_J, n_beta, T) -- no meshgrid/reshape bookkeeping.
    # NOTE (superseded): this used to run at a step of 1.0 with J carrying a 1/dt, because
    # dt was not passed and defaulted to 1.0. PARAMS now sets dt and dt0 explicitly, dt0 is
    # a required argument, and the input is built from the rate -- so J is passed through
    # unscaled, used unscaled, and plotted unscaled (see rdm.compute_total_input).
    #
    # The numbers therefore already differ from the pre-port ones. What is NOT settled here
    # is whether the J range below is quoted in physical units or still carries the old
    # 1/dt: nothing in the code can tell, since both are valid inputs. Check it against the
    # axis label before the figure is used.
    mf = RDMIsingModelBatch(device=device, **PARAMS)
    print(f"Running {mf.B} mean-fields. Grid axes: "
          f"{ {k: len(v) for k, v in mf.grid_axes.items()} }.")
    print(f"Batch steps: {EQUI+STEPS} per mean-field. ({EQUI} equilibration + {STEPS} recording)")
    print(f"Running on: {device}")

    t0 = time.time()
    mf.forward(EQUI, pb=True)
    traj = mf.entropy_trajectory(STEPS, pb=True, chunk=CHUNK)
    print(f"Done in {time.time() - t0:.2f}s")

    # Reduce here and keep only this: the trajectories are what makes the run expensive to
    # store, and the figure never looks at them again.
    stats = {f"{k}_{s}": getattr(traj[k], s)(dim=-1)
             for k in ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")
             for s in ("mean", "std")}
    # grid_axes holds the vectors as they were passed in
    stats["J"]    = mf.grid_axes["J"]
    stats["beta"] = mf.grid_axes["beta"]
    stats["params"] = {k: v for k, v in PARAMS.items()}
    stats["steps"] = (EQUI, STEPS)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(stats, STATS_FILE)
    print(f"Statistics saved to {STATS_FILE}.")

elif run_sim:
    print(f"{STATS_FILE} exists and overwrite is off. Skipping simulation.")


# ── Figure ────────────────────────────────────────────────────────────────────

if run_plot:

    if not STATS_FILE.exists():
        raise FileNotFoundError(f"{STATS_FILE} not found. Run with run_sim = True first.")

    hlp.pub_style()

    stats   = torch.load(STATS_FILE, weights_only=False)
    J, beta = stats["J"], stats["beta"]
    extent  = (beta.min().item(), beta.max().item(), J.min().item(), J.max().item())

    m_avg, sigma_avg = stats["m_tot_mean"], stats["sigma_tot_mean"]
    H_fwd_avg, H_rev_avg = stats["H_fwd_tot_mean"], stats["H_rev_tot_mean"]

    # The two entropies are read against each other -- their difference is the entropy
    # production -- so they share a colour scale; m and Sigma are autoscaled.
    H_lim = (min(H_fwd_avg.min(), H_rev_avg.min()).item(),
             max(H_fwd_avg.max(), H_rev_avg.max()).item())

    # (field, colorbar label, panel title, colour limits)
    panels = [
        (m_avg,     r"$\langle m \rangle$",         "(a) activity",                   None),
        (sigma_avg, r"$\langle \Sigma \rangle$",    "(b) entropy production",         None),
        (H_fwd_avg, r"$\langle H \rangle$",         "(c) path entropy",               H_lim),
        (H_rev_avg, r"$\langle \tilde{H} \rangle$", "(d) reverse path cross-entropy", H_lim),
    ]

    # Drawn at the SciPost text width so \includegraphics takes it at scale 1 and the
    # font sizes set above are the ones that reach the page.
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True,
                             figsize=(hlp.SCIPOST_TEXTWIDTH, 0.72*hlp.SCIPOST_TEXTWIDTH))

    for ax, (field, cbar_label, title, clim) in zip(axes.flat, panels):
        vmin, vmax = clim if clim is not None else (None, None)
        im = ax.imshow(field, extent=extent, origin="lower", aspect="auto",
                       cmap="magma", interpolation="nearest", rasterized=True,
                       vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.05)
        # label above the bar rather than rotated alongside it: it reads horizontally
        # and does not add to the width the panel has to give up
        cbar.ax.set_title(cbar_label, pad=4)
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(width=0.6, size=2.5)
        ax.set_title(title, loc="left", pad=4)
        # ticks point out of the panels: inside they sit on top of the dark end of the map
        ax.tick_params(direction="out", top=False, right=False)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"inverse temperature $\beta$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"coupling $J$")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"ising_betaJ_maps.{ext}")
    print(f"Figure saved to {FIG_DIR / 'ising_betaJ_maps.pdf'}.")

    plt.show()
