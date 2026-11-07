# ── NOT RUNNABLE (2026-11-07) ─────────────────────────────────────────────────
#
# Ported to the route-B API mechanically and nothing more: `dt0` is gone, the dead hazard keys
# are mapped, and the model builds. It is NOT wired to a preset and has NOT been re-run, because
# the two-population presets it needs (`EI`, `EE-P`) are still `PENDING` in rdme.presets and
# their values are a vault decision that has not been made. Running it produces numbers that
# belong to no named parameter set, so do not.
#
# What the port changed, and why it is a change of MODEL and not a rename:
#   `dt0`         removed. There is no time quantum under route B; `nu0` is the rate unit of
#                 the Poisson reference and `t_abs` the physical dead time.
#   "async"    -> "poisson_exact". The route-B kernel with async's per-bin FORM is "bernoulli"
#                 (nu0*dt*sigma), not this one; the 2026-11-07 brief asks for the solver here
#                 instead, so the rate function changes from saturating to exponential.
#   "sync"/"synchronous"
#              -> "bernoulli", which at nu0*dt = 1 is the sigmoid bit for bit.
#
# ──────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time

import rdme.helpers as hlp
import rdme.plots as rp
from rdme.rdm_batch import RDMNetworkBatch, RDMWilsonCowanBatch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[1] / "cache"
CACHE_TRAJ_FILE = CACHE_DIR / f"{bname}_traj.pt"
CACHE_SIM_FILE  = CACHE_DIR / f"{bname}_sim.pt"

run_sim   = True
run_plot  = True
overwrite = False

# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # swept parameters
    w_EE = torch.linspace(1, 20, 51)
    w_EI = torch.linspace(1, 20, 51)

    # fixed parameters, kept as in wc-mf_vs_sm.py
    E_ratio = 0.8
    w_IE    = -6.0
    w_II    = -1.0
    E       = 1.0
    beta    = 30.0
    theta   = 1.0
    tau_int = 20.0
    tau_ref = 3.0
    K_ref   = 0.0
    dt      = 0.2

    equi  = 20000
    steps = 10000
    dt    = 0.2
    # Was: hazard = "async"  # "async", "sync", "escape"  -- none of those keys exist.
    # "escape" has no route-B counterpart at all; "sync" is "bernoulli" at nu0*dt = 1.
    hazard = "poisson_exact"   # "poisson_exact", "poisson", "bernoulli"
    chunk_size = 256

    if CACHE_TRAJ_FILE.exists() and not overwrite:
        print(f"Simulation already exists at {CACHE_TRAJ_FILE}. Skipping.")
    else:
        # w_EE and w_EI are passed as vectors, so the batch is their outer product and the
        # trajectory comes back shaped (n_EE, n_EI, T) — no meshgrid/reshape bookkeeping.
        # srm takes the weights in physical units -- no /dt, since the input is built
        # from the rate. "synchronous" is "bernoulli" at nu0*dt = 1 under route B.
        mf = RDMWilsonCowanBatch(E_ratio, w_EE, w_EI, w_IE, w_II,
                                  E, E, beta, beta, theta, theta,
                                  tau_int, tau_int, tau_ref, tau_ref, K_ref, K_ref,
                                  dt=dt, device=device, hazard=hazard, eps=0.01)
        print(f"Running {mf.B} mean-fields. Grid axes: "
              f"{ {k: len(v) for k, v in mf.grid_axes.items()} }.")
        print(f"Batch steps: {equi+steps} per mean-field. ({equi} equilibration + {steps} recording)")
        print(f"Running on: {device}")

        t0 = time.time()
        mf.forward(equi, pb=True)
        traj = mf.entropy_trajectory(steps, pb=True, chunk=chunk_size)
        print(f"Done in {time.time() - t0:.2f}s")

        traj["dt"] = torch.tensor(dt)

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(traj, CACHE_TRAJ_FILE)
        print(f"Trajectory saved to {CACHE_TRAJ_FILE}.")
        mf.save(CACHE_SIM_FILE)
        print(f"Simulation state saved to {CACHE_SIM_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    for f in (CACHE_TRAJ_FILE, CACHE_SIM_FILE):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run simulation first.")

    traj = torch.load(CACHE_TRAJ_FILE, weights_only=True)
    mf   = RDMNetworkBatch.load(CACHE_SIM_FILE, device="cpu")

    # grid_axes holds the vectors as they were passed in, now in physical units
    dt = traj["dt"].item()
    w_EE, w_EI = mf.grid_axes["w_EE"], mf.grid_axes["w_EI"]
    extent = (w_EI.min().item(), w_EI.max().item(), w_EE.min().item(), w_EE.max().item())

    # trajectories already come back grid-shaped: (n_EE, n_EI, T)
    m_avg,     m_std     = traj["m_tot"].mean(dim=-1),     traj["m_tot"].std(dim=-1)
    sigma_avg, sigma_std = traj["sigma_tot"].mean(dim=-1), traj["sigma_tot"].std(dim=-1)

    if True: # average / std stats, side by side

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 6))
        fig.set_label('activity-maps')   # the figure's filename slug

        ax1.set_title('Mean Activity')
        im1 = ax1.imshow(m_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax1, label='Mean Activity')
        ax1.set_ylabel('w_EE (Self-Excitation)')
        ax1.grid()

        ax2.set_title('Std of Activity')
        im2 = ax2.imshow(m_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax2, label='Std of Activity')
        ax2.grid()

        ax3.set_title('Mean Sigma')
        im3 = ax3.imshow(sigma_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im3, ax=ax3, label='Mean Sigma')
        ax3.set_xlabel('w_EI (E→E Coupling)')
        ax3.set_ylabel('w_EE (Self-Excitation)')
        ax3.grid()

        ax4.set_title('Std of Sigma')
        im4 = ax4.imshow(sigma_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Std of Sigma')
        ax4.set_xlabel('w_EI (E→E Coupling)')
        ax4.grid()

        fig.tight_layout()

    if True: # slices of the grid for a handful of w_EE values

        wEE_indices = hlp.grab_closest_idxs(w_EE, [1.0, 2.0, 3.0, 4.0, 5.0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Activity vs w_EI for Different w_EE Values')
        for idx in wEE_indices:
            ax.plot(w_EI, m_avg[idx], label=f'w_EE={w_EE[idx].item():.2f}')
            ax.fill_between(w_EI, m_avg[idx] - m_std[idx], m_avg[idx] + m_std[idx], alpha=0.3)
        ax.set_xlabel('w_EI (E→E Coupling)'); ax.set_ylabel('Mean Activity')
        ax.legend(); ax.grid()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Sigma vs w_EI for Different w_EE Values')
        for idx in wEE_indices:
            ax.plot(w_EI, sigma_avg[idx], label=f'w_EE={w_EE[idx].item():.2f}')
            ax.fill_between(w_EI, sigma_avg[idx] - sigma_std[idx], sigma_avg[idx] + sigma_std[idx], alpha=0.3)
        ax.set_xlabel('w_EI (E→E Coupling)'); ax.set_ylabel('Mean Sigma')
        ax.legend(); ax.grid()

    if True: # slices of the grid for a handful of w_EI values

        wEI_indices = hlp.grab_closest_idxs(w_EI, [1.0, 2.0, 3.0, 4.0, 5.0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Activity vs w_EE for Different w_EI Values')
        for idx in wEI_indices:
            ax.plot(w_EE, m_avg[:, idx], label=f'w_EI={w_EI[idx].item():.2f}')
            ax.fill_between(w_EE, m_avg[:, idx] - m_std[:, idx], m_avg[:, idx] + m_std[:, idx], alpha=0.3)
        ax.set_xlabel('w_EE (Self-Excitation)'); ax.set_ylabel('Mean Activity')
        ax.legend(); ax.grid()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Sigma vs w_EE for Different w_EI Values')
        for idx in wEI_indices:
            ax.plot(w_EE, sigma_avg[:, idx], label=f'w_EI={w_EI[idx].item():.2f}')
            ax.fill_between(w_EE, sigma_avg[:, idx] - sigma_std[:, idx], sigma_avg[:, idx] + sigma_std[:, idx], alpha=0.3)
        ax.set_xlabel('w_EE (Self-Excitation)'); ax.set_ylabel('Mean Sigma')
        ax.legend(); ax.grid()

    # if True: # full (w_EE, w_EI) maps: activity, sigma, forward/reverse entropy

    #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

    #     ax1.set_title('Activity')
    #     im1 = ax1.imshow(m_avg, extent=extent, origin='lower', aspect='auto')
    #     fig.colorbar(im1, ax=ax1, label='Mean Activity')
    #     ax1.grid()

    #     ax2.set_title('Sigma')
    #     im2 = ax2.imshow(sigma_avg, extent=extent, origin='lower', aspect='auto')
    #     fig.colorbar(im2, ax=ax2, label='Mean Sigma')
    #     ax2.grid()

    #     ax3.set_title('Forward Entropy')
    #     H_fwd_avg = traj["H_fwd_tot"].mean(dim=-1)
    #     im3 = ax3.imshow(H_fwd_avg, extent=extent, origin='lower', aspect='auto')
    #     fig.colorbar(im3, ax=ax3, label='Mean Forward Entropy')
    #     ax3.grid()

    #     ax4.set_title('Reverse Entropy')
    #     H_rev_avg = traj["H_rev_tot"].mean(dim=-1)
    #     im4 = ax4.imshow(H_rev_avg, extent=extent, origin='lower', aspect='auto')
    #     fig.colorbar(im4, ax=ax4, label='Mean Reverse Entropy')
    #     ax4.set_xlabel('w_EI (E→E Coupling)')
    #     ax4.grid()

    rp.show()
