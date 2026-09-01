import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time

from rdme.mean_field import RDMWilsonCowanBatch, RDMNetworkBatch
import rdme.shared as shrd

# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[2] / "cache"
CACHE_TRAJ_FILE = CACHE_DIR / f"{bname}_traj.pt"
CACHE_SIM_FILE  = CACHE_DIR / f"{bname}_sim.pt"

run_sim   = True
run_plot  = True
overwrite = False

torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # swept parameters
    w_EE = torch.linspace(0, 4, 21)
    w_EI = torch.linspace(0, 4, 21)

    # fixed parameters, kept as in wc-mf_vs_sm.py
    E_ratio = 0.8
    w_IE    = -1.0
    w_II    = -1.0
    I       = 1.0
    beta    = 30.0
    theta   = 1.0
    tau_int = 20.0
    tau_ref = 3.0
    K_ref   = 0.0
    dt      = 0.2

    equi  = 20000
    steps = 10000

    if CACHE_TRAJ_FILE.exists() and not overwrite:
        print(f"Simulation already exists at {CACHE_TRAJ_FILE}. Skipping.")
    else:
        # w_EE and w_EI are passed as vectors, so the batch is their outer product and the
        # trajectory comes back shaped (n_EE, n_EI, T) — no meshgrid/reshape bookkeeping.
        mf = RDMWilsonCowanBatch(E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt,
                                  I, I, beta, beta, theta, theta,
                                  tau_int, tau_int, tau_ref, tau_ref, K_ref, K_ref,
                                  dt=dt, device=device)
        print(f"Running {mf.B} mean-fields. Grid axes: "
              f"{ {k: len(v) for k, v in mf.grid_axes.items()} }.")
        print(f"Batch steps: {equi+steps} per mean-field. ({equi} equilibration + {steps} recording)")
        print(f"Running on: {device}")

        t0 = time.time()
        mf.forward(equi, pb=True)
        traj = mf.entropy_trajectory(steps, pb=True)
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

    # grid_axes holds the vectors as they were passed in (both were pre-scaled by 1/dt)
    dt = traj["dt"].item()
    w_EE, w_EI = mf.grid_axes["w_EE"] * dt, mf.grid_axes["w_EI"] * dt
    extent = (w_EI.min().item(), w_EI.max().item(), w_EE.min().item(), w_EE.max().item())

    # trajectories already come back grid-shaped: (n_EE, n_EI, T)
    m_avg,     m_std     = traj["a_pop"].mean(dim=-1),     traj["a_pop"].std(dim=-1)
    sigma_avg, sigma_std = traj["sigma_tot"].mean(dim=-1), traj["sigma_tot"].std(dim=-1)

    if True: # average / std stats, side by side

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 6))

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
        ax3.set_xlabel('w_EI (E→I Coupling)')
        ax3.set_ylabel('w_EE (Self-Excitation)')
        ax3.grid()

        ax4.set_title('Std of Sigma')
        im4 = ax4.imshow(sigma_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Std of Sigma')
        ax4.set_xlabel('w_EI (E→I Coupling)')
        ax4.grid()

        fig.tight_layout()

    if True: # slices of the grid for a handful of w_EE values

        wEE_indices = shrd.grab_closest_idxs(w_EE, [0.0, 1.0, 2.0, 3.0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Activity vs w_EI for Different w_EE Values')
        for idx in wEE_indices:
            ax.plot(w_EI, m_avg[idx], label=f'w_EE={w_EE[idx].item():.2f}')
            ax.fill_between(w_EI, m_avg[idx] - m_std[idx], m_avg[idx] + m_std[idx], alpha=0.3)
        ax.set_xlabel('w_EI (E→I Coupling)'); ax.set_ylabel('Mean Activity')
        ax.legend(); ax.grid()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Sigma vs w_EI for Different w_EE Values')
        for idx in wEE_indices:
            ax.plot(w_EI, sigma_avg[idx], label=f'w_EE={w_EE[idx].item():.2f}')
            ax.fill_between(w_EI, sigma_avg[idx] - sigma_std[idx], sigma_avg[idx] + sigma_std[idx], alpha=0.3)
        ax.set_xlabel('w_EI (E→I Coupling)'); ax.set_ylabel('Mean Sigma')
        ax.legend(); ax.grid()

    if True: # full (w_EE, w_EI) maps: activity, sigma, forward/reverse entropy

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))

        ax1.set_title('Activity')
        im1 = ax1.imshow(m_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax1, label='Mean Activity')
        ax1.grid()

        ax2.set_title('Sigma')
        im2 = ax2.imshow(sigma_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax2, label='Mean Sigma')
        ax2.grid()

        ax3.set_title('Forward Entropy')
        S_fwd_avg = traj["S_fwd_tot"].mean(dim=-1)
        im3 = ax3.imshow(S_fwd_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im3, ax=ax3, label='Mean Forward Entropy')
        ax3.grid()

        ax4.set_title('Reverse Entropy')
        S_rev_avg = traj["S_rev_tot"].mean(dim=-1)
        im4 = ax4.imshow(S_rev_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Mean Reverse Entropy')
        ax4.set_xlabel('w_EI (E→I Coupling)')
        ax4.grid()

    plt.show()
