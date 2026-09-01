from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

from rdme.mean_field import RDMIsingModelBatch, RDMNetworkBatch
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

    J       = torch.linspace(0, 8, 101)
    I       = torch.linspace(0.5, 2, 101)
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
        # J and I are passed as vectors, so the batch is their outer product and the
        # trajectory comes back shaped (n_J, n_I, T) — no meshgrid/reshape bookkeeping.
        mf = RDMIsingModelBatch(J=J/dt, I=I, beta=beta, theta=theta,
                                 tau_int=tau_int, tau_ref=tau_ref, K_ref=K_ref,
                                 device=device)
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

    # grid_axes holds the vectors as they were passed in (J was pre-scaled by 1/dt)
    dt   = traj["dt"].item()
    J, I = mf.grid_axes["J"] * dt, mf.grid_axes["I"]
    extent = (I.min().item(), I.max().item(), J.min().item(), J.max().item())

    # trajectories already come back grid-shaped: (n_J, n_I, T)
    m_avg,     m_std     = traj["a_pop"].mean(dim=-1),     traj["a_pop"].std(dim=-1)
    sigma_avg, sigma_std = traj["sigma_tot"].mean(dim=-1), traj["sigma_tot"].std(dim=-1)

    if True: # average / std stats, side by side

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 6))

        ax1.set_title('Mean Activity')
        im1 = ax1.imshow(m_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax1, label='Mean Activity')
        ax1.set_ylabel('J (Coupling Strength)')
        ax1.grid()

        ax2.set_title('Std of Activity')
        im2 = ax2.imshow(m_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax2, label='Std of Activity')
        ax2.grid()

        ax3.set_title('Mean Sigma')
        im3 = ax3.imshow(sigma_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im3, ax=ax3, label='Mean Sigma')
        ax3.set_xlabel('I (External Input)')
        ax3.set_ylabel('J (Coupling Strength)')
        ax3.grid()

        ax4.set_title('Std of Sigma')
        im4 = ax4.imshow(sigma_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Std of Sigma')
        ax4.set_xlabel('I (External Input)')
        ax4.grid()

        fig.tight_layout()

    if True: # slices of the grid for a handful of J values

        J_indices = shrd.grab_closest_idxs(J, [0.0, 1.0, 2.0, 3.0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Activity vs I for Different J Values')
        for idx in J_indices:
            ax.plot(I, m_avg[idx], label=f'J={J[idx].item():.2f}')
            ax.fill_between(I, m_avg[idx] - m_std[idx], m_avg[idx] + m_std[idx], alpha=0.3)
        ax.set_xlabel('I (External Input)'); ax.set_ylabel('Mean Activity')
        ax.legend(); ax.grid()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Sigma vs I for Different J Values')
        for idx in J_indices:
            ax.plot(I, sigma_avg[idx], label=f'J={J[idx].item():.2f}')
            ax.fill_between(I, sigma_avg[idx] - sigma_std[idx], sigma_avg[idx] + sigma_std[idx], alpha=0.3)
        ax.set_xlabel('I (External Input)'); ax.set_ylabel('Mean Sigma')
        ax.legend(); ax.grid()

    if True: # full (J, I) maps: activity, sigma, forward/reverse entropy

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
        ax4.set_xlabel('I (External Input)')
        ax4.grid()

    plt.show()
