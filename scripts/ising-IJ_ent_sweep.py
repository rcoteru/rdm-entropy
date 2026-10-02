from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
from rdme.batch import RDMIsingModelBatch, RDMNetworkBatch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[1] / "cache"
CACHE_TRAJ_FILE = CACHE_DIR / f"{bname}_traj.pt"
CACHE_SIM_FILE  = CACHE_DIR / f"{bname}_sim.pt"

run_sim   = True
run_plot  = True
overwrite = False
torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    J       = torch.linspace(0, 3, 101)
    E       = torch.linspace(0.5, 1.5, 101)
    beta    = 30.0
    theta   = 1.0
    tau_int = 20.0
    tau_ref = 3.0
    K_ref   = 0.0
    dt      = 0.2

    equi  = 20000
    steps = 10000
    chunk_size = 1024

    if CACHE_TRAJ_FILE.exists() and not overwrite:
        print(f"Simulation already exists at {CACHE_TRAJ_FILE}. Skipping.")
    else:
        # J and E are passed as vectors, so the batch is their outer product and the
        # trajectory comes back shaped (n_J, n_E, T) — no meshgrid/reshape bookkeeping.
        # NOTE: dt is NOT passed here, so the model runs at a step of 1.0 while J carries a
        # 1/dt and dt = 0.2 is used for the axis labels below -- the kernels are therefore in
        # units of steps, not ms. That predates this port and is preserved exactly: J/dt with
        # dt defaulting to 1.0 reproduces the previous numbers bit for bit. Passing J and
        # dt=dt instead would be the consistent version, and would change the results.
        mf = RDMIsingModelBatch(J=J/dt, E=E, beta=beta, theta=theta,
                                tau_int=tau_int, tau_ref=tau_ref, K_ref=K_ref,
                                device=device, hazard="synchronous")
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

    # grid_axes holds the vectors as they were passed in (J was pre-scaled by 1/dt)
    dt   = traj["dt"].item()
    J, E = mf.grid_axes["J"] * dt, mf.grid_axes["E"]
    extent = (E.min().item(), E.max().item(), J.min().item(), J.max().item())

    # trajectories already come back grid-shaped: (n_J, n_E, T)
    m_avg,     m_std     = traj["m_tot"].mean(dim=-1),     traj["m_tot"].std(dim=-1)
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
        ax3.set_xlabel('E (External Input)')
        ax3.set_ylabel('J (Coupling Strength)')
        ax3.grid()

        ax4.set_title('Std of Sigma')
        im4 = ax4.imshow(sigma_std, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Std of Sigma')
        ax4.set_xlabel('E (External Input)')
        ax4.grid()

        fig.tight_layout()

    if True: # slices of the grid for a handful of J values

        J_indices = hlp.grab_closest_idxs(J, [0.0, 1.0, 2.0, 3.0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Activity vs E for Different J Values')
        for idx in J_indices:
            ax.plot(E, m_avg[idx], label=f'J={J[idx].item():.2f}')
            ax.fill_between(E, m_avg[idx] - m_std[idx], m_avg[idx] + m_std[idx], alpha=0.3)
        ax.set_xlabel('E (External Input)'); ax.set_ylabel('Mean Activity')
        ax.legend(); ax.grid()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Sigma vs E for Different J Values')
        for idx in J_indices:
            ax.plot(E, sigma_avg[idx], label=f'J={J[idx].item():.2f}')
            ax.fill_between(E, sigma_avg[idx] - sigma_std[idx], sigma_avg[idx] + sigma_std[idx], alpha=0.3)
        ax.set_xlabel('E (External Input)'); ax.set_ylabel('Mean Sigma')
        ax.legend(); ax.grid()

    if True: # full (J, E) maps: activity, sigma, forward/reverse entropy

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
        H_fwd_avg = traj["H_fwd_tot"].mean(dim=-1)
        im3 = ax3.imshow(H_fwd_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im3, ax=ax3, label='Mean Forward Entropy')
        ax3.grid()

        ax4.set_title('Reverse Entropy')
        H_rev_avg = traj["H_rev_tot"].mean(dim=-1)
        im4 = ax4.imshow(H_rev_avg, extent=extent, origin='lower', aspect='auto')
        fig.colorbar(im4, ax=ax4, label='Mean Reverse Entropy')
        ax4.set_xlabel('E (External Input)')
        ax4.grid()

    plt.show()
