import matplotlib.pyplot as plt
import torch
import time

import rdme.fixed_points as fpts
from rdme.spin_model import SpinIsingModel
from rdme.mean_field import RDMIsingModel

# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

N = 40000
Q = 100

# integration-based oscillatory regime
J = 4    # coupling strength; adjust to test different regimes
E = 1    # external field; adjust to test different regimes
beta = 30
theta = 1
tau_int = 12
tau_ref = 3
K_ref = 0

# refraction-based oscillatory regime
# J = 4   # coupling strength; adjust to test different regimes
# E = 0   # external field; adjust to test different regimes
# beta = 30
# theta = 0
# tau_int = 12
# tau_ref = 3
# K_ref = 4

steps1 = 5000
steps2 = 10000
dt = 0.2

# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)

sm = SpinIsingModel(N, J/dt, E, beta, theta, 
            tau_int, tau_ref=tau_ref, K_ref=K_ref, 
            dt=dt, device=device, ic="silent")

mf = RDMIsingModel(J/dt, E, beta, theta,
        tau_int, tau_ref, K_ref, dt, eps=0.01, device=device)
print(mf.Qm)

# mf.P = sm.fdist(Q) # initialize mean-field distribution to match spin model

# Main simulation loop
# ~~~~~~~~~~~~~~~~~~~~
times = []
print(f"Running spin model and mean-field simulations for {steps1+steps2} steps on device: {device}...")
times.append(time.time())
print("Running mean-field simulation...")
mf_traj1 = mf.trajectory(T=steps1)
mf_traj2 = mf.entropy_trajectory(T=steps2)
times.append(time.time())
print("Running spin model simulation...")
sm_traj1 = sm.trajectory(T=steps1)
sm_traj2 = sm.entropy_trajectory_chunked(T=steps2)
times.append(time.time())
# show timings
print(f"Markovian mean-field simulation completed in {times[1] - times[0]:.2f} seconds.")
print(f"Markovian spin-model simulation completed in {times[2] - times[1]:.2f} seconds.")
print("All simulations completed successfully.")

# Plotting
# ~~~~~~~~

if True: # visualize final p(n) distribution

    plt.figure(figsize=(10, 4))
    plt.title("Firing age distribution (mean-field vs spin model)")
    plt.plot(sm.fdist(mf.Qm[0]).cpu().numpy(), label='Spin Model', linewidth=2)
    plt.plot(mf.p[0].cpu().numpy(), label='Mean Field', linewidth=2)
    plt.xlabel('p(n)'); plt.ylabel('Probability'); plt.legend(); plt.grid()

if True: # equilibration trajectories

    plt.figure(figsize=(10, 4))
    plt.title('Equilibration Trajectories')
    t = torch.arange(steps1) * dt
    plt.plot(t, sm_traj1["m_tot"], label='Spin Model', linewidth=2)
    plt.plot(t, mf_traj1["m"], label='Mean Field', linewidth=2)
    plt.xlabel('Time (ms)'); plt.ylabel('Mean Activity'); 
    plt.legend(); plt.grid(); plt.tight_layout()


if True: # return map (m_t vs m_{t+1}) of non-transient trajectories

    sm_m = sm_traj2["m_tot"].cpu().numpy()
    mf_m = mf_traj2["m_tot"].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1: plt.Axes; ax2: plt.Axes

    fig.suptitle('Return Map - Transient Trajectories')

    ax1.set_title('Spin Model')
    ax1.scatter(sm_m[:-1], sm_m[1:], s=1, alpha=1)
    ax1.set_xlabel(r'$m_t$'); ax1.set_ylabel(r'$m_{t+1}$')
    ax1.grid()

    ax2.set_title('Mean Field')
    ax2.scatter(mf_m[:-1], mf_m[1:], s=1, alpha=1)
    ax2.set_xlabel(r'$m_t$'); ax2.set_ylabel(r'$m_{t+1}$')
    ax2.grid()

    fig.tight_layout()


if True: # entropy trajectories

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    # ax1: plt.Axes; ax2: plt.Axes; ax3: plt.Axes; ax4: plt.Axes;

    # activity trajectories
    ax1.set_title('Activity')
    t = torch.arange(steps2) * dt
    ax1.plot(t, sm_traj2["m_tot"], label='Spin Model', linewidth=2)
    ax1.plot(t, mf_traj2["m_tot"], label='Mean Field', linewidth=2)
    ax1.set_xlabel('Time (ms)'); ax1.set_ylabel('Mean Activity');
    ax1.legend(); ax1.grid()

    # sigma trajectories
    ax2.set_title('Entropy Production Rate')
    ax2.plot(t, sm_traj2["sigma"], label='Spin Model', linewidth=2)
    ax2.plot(t, mf_traj2["sigma"], label='Mean Field', linewidth=2)
    ax2.set_xlabel('Time (ms)'); ax2.set_ylabel('Entropy Production Rate (nats/ms)');
    ax2.legend(); ax2.grid()

    # H_fwd trajectories
    ax3.set_title('Forward Entropy')
    ax3.plot(t, sm_traj2["H_fwd"], label='Spin Model', linewidth=2)
    ax3.plot(t, mf_traj2["H_fwd"], label='Mean Field', linewidth=2)
    ax3.set_xlabel('Time (ms)'); ax3.set_ylabel('Forward Entropy (nats)');
    ax3.legend(); ax3.grid()

    # H_rev trajectories
    ax4.set_title('Backward Entropy')
    ax4.plot(t, sm_traj2["H_rev"], label='Spin Model', linewidth=2)
    ax4.plot(t, mf_traj2["H_rev"], label='Mean Field', linewidth=2)
    ax4.set_xlabel('Time (ms)'); ax4.set_ylabel('Backward Entropy (nats)');
    ax4.legend(); ax4.grid()

    # print trajetcory averages
    skip = Q # skip initial and final transient
    print(f"Spin Model: <sigma> = {sm_traj2['sigma'][skip:-skip].mean()/dt:.8f} nats/ms, <H_fwd> = {sm_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {sm_traj2['H_rev'][skip:-skip].mean():.8f}")
    print(f"Mean Field: <sigma> = {mf_traj2['sigma'][skip:-skip].mean()/dt:.8f} nats/ms, <H_fwd> = {mf_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {mf_traj2['H_rev'][skip:-skip].mean():.8f}")

    plt.tight_layout()


if True: # fixed-point objective G(m) = m - F(m) over the overlap range

    # G is the self-consistency residual: every zero crossing is a stationary state of the
    # mean field, and its slope there is what the bracketing solver keys on. Plotting it
    # next to the trajectory says whether the dynamics is sitting on a root, orbiting one,
    # or being pushed past the end of the range.
    m_grid = torch.linspace(0, 1, 2001, device=device).unsqueeze(-1)     # (K, 1)
    G      = mf.fp_residual(m_grid).squeeze(-1).cpu()
    roots  = mf.fixed_points(method='bracket', n_grid=2001)
    rho    = torch.tensor([fpts.spectral_radius(mf.fp_eigenvalues(r)) for r in roots])

    m_traj = mf_traj2["m"][:, 0]
    # the roots sit at a few percent activity, so the full range is mostly empty: the
    # second panel is the same curve over the span the roots and the orbit actually occupy
    hi   = max(roots.max().item(), m_traj.max().item()) * 1.4
    zoom = (max(0.0, min(roots.min().item(), m_traj.min().item()) - 0.1 * hi), hi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fixed-point objective  G(m) = m - F(m)')

    for ax, xlim, title in ((ax1, (0.0, 1.0), 'Full range'), (ax2, zoom, 'Around the roots')):
        ax.set_title(title)
        ax.plot(m_grid.squeeze(-1).cpu(), G, linewidth=2, color='C0', label='G(m)')
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.axvspan(m_traj.min(), m_traj.max(), color='C1', alpha=0.15,
                   label='range visited by the mean field')
        for r, rh in zip(roots, rho):
            stable = rh < 1.0
            ax.plot(r.item(), 0.0, marker='o', markersize=8, zorder=3,
                    color='C0' if stable else 'C3',
                    markerfacecolor='C0' if stable else 'none',
                    label=f"m*={r.item():.5f}  |λ|={rh:.4f}  ({'stable' if stable else 'unstable'})")
        ax.set_xlim(*xlim)
        ax.set_xlabel('m'); ax.set_ylabel('G(m)')
        ax.grid(); ax.legend(fontsize=8)
    # the full-range panel is dominated by G ~ m far from the roots; let the zoom set its
    # own y scale from the data inside its window
    inside = (m_grid.squeeze(-1).cpu() >= zoom[0]) & (m_grid.squeeze(-1).cpu() <= zoom[1])
    pad = 0.1 * (G[inside].max() - G[inside].min()).abs().clamp(min=1e-12)
    ax2.set_ylim(G[inside].min() - pad, G[inside].max() + pad)

    fig.tight_layout()



plt.show()