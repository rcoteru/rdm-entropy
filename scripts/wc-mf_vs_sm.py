import matplotlib.pyplot as plt
import torch
import time

import rdme.fixed_points as fpts
from rdme.spin_model import SpinWilsonCowan
from rdme.mean_field import RDMWilsonCowan, fp_guess_lattice

# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

N = 50000

E_ratio = 0.8 # ratio of excitatory neurons

w_EE = 5.0      # coupling strength; adjust to test different regimes
w_EI = 3.0      # external field; adjust to test different regimes
w_IE = -6.0       # external field; adjust to test different regimes
w_II = -0.5       # external field; adjust to test different regimes

E = 1; 
E_exc = E 
E_inh = E
beta = 30; beta_E = beta; beta_I = beta
theta = 1; theta_E = theta; theta_I = theta
tau_int = 20; tau_int_E = tau_int; tau_int_I = tau_int
tau_ref = 3; tau_ref_E = tau_ref; tau_ref_I = tau_ref
K_ref = 0; K_ref_E = K_ref; K_ref_I = K_ref

dt = 0.2
steps1 = 20000
steps2 = 10000

# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)

sm = SpinWilsonCowan(N, E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I, 
                    dt=dt, device=device, ic="silent")

mf = RDMWilsonCowan(E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I,
                    dt=dt, eps=0.01, device=device)

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


    fdists = sm.fdists(max(mf.Qm)).cpu().numpy()
    plt.plot(fdists[0], label='Spin Model [E]', linewidth=2)
    plt.plot(fdists[1], label='Spin Model [E]', linewidth=2)

    plt.plot(mf.p[0].cpu().numpy(), label='Mean Field [E]', linewidth=2)
    plt.plot(mf.p[1].cpu().numpy(), label='Mean Field [E]', linewidth=2)

    plt.xlabel('p(n)'); plt.ylabel('Probability'); plt.legend(); plt.grid()

if True: # equilibration trajectories

    plt.figure(figsize=(10, 4))
    plt.title('Equilibration Trajectories')
    plt.plot(sm_traj1["m"][:, 0], label='Spin Model [E]', linewidth=2)
    plt.plot(sm_traj1["m"][:, 1], label='Spin Model [E]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 0], label='Mean Field [E]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 1], label='Mean Field [E]', linewidth=2)
    plt.xlabel('Time Steps'); plt.ylabel('Mean Activity'); 
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


if True: # fixed-point objective |G(m)| over the (m_E, m_I) square

    # With M=2 the objective is a vector field on [0,1]^2, so what is plotted is its
    # magnitude, and the two zero contours -- the nullclines G_E = 0 and G_I = 0 -- are
    # drawn on top: the fixed points are exactly where they cross, which is also the check
    # that the Newton solver did not miss one. The orbit is overlaid to show what the
    # dynamics does with them.
    n_grid = 201

    roots = mf.fixed_points(method='newton', guesses=fp_guess_lattice(2, 15, device=device))
    rho   = torch.tensor([fpts.spectral_radius(mf.fp_eigenvalues(r)) for r in roots])
    orbit = mf_traj2["m"]                                                   # (T, 2)

    # everything interesting sits within a few percent of the origin, so the second panel
    # covers only the span the roots and the orbit occupy -- on its own grid, since
    # subsetting the full-range one would leave a handful of cells across the whole window
    hi     = max(roots.max().item(), orbit.max().item()) * 1.4
    windows = [((0.0, 1.0), 'Full range'), ((-0.05 * hi, hi), 'Around the roots')]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Fixed-point objective  |G(m)| = |m - F(m)|')

    for ax, ((lo, up), title) in zip(axes, windows):
        axis   = torch.linspace(lo, up, n_grid, device=device)
        mE, mI = torch.meshgrid(axis, axis, indexing='ij')
        pts    = torch.stack([mE.reshape(-1), mI.reshape(-1)], dim=-1)      # (K, 2)
        # chunked: each residual evaluation materialises a (chunk, Q) field per population
        G  = torch.cat([mf.fp_residual(c) for c in pts.split(4096)]).cpu()  # (K, 2)
        # .T throughout because G is indexed [m_E, m_I] and matplotlib wants [row=y, col=x]
        GE = G[:, 0].reshape(n_grid, n_grid).T
        GI = G[:, 1].reshape(n_grid, n_grid).T
        Gn = G.norm(dim=-1).reshape(n_grid, n_grid).T
        a  = axis.cpu()

        ax.set_title(title)
        pcm = ax.pcolormesh(a, a, torch.log10(Gn + 1e-12), shading='auto', cmap='viridis')
        ax.contour(a, a, GE, levels=[0.0], colors='w', linewidths=1.5)
        ax.contour(a, a, GI, levels=[0.0], colors='w', linewidths=1.5, linestyles='--')
        ax.plot(orbit[:, 0], orbit[:, 1], color='C1', linewidth=0.8, alpha=0.8)
        for r, rh in zip(roots, rho):
            ax.plot(r[0].item(), r[1].item(), marker='o', markersize=9, zorder=3, color='r',
                    markerfacecolor='r' if rh < 1.0 else 'none',
                    label=f"m*=({r[0]:.4f}, {r[1]:.4f})  |λ|={rh:.4f}")
        ax.set_xlim(lo, up); ax.set_ylim(lo, up)
        ax.set_xlabel(r'$m_E$'); ax.set_ylabel(r'$m_I$')
        # one colorbar per panel: the two windows span very different ranges of |G|, and a
        # single shared bar would be labelled with only one of the two norms
        fig.colorbar(pcm, ax=ax, label=r'$\log_{10}|G(m)|$')

    # empty handles so the legend can name the contours and the orbit too; filled marker =
    # stable root, open = unstable, same convention as everywhere else
    for ax in axes:
        ax.plot([], [], color='w', linewidth=1.5, label=r'$G_E = 0$')
        ax.plot([], [], color='w', linewidth=1.5, linestyle='--', label=r'$G_I = 0$')
        ax.plot([], [], color='C1', linewidth=0.8, label='mean-field orbit')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)

    fig.tight_layout()


if True: # entropy trajectories

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    # ax1: plt.Axes; ax2: plt.Axes; ax3: plt.Axes; ax4: plt.Axes;

    # activity trajectories
    ax1.set_title('Activity')
    ax1.plot(sm_traj2["m_tot"], label='Spin Model', linewidth=2)
    ax1.plot(mf_traj2["m_tot"], label='Mean Field', linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.legend(); ax1.grid()

    # sigma trajectories
    ax2.set_title('Entropy Production Rate')
    ax2.plot(sm_traj2["sigma"], label='Spin Model', linewidth=2)
    ax2.plot(mf_traj2["sigma"], label='Mean Field', linewidth=2)
    ax2.legend(); ax2.grid()

    # H_fwd trajectories
    ax3.set_title('Forward Entropy')
    ax3.plot(sm_traj2["H_fwd"], label='Spin Model', linewidth=2)
    ax3.plot(mf_traj2["H_fwd"], label='Mean Field', linewidth=2)
    ax3.legend(); ax3.grid()

    # H_rev trajectories
    ax4.set_title('Backward Entropy')
    ax4.plot(sm_traj2["H_rev"], label='Spin Model', linewidth=2)
    ax4.plot(mf_traj2["H_rev"], label='Mean Field', linewidth=2)
    ax4.legend(); ax4.grid()

    # print trajetcory averages
    skip = 100 # skip initial and final transient
    print(f"Spin Model: <sigma> = {sm_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <H_fwd> = {sm_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {sm_traj2['H_rev'][skip:-skip].mean():.8f}")
    print(f"Mean Field: <sigma> = {mf_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <H_fwd> = {mf_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {mf_traj2['H_rev'][skip:-skip].mean():.8f}")

    plt.tight_layout()



plt.show()