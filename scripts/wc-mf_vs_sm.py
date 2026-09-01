import matplotlib.pyplot as plt
import torch
import time

from rdme.spin_model import SpinWilsonCowan
from rdme.mean_field import RDMWilsonCowan

# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

N = 20000

E_ratio = 0.8 # ratio of excitatory neurons

w_EE = 2        # coupling strength; adjust to test different regimes
w_EI = 1        # external field; adjust to test different regimes
w_IE = -1       # external field; adjust to test different regimes
w_II = -1       # external field; adjust to test different regimes

I = 1; I_E = I; I_I = I
beta = 30; beta_E = beta; beta_I = beta
theta = 1; theta_E = theta; theta_I = theta
tau_int = 20; tau_int_E = tau_int; tau_int_I = tau_int
tau_ref = 3; tau_ref_E = tau_ref; tau_ref_I = tau_ref
K_ref = 0; K_ref_E = K_ref; K_ref_I = K_ref

dt = 0.2
steps1 = 10000
steps2 = 2000

# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)

sm = SpinWilsonCowan(N, E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt, 
                    I_E, I_I, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I, 
                    dt=dt, device=device, ic="silent")

mf = RDMWilsonCowan(E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt, 
                    I_E, I_I, beta_E, beta_I, theta_E, theta_I, 
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
    plt.plot(fdists[1], label='Spin Model [I]', linewidth=2)

    plt.plot(mf.p[0].cpu().numpy(), label='Mean Field [E]', linewidth=2)
    plt.plot(mf.p[1].cpu().numpy(), label='Mean Field [I]', linewidth=2)

    plt.xlabel('p(n)'); plt.ylabel('Probability'); plt.legend(); plt.grid()

if True: # equilibration trajectories

    plt.figure(figsize=(10, 4))
    plt.title('Equilibration Trajectories')
    plt.plot(sm_traj1["a_pop"][:, 0], label='Spin Model [E]', linewidth=2)
    plt.plot(sm_traj1["a_pop"][:, 1], label='Spin Model [I]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 0], label='Mean Field [E]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 1], label='Mean Field [I]', linewidth=2)
    plt.xlabel('Time Steps'); plt.ylabel('Mean Activity'); 
    plt.legend(); plt.grid(); plt.tight_layout()


if True: # return map (m_t vs m_{t+1}) of non-transient trajectories

    sm_m = sm_traj2["a_tot"].cpu().numpy()
    mf_m = mf_traj2["a_pop"].cpu().numpy()

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
    ax1.plot(sm_traj2["a_tot"], label='Spin Model', linewidth=2)
    ax1.plot(mf_traj2["a_pop"], label='Mean Field', linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.legend(); ax1.grid()

    # sigma trajectories
    ax2.set_title('Entropy Production Rate')
    ax2.plot(sm_traj2["sigma"], label='Spin Model', linewidth=2)
    ax2.plot(mf_traj2["sigma"], label='Mean Field', linewidth=2)
    ax2.legend(); ax2.grid()

    # S_fwd trajectories
    ax3.set_title('Forward Entropy')
    ax3.plot(sm_traj2["S_fwd"], label='Spin Model', linewidth=2)
    ax3.plot(mf_traj2["S_fwd"], label='Mean Field', linewidth=2)
    ax3.legend(); ax3.grid()

    # S_rev trajectories
    ax4.set_title('Backward Entropy')
    ax4.plot(sm_traj2["S_rev"], label='Spin Model', linewidth=2)
    ax4.plot(mf_traj2["S_rev"], label='Mean Field', linewidth=2)
    ax4.legend(); ax4.grid()

    # print trajetcory averages
    skip = 100 # skip initial and final transient
    print(f"Spin Model: <sigma> = {sm_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <S_fwd> = {sm_traj2['S_fwd'][skip:-skip].mean():.8f}, <S_rev> = {sm_traj2['S_rev'][skip:-skip].mean():.8f}")
    print(f"Mean Field: <sigma> = {mf_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <S_fwd> = {mf_traj2['S_fwd'][skip:-skip].mean():.8f}, <S_rev> = {mf_traj2['S_rev'][skip:-skip].mean():.8f}")

    plt.tight_layout()



plt.show()