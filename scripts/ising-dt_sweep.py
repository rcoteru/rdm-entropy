import matplotlib.pyplot as plt
import torch
import time

from rdme.single import RDMIsingModel
from rdme.spin_model import SpinIsingModel


# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~
#
# Sweep the integration step dt and see how the final (non-transient) averages
# of activity and of the entropy-production quantities change. J is rescaled as
# J/dt (as in ising-mf_vs_sm.py) so the per-ms coupling stays fixed; the number
# of steps is chosen so every run covers the same simulated time in ms.
#
# The hazard is the logistic sigmoid(h), which fixes the firing probability per
# *bin* rather than per unit time: the escape rate it implies is softplus(h)/dt, so
# both the activity and the EPR climb as dt shrinks rather than converging. That is
# what this sweep measures. (A rate-based hazard removes the drift, but the escape
# extension is not in the codebase at present.)

device = 'cpu'
torch.set_default_dtype(torch.float64)

N = 20000

J = 4      # coupling strength (per ms, before the 1/dt rescaling)
E = 1      # external field
beta = 30
theta = 1

tau_int = 12
tau_ref = 3
K_ref = 0

dt_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]   # integration step (ms)

equi_ms = 500   # equilibration time (ms)
meas_ms = 2000  # measurement window for the entropy trajectory (ms)
skip_ms = 10    # transient trimmed from each end of the measurement window (ms)

run_spin_model = True   # the spin model is the slow part; flip off for a quick look

# Sweep
# ~~~~~
results = {
    "dt": [],
    "mf": {"a": [], "sigma": [], "H_fwd": [], "H_rev": []},
    "sm": {"a": [], "sigma": [], "H_fwd": [], "H_rev": []},
}

for dt in dt_values:

    steps_equi = int(round(equi_ms / dt))
    steps_meas = int(round(meas_ms / dt))
    skip = max(1, int(round(skip_ms / dt)))

    print(f"\ndt = {dt}: equi {steps_equi} steps, measure {steps_meas} steps, "
          f"skip {skip} steps/end")

    # --- mean field ---
    t0 = time.time()
    # hazard="synchronous" is the sigmoid of the original model. Note srm takes J in
    # physical units: the 1/dt the old call site applied is now internal, since the input is
    # built from the rate rather than the per-bin overlap.
    mf = RDMIsingModel(J, E, beta, theta,
                       tau_int, tau_ref, K_ref, dt, hazard="synchronous",
                       eps=0.01, device=device)
    mf.forward(steps_equi)
    mf_traj = mf.entropy_trajectory(T=steps_meas)
    print(f"  mean field done in {time.time() - t0:.1f}s")

    results["dt"].append(dt)
    results["mf"]["a"].append(mf_traj["m_tot"][skip:-skip].mean().item()/dt)
    results["mf"]["sigma"].append(mf_traj["sigma_tot"][skip:-skip].mean().item()/dt)
    results["mf"]["H_fwd"].append(mf_traj["H_fwd_tot"][skip:-skip].mean().item()/dt)
    results["mf"]["H_rev"].append(mf_traj["H_rev_tot"][skip:-skip].mean().item()/dt)

    # --- spin model ---
    if run_spin_model:
        t0 = time.time()
        sm = SpinIsingModel(N, J / dt, E, beta, theta,
                            tau_int, tau_ref=tau_ref, K_ref=K_ref,
                            dt=dt, device=device, ic="silent")
        sm.trajectory(T=steps_equi)
        sm_traj = sm.entropy_trajectory_chunked(T=steps_meas)
        print(f"  spin model done in {time.time() - t0:.1f}s")

        results["sm"]["a"].append(sm_traj["m_tot"][skip:-skip].mean().item()/dt)
        results["sm"]["sigma"].append(sm_traj["sigma_tot"][skip:-skip].mean().item()/dt)
        results["sm"]["H_fwd"].append(sm_traj["H_fwd_tot"][skip:-skip].mean().item()/dt)
        results["sm"]["H_rev"].append(sm_traj["H_rev_tot"][skip:-skip].mean().item()/dt)

torch.save(results, "ising_dt_sweep_results.pt")

# Report
# ~~~~~~
print("\n dt      <a>_mf     <sigma>_mf   <a>_sm     <sigma>_sm")
for i, dt in enumerate(results["dt"]):
    line = f"{dt:5.2f}  {results['mf']['a'][i]:9.6f}  {results['mf']['sigma'][i]:11.8f}"
    if run_spin_model:
        line += f"  {results['sm']['a'][i]:9.6f}  {results['sm']['sigma'][i]:11.8f}"
    print(line)

# Plotting
# ~~~~~~~~
dts = results["dt"]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
fig.suptitle(f'Final averages vs dt  (J={J}, E={E}, beta={beta})')

for ax, key, title in [
    (ax1, "a",     'Mean Activity'),
    (ax2, "sigma", 'Entropy Production Rate  <sigma>  (nats/ms)'),
    (ax3, "H_fwd", 'Forward Entropy  <H_fwd>'),
    (ax4, "H_rev", 'Backward Entropy  <H_rev>'),
]:
    ax.set_title(title)
    ax.plot(dts, results["mf"][key], 'o-', label='Mean Field', linewidth=2)
    if run_spin_model:
        ax.plot(dts, results["sm"][key], 's-', label='Spin Model', linewidth=2)
    ax.legend(); ax.grid()

ax4.set_xlabel('dt (ms)')
plt.tight_layout()
plt.show()
