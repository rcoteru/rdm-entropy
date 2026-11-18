""" Mean field vs spin network for one single-population (Ising) system: equilibration,
age distribution, return map, entropy trajectories and the fixed-point objective G(m). """

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.analysis as an
import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel

# ── Parameters ────────────────────────────────────────────────────────────────

N       = 20000     # spins in the network model
J       = 30.0      # recurrent coupling, physical units (not divided by dt)
E       = 0.0       # external drive
beta    = 6.0       # gain
theta   = 0.5       # threshold
nu0     = 1.0       # 1/ms, Poisson reference rate
t_abs   = 1.0       # ms, absolute refractory period
tau_int = 15.0      # ms, synaptic integration
tau_ref = 20.0      # ms, relative refractory decay
K_ref   = 15.0      # relative refractory amplitude

hazard  = "poisson_exact"   # "poisson_exact" | "poisson" | "bernoulli"
IC      = "silent"          # "silent" | "random", the same start for both models
AGE_MAX = 25.0              # ms, age window of the "random" start

# numerical parameters
dt     = 0.1       # ms, solver step; must divide t_abs
TOL    = 1e-2      # truncation tolerance: sets the mean field's Q and the spin EPR's lookahead
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

T_EQUI, T_MEAS = 5000.0, 2500.0   # ms: equilibration, then entropy recording

run_sim   = True
run_plot  = True
overwrite = False   # the hash covers every parameter, 
                    # so this is only for a change to the CODE


# ── Paths, dicts & switches ──────────────────────────────────────────────────

PARAMS: dict = dict(J=J, E=E, beta=beta, theta=theta, tau_int=tau_int, tau_ref=tau_ref, K_ref=K_ref,
                      nu0=nu0, t_abs=t_abs, dt=dt, hazard=hazard)
STEPS  = (hlp.n_steps(T_EQUI, dt, name="T_EQUI"), hlp.n_steps(T_MEAS, dt, name="T_MEAS"))

TITLE = (f'hazard={hazard}  ·  dt={dt:g}  ·  ic={IC}  —  '
         + ',  '.join(f'{k}={PARAMS[k]:g}' for k in
                      ("J", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "nu0", "t_abs")))

# the parameters name the file, so a changed parameter never reads a stale cache
CACHE = hlp.cache_path(Path(__file__).stem,
                       dict(PARAMS, N=N, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                            ic=IC, age_max=AGE_MAX),
                       tag=f"J{J:g}-{IC}", root=Path(__file__).parents[1] / "cache")


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    steps1, steps2 = STEPS

    print("Initializing models.")
    sm = SpinIsingModel(N, **PARAMS, ic=IC, age_max=AGE_MAX, tol=TOL, device=DEVICE)
    mf = RDMIsingModel(**PARAMS, ic=IC, age_max=AGE_MAX, tol=TOL, device=DEVICE)
    roots = mf.fixed_points(method='bracket', n_grid=2001)
    print(f"Running {T_EQUI + T_MEAS:g} ms of each model "
          f"({steps1 + steps2} steps at dt={dt:g}) on {DEVICE}.")
    print(f"Q = {mf.Q} age bins.")
    t0 = time.time()
    print("Running the mean field trajectories.")
    mf_traj1 = mf.trajectory(T=steps1)
    mf_traj2 = mf.entropy_trajectory(T=steps2)
    t1 = time.time()
    print("Running the spin model trajectories.")
    sm_traj1 = sm.trajectory(T=steps1)
    sm_traj2 = sm.entropy_trajectory(T=steps2)
    t2 = time.time()
    print(f"Mean field: {t1 - t0:.2f}s. Spin model: {t2 - t1:.2f}s.")

    # G(m) = m - F(m): every zero is a stationary state of the mean field
    m_grid = torch.linspace(0, 1, 2001, device=DEVICE, dtype=torch.float64).unsqueeze(-1)  # (K, 1)
    out = {
        "params": PARAMS, "ic": IC, "t_equi": T_EQUI, "t_meas": T_MEAS,
        "steps": STEPS, "N": N, "Q": mf.Q,
        "mf_traj1": mf_traj1, "mf_traj2": mf_traj2,
        "sm_traj1": sm_traj1, "sm_traj2": sm_traj2,
        "p_mf": mf.p[0].cpu(),               # age distributions, on the mean field's grid
        "p_sm": sm.fdist(mf.Q)[0].cpu(),
        "m_grid": m_grid.squeeze(-1).cpu(),
        "G": mf.fp_residual(m_grid).squeeze(-1).cpu(),
        "roots": roots.cpu(),
        "rho": torch.tensor([fpts.spectral_radius(mf.fp_eigenvalues(r)) for r in roots],
                            dtype=torch.float64),
    }
    hlp.save_atomic(out, CACHE)
    print(f"Saved to {CACHE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"Plotting {CACHE.name}.")
    d = torch.load(CACHE, weights_only=False)

    mf_traj1, mf_traj2 = d["mf_traj1"], d["mf_traj2"]
    sm_traj1, sm_traj2 = d["sm_traj1"], d["sm_traj2"]

    if True:  # final age distribution, on the physical age axis
        rp.curves({'Spin Model': d["p_sm"], 'Mean Field': d["p_mf"]},
                  x=torch.arange(len(d["p_mf"]), dtype=torch.float64) * dt,
                  title="Firing age distribution (mean-field vs spin model)",
                  xlabel='age u (ms)', ylabel='p(u)')

    if True:  # equilibration trajectories
        rp.curves({'Spin Model': sm_traj1["m_tot"], 'Mean Field': mf_traj1["m"]},
                  dt=dt, title=f'Equilibration trajectories\n{TITLE}')

    if True:  # return map (m_t vs m_{t+1}) of the non-transient trajectories
        rp.return_maps({'Spin Model': sm_traj2["m_tot"], 'Mean Field': mf_traj2["m_tot"]},
                       suptitle='Return Map - Non-transient Trajectories')

    if True:  # entropy trajectories
        rp.panels({'Spin Model': sm_traj2, 'Mean Field': mf_traj2},
                  ["m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot"], dt=dt,
                  suptitle='Entropy trajectories')

    if True:  # standard outputs: rate and EPR in physical units
        out = {"Spin Model": an.rate_and_epr(sm_traj2, dt, t_abs),
               "Mean Field": an.rate_and_epr(mf_traj2, dt, t_abs, cycles=True)}

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        fig.suptitle(TITLE, fontsize=9)
        rp.curves({k: o["r"] for k, o in out.items()}, dt=dt, ax=ax1, ylabel='r (Hz)',
                  title='Firing rate')
        rp.curves({k: o["sigma"] for k, o in out.items()}, dt=dt, ax=ax2,
                  ylabel='σ (nats / neuron / s)', title='Entropy production (spin model: '
                  'sampled per step, unsmoothed)')
        for ax in (ax1, ax2):
            ax.grid()
        fig.tight_layout()

        print(f"\n{'model':<11} {'r (Hz)':>9} {'r_max':>9} {'σ (nats/s)':>12} "
              f"{'σ/r (nats/spike)':>17} {'r_max·t_abs':>12} {'cycles':>7}")
        for k, o in out.items():
            print(f"{k:<11} {o['r_mean']:9.3f} {o['r_max']:9.3f} "
                  f"{o['sigma_mean']:12.6g} {o['sigma_per_spike']:17.6g} {o['dead_frac']:12.4f} "
                  f"{'n/a' if o['n_cycles'] is None else o['n_cycles']:>7}")

    if True:  # fixed-point objective G(m) = m - F(m)
        m_grid, G, roots, rho = d["m_grid"], d["G"], d["roots"], d["rho"]
        m_traj = mf_traj2["m"][:, 0]

        # the roots sit at a few percent activity: the second panel zooms onto them
        hi   = max(roots.max().item(), m_traj.max().item()) * 1.4
        zoom = (max(0.0, min(roots.min().item(), m_traj.min().item()) - 0.1 * hi), hi)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Fixed-point objective  G(m) = m - F(m)')

        for ax, xlim, title in ((ax1, (0.0, 1.0), 'Full range'), (ax2, zoom, 'Around the roots')):
            rp.curves({'G(m)': G}, x=m_grid, ax=ax, title=title, xlabel='m', ylabel='G(m)',
                      color='C0')
            ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
            ax.axvspan(m_traj.min(), m_traj.max(), color='C1', alpha=0.15,
                       label='range visited by the mean field')
            for r, rh in zip(roots, rho):
                stable = rh < 1.0
                ax.plot(r.item(), 0.0, marker='o', markersize=8, zorder=3,
                        color='C0' if stable else 'C3',
                        markerfacecolor='C0' if stable else 'none',
                        label=f"m*={r.item():.5f}  |λ|={rh:.4f}  "
                              f"({'stable' if stable else 'unstable'})")
            ax.set_xlim(*xlim)
            ax.legend(fontsize=8)

        inside = (m_grid >= zoom[0]) & (m_grid <= zoom[1])
        if not inside.any():
            inside[((m_grid - sum(zoom) / 2).abs()).argmin()] = True
        pad = 0.1 * (G[inside].max() - G[inside].min()).abs().clamp(min=1e-12)
        ax2.set_ylim(G[inside].min() - pad, G[inside].max() + pad)
        fig.tight_layout()

    rp.show()
