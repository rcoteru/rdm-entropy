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

# model points: J physical (not divided by dt), nu0 in 1/ms (Poisson reference rate),
# t_abs / tau_int / tau_ref in ms (absolute window, synaptic integration, relative decay)
POINTS = {
    "default": dict(J=15.0, E=0.0, beta=6.0, theta=0.5, 
                    tau_int=15.0, tau_ref=20.0, K_ref=15.0,
                    nu0=1.0, t_abs=1.0),
    # Schwalger, Deger & Gerstner 2017, Fig. 3: uncoupled LIF with exponential escape noise
    # (c = 10 Hz, Delta_u = 2 mV, u_th = 15 mV, tau_m = 20 ms, t_ref = 4 ms); paper reports
    # 6.6 Hz / CV 0.66 at E = 15 mV and 37 Hz / CV 0.2 at E = 30 mV
    "fig3-E15": dict(J=0.0, E=15.0, beta=0.5, theta=15.0, 
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    "fig3-E30": dict(J=0.0, E=30.0, beta=0.5, theta=15.0,
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    # Single excitatory population. Base neuron = Schwalger 2017 Table 1 / Fig. 3 (above);
    # only J (mV ms, on the rate in 1/ms) and E change. J is set from the mean field's Hopf
    # point J_H (leading complex pair on the unit circle, poisson_exact, dt = 0.1); no fold
    # below the half-ceiling r t_abs = 0.5 at either drive. No resolved cycle preset: at 1.25,
    # 1.5 and 1.75 J_H the mean field's sigma moves by 9-21 % from dt = 0.1 to 0.05.
    # E = 15 mV: J_H = 276.64
    "E1-15-async": dict(J=138.32, E=15.0, beta=0.5, theta=15.0,      # 0.5 J_H
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    "E1-15-onset": dict(J=304.31, E=15.0, beta=0.5, theta=15.0,      # 1.1 J_H
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    "E1-15-strong": dict(J=553.29, E=15.0, beta=0.5, theta=15.0,     # 2 J_H; burst
                    # unresolved at dt = 0.1
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    # E = 30 mV: J_H = 139.34
    "E1-30-async": dict(J=69.67, E=30.0, beta=0.5, theta=15.0,       # 0.5 J_H
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    "E1-30-onset": dict(J=153.27, E=30.0, beta=0.5, theta=15.0,      # 1.1 J_H
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
    "E1-30-strong": dict(J=278.68, E=30.0, beta=0.5, theta=15.0,     # 2 J_H; burst
                    # unresolved at dt = 0.1
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
}
POINT = POINTS["E1-15-onset"]  # choose one of the above points

hazard  = "poisson_exact"   # "poisson_exact" | "poisson" | "bernoulli"
IC      = "silent"          # "silent" | "random", the same start for both models
AGE_MAX = 25.0              # ms, age window of the "random" start

# numerical parameters
dt     = 0.1       # ms, solver step; must divide t_abs
TOL    = 1e-2      # truncation tolerance: sets the mean field's Q and the spin EPR's lookahead
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

T_EQUI, T_MEAS = 5000.0, 3000.0  # ms: equilibration, then entropy recording; T_MEAS holds
                                 # >= 30 periods on the oscillating presets (~70 ms at E = 15)
STORE_SPIKES = False   # keep the spin raster of the recording, for the spin ISI CV;
                       # int8 (T_MEAS/dt, N): 600 MB at the defaults

run_sim   = True
run_plot  = True
overwrite = False   # the hash covers every parameter, 
                    # so this is only for a change to the CODE


# ── Paths, dicts & switches ──────────────────────────────────────────────────

PARAMS: dict = dict(POINT, dt=dt, hazard=hazard)
STEPS  = (hlp.n_steps(T_EQUI, dt, name="T_EQUI"), hlp.n_steps(T_MEAS, dt, name="T_MEAS"))

TITLE = (f'hazard={hazard}  ·  dt={dt:g}  ·  ic={IC}  —  '
         + ',  '.join(f'{k}={PARAMS[k]:g}' for k in
                      ("J", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "nu0", "t_abs")))

# the parameters name the file, so a changed parameter never reads a stale cache
CACHE = hlp.cache_path(Path(__file__).stem,
                       dict(PARAMS, N=N, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                            ic=IC, age_max=AGE_MAX,
                            # only when on, so a default run still finds its existing cache
                            **({"spikes": True} if STORE_SPIKES else {})),
                       tag=f"J{POINT['J']:g}-{IC}", root=Path(__file__).parents[1] / "cache")


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
    sm_traj2 = sm.entropy_trajectory(T=steps2, s=STORE_SPIKES)
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
        "phi_mf": mf.firing_prob()[0].cpu(), # final hazard, for the stationary ISI density
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
        out = {"Spin Model": an.rate_and_epr(sm_traj2, dt, POINT["t_abs"]),
               "Mean Field": an.rate_and_epr(mf_traj2, dt, POINT["t_abs"], cycles=True)}

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

        # ISI CV: the mean field's from the stationary ISI density of its final hazard, only
        # when settled (on a cycle or while ringing that hazard is not an ISI distribution);
        # the spins' from recorded ISIs, only if the spikes were stored
        mf_settled = out["Mean Field"]["state"] == "settled"
        cv = {"Spin Model": an.isi_cv_spikes(sm_traj2["s"], dt) if "s" in sm_traj2 else None,
              "Mean Field": an.isi_cv(d["phi_mf"], dt) if mf_settled and "phi_mf" in d
                            else None}

        def cycles_col(o):
            if o["state"] is None:
                return "n/a"
            if o["state"] == "settling":
                return f"settling ({o['amp_ratio']:.3f})"
            return str(o["n_cycles"])

        print(f"\n{'model':<11} {'r (Hz)':>9} {'r range (Hz)':>20} {'σ (nats/s)':>12} "
              f"{'σ/r (nats/spike)':>17} {'r·t_abs':>8} {'ISI CV':>7} {'cycles':>7}")
        for k, o in out.items():
            rng = f"[{o['r_min']:.3f}, {o['r_max']:.3f}]"
            print(f"{k:<11} {o['r_mean']:9.3f} {rng:>20} "
                  f"{o['sigma_mean']:12.6g} {o['sigma_per_spike']:17.6g} {o['dead_frac']:8.4f} "
                  f"{'n/a' if cv[k] is None else f'{cv[k]:.4f}':>7} {cycles_col(o):>7}")

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
