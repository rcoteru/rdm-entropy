""" Mean field vs spin network for one single-population (Ising) system: equilibration,
age distribution, return map, entropy trajectories and the fixed-point objective G(m). """

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp

from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

# Base parameter sets, written out in full. A regime is one of these plus overrides.
S1     = dict(E=0.0, beta=6.0,  theta=0.5, tau_int=15.0, tau_ref=15.0, K_ref=15.0, nu0=1.0, t_abs=1.0)
S1_SAT = dict(E=0.0, beta=8.0,  theta=0.5, tau_int=10.0, tau_ref=25.0, K_ref=25.0, nu0=1.0, t_abs=1.0, J=40.0)
D2     = dict(E=1.0, beta=40.0, theta=0.5, tau_int=15.0, tau_ref=1.0,  K_ref=0.0,  nu0=1.0, t_abs=1.0, J=6.0)

# dt must divide t_abs. Optional "t_equi"/"t_meas" (ms) override the defaults below.
REGIMES: dict[str, dict] = {
    "S1_fp":                S1 | dict(J=10.0, dt=0.5),
    "S1_cycle":             S1 | dict(J=17.0, dt=0.1),
    "ref_lowbeta":          S1 | dict(tau_int=20.0, J=30.0, dt=0.5),
    "noref_lowbeta":        S1_SAT | dict(tau_int=15.0, K_ref=0.0, J=20.0, dt=0.5),
    "ref_lowbeta_strong":   S1_SAT | dict(dt=0.5),
    "noref_driven_strong":  D2 | dict(dt=0.1),
    "nu0_reanchored_fp":    S1 | dict(nu0=0.01, J=10.0, dt=1.0),
    "nu0_reanchored_cycle": S1 | dict(nu0=0.01, J=80.0, dt=1.0),
}

regime = "S1_cycle"       # pick one of the above
HAZARD = "poisson_exact"  # "poisson_exact" | "poisson" | "bernoulli"
MF_IC  = "quiet"         # "active": start MF just above the highest root; "quiet": all age
                          # mass in the oldest bin (the constructor's default)

TOL    = 1e-2      # age-grid truncation tolerance (sets Q)
N      = 40000     # spins in the network model
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

T_EQUI, T_MEAS = 5000.0, 2500.0   # ms: equilibration, then entropy recording
SKIP_MS        = 10.0             # ms trimmed from each end of the recording before averaging

RUN_KEYS = ("t_equi", "t_meas")   # regime entries that are the run, not the model
PARAMS: dict = {k: v for k, v in REGIMES[regime].items() if k not in RUN_KEYS} | {"hazard": HAZARD}

T_EQUI = REGIMES[regime].get("t_equi", T_EQUI)
T_MEAS = REGIMES[regime].get("t_meas", T_MEAS)
DT     = PARAMS["dt"]
STEPS  = (hlp.n_steps(T_EQUI, DT, name="T_EQUI"), hlp.n_steps(T_MEAS, DT, name="T_MEAS"))
SKIP   = hlp.n_steps(SKIP_MS, DT, name="SKIP_MS")

TITLE = (f'{regime}  ·  hazard={HAZARD}  ·  dt={DT:g}  —  '
         + ',  '.join(f'{k}={PARAMS[k]:g}' for k in
                      ("J", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "nu0", "t_abs")))


# ── Paths & switches ──────────────────────────────────────────────────────────

# the parameters name the file, so a changed parameter never reads a stale cache
CACHE = hlp.cache_path(Path(__file__).stem, dict(PARAMS, N=N, t_equi=T_EQUI, t_meas=T_MEAS,
                                                tol=TOL, mf_ic=MF_IC),
                       tag=f"{regime}-{MF_IC}", root=Path(__file__).parents[1] / "cache")

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    steps1, steps2 = STEPS

    sm = SpinIsingModel(N, **PARAMS, device=DEVICE, ic="silent")
    mf = RDMIsingModel(**PARAMS, tol=TOL, device=DEVICE)
    roots = mf.fixed_points(method='bracket', n_grid=2001)
    if MF_IC == "active":
        if roots.shape[0] == 0:
            raise RuntimeError("MF_IC='active' needs a fixed point to start from; none found")
        mf.set_state(roots[-1] * 1.001)
    elif MF_IC != "quiet":
        raise ValueError(f"Unknown MF_IC {MF_IC!r}; use 'quiet' or 'active'")
    print(f"Q = {mf.Q} age bins. Running {T_EQUI + T_MEAS:g} ms of each model "
          f"({steps1 + steps2} steps at dt={DT:g}) on {DEVICE}.")

    t0 = time.time()
    mf_traj1 = mf.trajectory(T=steps1)
    mf_traj2 = mf.entropy_trajectory(T=steps2)
    t1 = time.time()
    sm_traj1 = sm.trajectory(T=steps1)
    sm_traj2 = sm.entropy_trajectory_chunked(T=steps2, overlap=1024)
    t2 = time.time()
    print(f"Mean field: {t1 - t0:.2f}s. Spin model: {t2 - t1:.2f}s.")

    # G(m) = m - F(m): every zero is a stationary state of the mean field
    m_grid = torch.linspace(0, 1, 2001, device=DEVICE, dtype=torch.float64).unsqueeze(-1)  # (K, 1)
    out = {
        "params": PARAMS, "regime": regime, "t_equi": T_EQUI, "t_meas": T_MEAS,
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
                  x=torch.arange(len(d["p_mf"]), dtype=torch.float64) * DT,
                  title="Firing age distribution (mean-field vs spin model)",
                  xlabel='age u (ms)', ylabel='p(u)')

    if True:  # equilibration trajectories
        rp.curves({'Spin Model': sm_traj1["m_tot"], 'Mean Field': mf_traj1["m"]},
                  dt=DT, title=f'Equilibration trajectories\n{TITLE}')

    if True:  # return map (m_t vs m_{t+1}) of the non-transient trajectories
        rp.return_maps({'Spin Model': sm_traj2["m_tot"], 'Mean Field': mf_traj2["m_tot"]},
                       suptitle='Return Map - Non-transient Trajectories')

    if True:  # entropy trajectories
        rp.panels({'Spin Model': sm_traj2, 'Mean Field': mf_traj2},
                  ["m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot"], dt=DT,
                  suptitle='Entropy trajectories')

        for name, traj in (("Spin Model", sm_traj2), ("Mean Field", mf_traj2)):
            print(f"{name} (trimmed {SKIP_MS:g} ms each end): "
                  f"<sigma> = {traj['sigma'][SKIP:-SKIP].mean():.8f} nats/bin, "
                  f"<H_fwd> = {traj['H_fwd'][SKIP:-SKIP].mean():.8f}, "
                  f"<H_rev> = {traj['H_rev'][SKIP:-SKIP].mean():.8f}")

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
