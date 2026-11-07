from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre

from rdme.rdm_single import RDMIsingModel
from rdme.spin_model import SpinIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

# A regime is a PRESET plus the overrides that make it a system, which is what presets.py
# asks of every run. Before the route-B port these were four inline dicts with no names; the
# check, done rather than assumed:
#
#   ref_lowbeta          = S1      + tau_int = 20, J = 30
#   noref_lowbeta        = S1-sat  + tau_int = 15, K_ref = 0, J = 20
#   ref_lowbeta_strong   = S1-sat  EXACTLY (J = 40 included) -- it is the preset, not a variant
#   noref_driven_strong  = no preset at all; it is now PRESETS["D2"] (provisional name)
#
# The old comment above this block claimed the first two "differ in K_ref and in NOTHING else".
# They do not: they differ in beta, tau_int, tau_ref, K_ref and J. That claim belonged to an
# earlier pairing and is removed rather than repaired, because the pair it describes does not
# exist in this file. A genuine K_ref contrast is S1-sat against S1-sat with K_ref = 0, and
# nothing here has been re-run to say what that contrast now looks like under route B.
#
# Each regime is cached under its own name, so switching back and forth costs nothing after
# the first run of each.
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. The pre-port dicts carried a MIXTURE -- two "async"
                           # (route A, = "bernoulli" here) and two "poisson" -- which is the
                           # one thing CLAUDE.md forbids inside a comparison. One key for all
                           # four, and it is the solver.

REGIMES: dict[str, dict] = {
    # The refractory arm and its low-K_ref partner. dt = 0.5, not coarser: at dt = 1 the
    # K_ref = 0 arm does not sit at a fixed point, it flips the whole population on and off on
    # alternate steps (acf1 = -1.000, the two-step artifact) for every J >= 15. Refining halves
    # it away. Both arms use the same dt, since a contrast whose arms differ in the solver is
    # not a contrast. The age grid is NOT set here: it is fitted at run time, and the two arms
    # genuinely need different ones because the control fires far faster and so needs far less
    # age depth.
    #
    # The rates quoted below (K_ref = 25 at 0.0073/ms, CV 3.81; K_ref = 0 at 0.598/ms, CV 0)
    # were measured under route A and are NOT what this now runs.
    "ref_lowbeta":        dict(preset="S1",     tau_int=20.0, J=30.0, dt=0.5),
    "noref_lowbeta":      dict(preset="S1-sat", tau_int=15.0, K_ref=0.0, J=20.0, dt=0.5),

    # Same mechanism driven harder: a cleaner limit cycle (CV 5.45 under route A) at the cost
    # of a coupling further from anything defensible. This one IS S1-sat, with no override.
    "ref_lowbeta_strong": dict(preset="S1-sat", dt=0.5),

    # Driven, no refraction, beta = 40: the stiff one, which is why its step is 0.1.
    "noref_driven_strong": dict(preset="D2",    dt=0.1),
}

regime = "ref_lowbeta"  # pick one of the above regimes to run
regime = "noref_driven_strong"

PRESET = REGIMES[regime]["preset"]

TOL    = 1e-2      # age-grid accuracy; Qm is fitted to it rather than set by hand
N      = 40000     # spins in the network model
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

# Durations in MILLISECONDS; the step counts follow the regime's dt (helpers.n_steps). The
# regimes run at different dt, so a shared step count would have meant a different physical
# window in each -- these are the windows each regime was actually run for, written down.
# A regime may override them with "t_equi"/"t_meas", which are run settings and are stripped
# out before the rest of the dict is handed to the model.
T_EQUI, T_MEAS = 5000.0, 2500.0   # ms: equilibration, then entropy recording
SKIP_MS        = 10.0             # ms trimmed from each end of the recording before averaging

RUN_KEYS = ("t_equi", "t_meas", "preset")   # regime entries that are the run, not the model

PARAMS: dict = pre.params(PRESET, hazard=HAZARD,
                          **{k: v for k, v in REGIMES[regime].items() if k not in RUN_KEYS})

T_EQUI = REGIMES[regime].get("t_equi", T_EQUI)
T_MEAS = REGIMES[regime].get("t_meas", T_MEAS)
STEPS  = (hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI"),
          hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS"))
SKIP   = hlp.n_steps(SKIP_MS, PARAMS["dt"], name="SKIP_MS")


# ── Paths & switches ──────────────────────────────────────────────────────────

# The parameters name the file, so a changed parameter cannot be read back from a cache that
# predates it and there is nothing to invalidate by hand. Switching regimes, or back to a
# parameter set already run, is free. The cost is one file per set in cache/.
#
# TOL is in the key but the fitted Qm is not, because Qm is a solver setting rather than a
# model parameter: two runs at the same physics and the same accuracy are the same run,
# whatever bin count each turned out to need. A change to the FITTING code is a code change,
# which the cache does not track and clearing cache/ handles.
CACHE = hlp.cache_path(Path(__file__).stem, dict(PARAMS, preset=PRESET, N=N, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                       tag=regime, root=Path(__file__).parents[1] / "cache")

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    steps1, steps2 = STEPS
    mf_args = {k: v for k, v in PARAMS.items() if k != "J"}

    # The mean field's age grid is fitted, not declared: q_from_tau is the model's default and
    # is the wrong rule at these rates, and the right Q differs by 2x between the two arms.
    Qm = hlp.fit_age_grid(lambda qm: RDMIsingModel(PARAMS["J"], **mf_args, Qm=qm, eps=0.01,
                                                   device=DEVICE), tol=TOL, equi=steps1 + steps2)

    # the spin network carries an age per neuron and has no age grid, so Qm is the one thing
    # the two models do not share
    sm = SpinIsingModel(N, PARAMS["J"], **mf_args, device=DEVICE, ic="random")
    mf = RDMIsingModel(PARAMS["J"], **mf_args, Qm=Qm, eps=0.01, device=DEVICE)
    print(f"Q = {mf.Qm} age bins. Running {T_EQUI + T_MEAS:g} ms of each model "
          f"({steps1 + steps2} steps at dt={PARAMS['dt']:g}) on {DEVICE}.")

    t0 = time.time()
    mf_traj1 = mf.trajectory(T=steps1)
    mf_traj2 = mf.entropy_trajectory(T=steps2)
    t1 = time.time()
    sm_traj1 = sm.trajectory(T=steps1)
    sm_traj2 = sm.entropy_trajectory_chunked(T=steps2)
    t2 = time.time()
    print(f"Mean field: {t1 - t0:.2f}s. Spin model: {t2 - t1:.2f}s.")

    # G is the self-consistency residual: every zero crossing is a stationary state of the
    # mean field, and its slope there is what the bracketing solver keys on. Evaluated here
    # rather than at plot time because it needs the live model, which the cache does not hold.
    m_grid = torch.linspace(0, 1, 2001, device=DEVICE).unsqueeze(-1)     # (K, 1)
    roots  = mf.fixed_points(method='bracket', n_grid=2001)

    out = {
        "params": PARAMS, "t_equi": T_EQUI, "t_meas": T_MEAS, "steps": STEPS,
        "N": N, "preset": PRESET,
        "mf_traj1": mf_traj1, "mf_traj2": mf_traj2,
        "sm_traj1": sm_traj1, "sm_traj2": sm_traj2,
        # the two age distributions, on the mean field's grid
        "p_mf": mf.p[0].cpu(),
        "p_sm": sm.fdist(mf.Qm[0]).cpu(),
        "m_grid": m_grid.squeeze(-1).cpu(),
        "G": mf.fp_residual(m_grid).squeeze(-1).cpu(),
        "roots": roots.cpu(),
        "rho": torch.tensor([fpts.spectral_radius(mf.fp_eigenvalues(r)) for r in roots]),
    }

    CACHE.parent.mkdir(exist_ok=True)
    torch.save(out, CACHE)
    print(f"Saved to {CACHE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    # No parameter check here: the filename is the parameters, so this file cannot be anything
    # but the current run. The copy of PARAMS inside it is provenance, for reading a cache
    # found later without the script that wrote it.
    print(f"Plotting {CACHE.name}.")
    d = torch.load(CACHE, weights_only=False)

    dt = PARAMS["dt"]
    steps1, steps2 = STEPS
    mf_traj1, mf_traj2 = d["mf_traj1"], d["mf_traj2"]
    sm_traj1, sm_traj2 = d["sm_traj1"], d["sm_traj2"]

    if True:  # final p(n) distribution
        # the age axis in ms (u = n*dt), not in bins: the bin index is a property of the
        # solver, and the two models are only comparable on the physical age
        rp.curves({'Spin Model': d["p_sm"], 'Mean Field': d["p_mf"]},
                  x=torch.arange(len(d["p_mf"])) * dt,
                  title="Firing age distribution (mean-field vs spin model)",
                  xlabel='age u (ms)', ylabel='p(u)')

    if True:  # equilibration trajectories
        rp.curves({'Spin Model': sm_traj1["m_tot"], 'Mean Field': mf_traj1["m"]},
                  dt=dt, title=f'Equilibration trajectories\n{pre.banner(PRESET, PARAMS)}')

    if True:  # return map (m_t vs m_{t+1}) of the non-transient trajectories
        rp.return_maps({'Spin Model': sm_traj2["m_tot"], 'Mean Field': mf_traj2["m_tot"]},
                       suptitle='Return Map - Non-transient Trajectories')

    if True:  # entropy trajectories
        rp.panels({'Spin Model': sm_traj2, 'Mean Field': mf_traj2},
                  ["m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot"], dt=dt,
                  suptitle='Entropy trajectories')

        for name, traj in (("Spin Model", sm_traj2), ("Mean Field", mf_traj2)):
            print(f"{name} (trimmed {SKIP_MS:g} ms each end): "
                  f"<sigma> = {traj['sigma'][SKIP:-SKIP].mean():.8f} nats, "
                  f"<H_fwd> = {traj['H_fwd'][SKIP:-SKIP].mean():.8f}, "
                  f"<H_rev> = {traj['H_rev'][SKIP:-SKIP].mean():.8f}")

    if True:  # fixed-point objective G(m) = m - F(m) over the overlap range
        # Plotting G next to the trajectory says whether the dynamics is sitting on a root,
        # orbiting one, or being pushed past the end of the range. Custom rather than a
        # plots.py call: the roots and the visited band are annotations nothing else draws.
        m_grid, G, roots, rho = d["m_grid"], d["G"], d["roots"], d["rho"]
        m_traj = mf_traj2["m"][:, 0]

        # the roots sit at a few percent activity, so the full range is mostly empty: the
        # second panel is the same curve over the span the roots and the orbit actually occupy
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

        # the full-range panel is dominated by G ~ m far from the roots; let the zoom set its
        # own y scale from the data inside its window
        inside = (m_grid >= zoom[0]) & (m_grid <= zoom[1])
        pad = 0.1 * (G[inside].max() - G[inside].min()).abs().clamp(min=1e-12)
        ax2.set_ylim(G[inside].min() - pad, G[inside].max() + pad)
        fig.tight_layout()

    rp.show()
