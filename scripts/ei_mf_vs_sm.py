""" Mean field vs spin network for one two-population (E-I) system: the conventions table per
population, then rate and entropy production folded over the cycle, the last stretch of both,
sigma against r and the age distributions (plots.mf_vs_sm), and the attractor in the
(r_E, r_I) plane against the mean field's fixed points. The E-I counterpart of e1_mf_vs_sm.py. """

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from scipy.sparse.linalg import ArpackNoConvergence

import rdme.analysis as an
import rdme.helpers as hlp
import rdme.plots as rp
from rdme.mean_field import RDMWilsonCowan
from rdme.spin_model import SpinWilsonCowan

# ── Parameters ────────────────────────────────────────────────────────────────

# Schwalger, Deger & Gerstner 2017, Fig. 5 neuron, identical for E and I: LIF with exponential
# escape noise, Delta_u = 2.5 mV (beta = 0.4 /mV), u_th = 15 mV, tau_m = 20 ms, t_ref = 4 ms,
# c = 10 Hz; no relative refractoriness (K_ref = 0, so tau_ref is unused)
NEURON = dict(beta_E=0.4, beta_I=0.4, theta_E=15.0, theta_I=15.0,
              tau_int_E=20.0, tau_int_I=20.0, tau_ref_E=1.0, tau_ref_I=1.0,
              K_ref_E=0.0, K_ref_I=0.0, nu0=0.01, t_abs=4.0)

# model points. Weights in mV ms, physical (not divided by dt), on the rate in 1/ms.
# NAMING: w_XY is X onto Y -- RDMWilsonCowan / SpinWilsonCowan store [[w_EE, w_IE],
# [w_EI, w_II]] with rows = targets -- so every weight below carries its arrow.
POINTS = {
    # Schwalger 2017 Fig. 5 as published. Identical neurons and equal rows: both populations
    # get the same input, so the mean field collapses onto one population with
    # J = 960 - 1200 = -240 (ising_mf_vs_sm.py, "I1-fig5-collapsed")
    "EI-fig5": dict(NEURON,
                    w_EE=960.0,     # E→E
                    w_EI=960.0,     # E→I
                    w_IE=-1200.0,   # I→E
                    w_II=-1200.0,   # I→I
                    E_exc=24.0, E_inh=24.0),
    # E-I presets along w(E→I) through Fig. 5. Neuron, w(E→E) = 960, w(I→E) = -1200,
    # w(I→I) = -1200 and E_exc = E_inh = 24 mV all as published; only w(E→I) moves. On this
    # cut the mean field has one Hopf, at J_H = 885.98 (19.91 Hz; r_E 21.75 / r_I 18.69 Hz),
    # unstable BELOW J_H, and the cycle grows continuously from it, with the silent start and
    # a start at the fixed point agreeing everywhere (no bistability; ei_bifurcations.py and a
    # mean-field attractor map, 2026-11-19). The onset regime is narrow -- r_E max passes the
    # 1/t_abs = 250 Hz ceiling near 0.94 J_H -- so the presets are placed by behaviour, not
    # by fixed fractions of J_H. r_max is dt-resolved down to ~0.97 J_H (dt = 0.2/0.1/0.05).
    "EI-onset": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=875.0,     # E→I; 0.988 J_H: small cycle, r_E max ~62 Hz
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
    "EI-cycle": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=865.0,     # E→I; 0.976 J_H: cycle, r_E max ~100 Hz, still resolved
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
    "EI-burst": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=800.0,     # E→I; 0.903 J_H: bursts past the ceiling, r_max NOT
                                     # resolved in dt (428/678/1022 Hz at dt 0.2/0.1/0.05)
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
}
# the async preset is Fig. 5 itself: w(E→I) = 960 = 1.084 J_H, stable (mu = -0.0365 /ms)
POINTS["EI-async"] = POINTS["EI-fig5"]
NAME  = "EI-cycle"      # choose one of the above points
POINT = POINTS[NAME]

N       = 20000     # spins in the network model
E_RATIO = 0.8       # excitatory fraction; in the mean field it only weights the aggregates
hazard  = "poisson_exact"   # "poisson_exact" | "poisson" | "bernoulli"
IC      = "silent"          # "silent" | "random", the same start for both models
AGE_MAX = 25.0              # ms, age window of the "random" start

# numerical parameters
dt     = 0.1       # ms, solver step; must divide t_abs
TOL    = 1e-2      # truncation tolerance: sets the mean field's Q and the spin EPR's lookahead
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

T_EQUI, T_MEAS = 1000.0, 2000.0  # ms: equilibration, then entropy recording
STORE_SPIKES = False   # keep the spin raster of the recording, for the spin ISI CV;
                       # int8 (T_MEAS/dt, N): 400 MB at the defaults

run_sim   = True
run_plot  = True
overwrite = False   # the hash covers every parameter,
                    # so this is only for a change to the CODE


# ── Paths, dicts & switches ──────────────────────────────────────────────────

PARAMS: dict = dict(POINT, E_ratio=E_RATIO, dt=dt, hazard=hazard)
STEPS  = (hlp.n_steps(T_EQUI, dt, name="T_EQUI"), hlp.n_steps(T_MEAS, dt, name="T_MEAS"))
T_ABS  = POINT["t_abs"]
POPS   = ("E", "I")

TITLE = (f'{NAME}  ·  hazard={hazard}  ·  dt={dt:g}  ·  ic={IC}  —  '
         f'w(E→E)={POINT["w_EE"]:g}, w(E→I)={POINT["w_EI"]:g}, w(I→E)={POINT["w_IE"]:g}, '
         f'w(I→I)={POINT["w_II"]:g}, E_exc={POINT["E_exc"]:g}, E_inh={POINT["E_inh"]:g}')

# the parameters name the file, so a changed parameter never reads a stale cache
CACHE = hlp.cache_path(Path(__file__).stem,
                       dict(PARAMS, N=N, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                            ic=IC, age_max=AGE_MAX,
                            # only when on, so a default run still finds its existing cache
                            **({"spikes": True} if STORE_SPIKES else {})),
                       tag=f"{NAME}-{IC}", root=Path(__file__).parents[1] / "cache")


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    steps1, steps2 = STEPS

    print("Initializing models.")
    sm = SpinWilsonCowan(N, **PARAMS, ic=IC, age_max=AGE_MAX, tol=TOL, device=DEVICE)
    mf = RDMWilsonCowan(**PARAMS, ic=IC, age_max=AGE_MAX, tol=TOL, device=DEVICE)

    # fixed points (Newton from the physical box, plus the exact 1-D starts when w(I→I) = 0),
    # and the leading mode of each one's Jacobian: ARPACK, or the growth rate alone by power
    # iteration where the leading modes are too degenerate for it (growth rate mu < 0 is stable)
    roots = mf.fixed_points()
    mu, omega = torch.zeros(len(roots), dtype=torch.float64), torch.zeros(len(roots),
                                                                          dtype=torch.float64)
    for i, r in enumerate(roots):
        try:
            o = mf.leading_modes(r, k=2)
            mu[i], omega[i] = o["mu"][0], o["omega"][0]
        except ArpackNoConvergence:
            mu[i], omega[i] = mf.leading_growth(r), float("nan")

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

    out = {
        "params": PARAMS, "ic": IC, "t_equi": T_EQUI, "t_meas": T_MEAS,
        "steps": STEPS, "N": N, "Q": mf.Q,
        "mf_traj1": mf_traj1, "mf_traj2": mf_traj2,
        "sm_traj1": sm_traj1, "sm_traj2": sm_traj2,
        "p_mf": mf.p.cpu(),                 # (2, Q) age distributions, on the mean field's grid
        "phi_mf": mf.firing_prob().cpu(),   # (2, Q) final hazards, for the stationary ISI density
        "p_sm": sm.fdist(mf.Q).cpu(),
        "roots": roots.cpu(),               # (R, 2) overlaps
        "mu": mu,                           # (R,) leading growth rate of each, 1/ms
        "omega": omega,                     # (R,) and its angular frequency, rad/ms
    }
    hlp.save_atomic(out, CACHE)
    print(f"Saved to {CACHE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

def rows(traj: dict, a: int, b: int) -> dict:
    """ rate_and_epr of E, I and the aggregate over steps [a, b). """
    pop = lambda k: {"m_tot": traj["m"][a:b, k], "sigma_tot": traj["sigma"][a:b, k]}
    tot = {"m_tot": traj["m_tot"][a:b], "sigma_tot": traj["sigma_tot"][a:b]}
    return {"E": an.rate_and_epr(pop(0), dt, T_ABS), "I": an.rate_and_epr(pop(1), dt, T_ABS),
            "tot": an.rate_and_epr(tot, dt, T_ABS)}


if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"Plotting {CACHE.name}.")
    d = torch.load(CACHE, weights_only=False)
    mf_traj2, sm_traj2 = d["mf_traj2"], d["sm_traj2"]
    r_mf = mf_traj2["m"] / dt * 1e3       # (T, 2), Hz
    r_sm = sm_traj2["m"] / dt * 1e3
    s_mf = mf_traj2["sigma"] / dt * 1e3   # (T, 2), nats per neuron per s
    s_sm = sm_traj2["sigma"] / dt * 1e3

    if True:  # conventions table: the mean field over whole cycles of r_E, spins over all
        cyc = an.rate_and_epr({"m_tot": mf_traj2["m"][:, 0],
                               "sigma_tot": mf_traj2["sigma"][:, 0]}, dt, T_ABS, cycles=True)
        a, b, n = an.cycle_window(cyc["r"])
        out = {"Mean Field": rows(mf_traj2, a, b),
               "Spin Model": rows(sm_traj2, 0, len(r_sm))}

        # ISI CV: the mean field's from the stationary ISI density of its final hazard, only
        # when settled; the spins' from recorded ISIs, only if the spikes were stored
        settled = cyc["state"] == "settled"
        N_E = int(E_RATIO * N)
        cv = {"Mean Field": {p: an.isi_cv(d["phi_mf"][k], dt) if settled else None
                             for k, p in enumerate(POPS)},
              "Spin Model": {p: an.isi_cv_spikes(s, dt) if s is not None else None
                             for p, s in zip(POPS, (sm_traj2["s"][:, :N_E],
                                                    sm_traj2["s"][:, N_E:])
                                             if "s" in sm_traj2 else (None, None))}}
        if cyc["state"] == "settling":
            cycles = f"settling ({cyc['amp_ratio']:.3f})"
        else:
            cycles = str(cyc["n_cycles"])

        print(f"\n{TITLE}")
        print(f"mean field from r_E: {cycles} cycles ({cyc['state']})")
        print(f"\n{'model':<11} {'pop':>4} {'r (Hz)':>9} {'r range (Hz)':>20} "
              f"{'σ (nats/s)':>12} {'σ/r (nats/spike)':>17} {'r·t_abs':>8} {'ISI CV':>7}")
        for k, o in out.items():
            for p, q in o.items():
                c = cv[k].get(p)
                rng = f"[{q['r_min']:.3f}, {q['r_max']:.3f}]"
                print(f"{k:<11} {p:>4} {q['r_mean']:9.3f} {rng:>20} {q['sigma_mean']:12.6g} "
                      f"{q['sigma_per_spike']:17.6g} {q['dead_frac']:8.4f} "
                      f"{'n/a' if c is None else f'{c:.4f}':>7}")

        # period against mean ISI (= 1/r_mean) per population
        per_mf = (b - a) * dt / n if n else float("nan")
        print("\nperiod vs mean ISI (1/r_mean), ms:")
        for k, o in out.items():
            for j, p in enumerate(POPS):
                per = (per_mf if k == "Mean Field" or not n
                       else an.noisy_cycles(r_sm[:, j].double(), dt, per_mf)[1])
                how = ("whole cycles of r_E" if k == "Mean Field"
                       else "crossings of the smoothed rate")
                print(f"  {k:<11} {p}: period {per:9.3f} ({how})   mean ISI "
                      f"{1e3 / o[p]['r_mean']:9.3f}")

        print("\nmean-field fixed points (leading mode of the Jacobian):")
        for r, u, w in zip(d["roots"], d["mu"], d["omega"]):
            print(f"  r_E {r[0] / dt * 1e3:9.4f}  r_I {r[1] / dt * 1e3:9.4f} Hz   μ "
                  f"{u:+.6f} /ms  f {abs(w) / (2 * math.pi) * 1e3:7.3f} Hz  "
                  f"({'stable' if u < 0 else 'unstable'})")

    if True:  # cycle, last stretch, sigma against r, age distributions
        # E and I overlaid, the aggregate (E_RATIO-weighted, as m_tot) in its own column
        w = torch.tensor([E_RATIO, 1 - E_RATIO], dtype=torch.float64)
        tot = lambda x, x_tot: torch.cat([x.double(), x_tot.double()[:, None] / dt * 1e3], 1)
        ages = lambda p: torch.cat([p.double(), (w @ p.double())[None]])
        rp.mf_vs_sm(tot(r_mf, mf_traj2["m_tot"]), tot(s_mf, mf_traj2["sigma_tot"]),
                    tot(r_sm, sm_traj2["m_tot"]), tot(s_sm, sm_traj2["sigma_tot"]),
                    ages(d["p_mf"]), ages(d["p_sm"]), dt, pops=POPS + ("tot",),
                    panels=(POPS, ("tot",)),
                    fixed_points=[(torch.cat([r, (w @ r.double())[None]]) / dt * 1e3,
                                   bool(u < 0)) for r, u in zip(d["roots"], d["mu"])],
                    title=TITLE)

    if True:  # attractor in the (r_E, r_I) plane, with the mean field's fixed points
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(r_sm[:, 0], r_sm[:, 1], '.', color='C0', alpha=0.15, markersize=2,
                label='Spin Model')
        ax.plot(r_mf[:, 0], r_mf[:, 1], color='C1', linewidth=1.5, label='Mean Field')
        for r, u in zip(d["roots"], d["mu"]):
            stable = u < 0
            ax.plot(r[0] / dt * 1e3, r[1] / dt * 1e3, marker='o', markersize=9, zorder=3,
                    color='k' if stable else 'C3', markerfacecolor='k' if stable else 'none',
                    linestyle='none',
                    label=f"fixed point μ={u:+.5f} /ms "
                          f"({'stable' if stable else 'unstable'})")
        ax.set_xlabel('r_E (Hz)')
        ax.set_ylabel('r_I (Hz)')
        ax.set_title(f'Attractor in the (r_E, r_I) plane\n{TITLE}', fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()

    rp.show()
