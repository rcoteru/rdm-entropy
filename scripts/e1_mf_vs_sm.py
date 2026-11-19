""" Mean field vs spin network for one single-population (Ising) system: the conventions table
and the fixed points, then rate and entropy production folded over the cycle, the last stretch
of both, sigma against r, and the age distribution (plots.mf_vs_sm). """

import math
import time
from pathlib import Path

import torch

import rdme.analysis as an
import rdme.helpers as hlp
import rdme.plots as rp
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel

# ── Parameters ────────────────────────────────────────────────────────────────


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
    # Schwalger 2017 Fig. 5 (wc_mf_vs_sm.py, "EI-fig5") collapsed onto one population: its
    # neuron (Delta_u = 2.5 mV) with J = w(E→E) + w(I→E) = 960 - 1200, which is what both
    # populations receive there since the neurons are identical and the rows equal
    "I1-fig5-collapsed": dict(J=-240.0, E=24.0, beta=0.4, theta=15.0,
                    tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                    nu0=0.01, t_abs=4.0),
}
POINT = POINTS["E1-15-onset"]  # choose one of the above points

N       = 20000             # spins in the network model
hazard  = "poisson_exact"   # "poisson_exact" | "poisson" | "bernoulli"
IC      = "silent"          # "silent" | "random", the same start for both models
AGE_MAX = 25.0              # ms, age window of the "random" start

# numerical parameters
dt     = 0.1       # ms, solver step; must divide t_abs
TOL    = 1e-2      # truncation tolerance: sets the mean field's Q and the spin EPR's lookahead
DEVICE = 'cpu'     # 'cuda' if torch.cuda.is_available() else 'cpu'

T_EQUI, T_MEAS = 1000.0, 2000.0  # ms: equilibration, then entropy recording; T_MEAS holds
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

    out = {
        "params": PARAMS, "ic": IC, "t_equi": T_EQUI, "t_meas": T_MEAS,
        "steps": STEPS, "N": N, "Q": mf.Q,
        "mf_traj1": mf_traj1, "mf_traj2": mf_traj2,
        "sm_traj1": sm_traj1, "sm_traj2": sm_traj2,
        "p_mf": mf.p[0].cpu(),               # age distributions, on the mean field's grid
        "phi_mf": mf.firing_prob()[0].cpu(), # final hazard, for the stationary ISI density
        "p_sm": sm.fdist(mf.Q)[0].cpu(),
        "roots": roots.cpu(),
        # largest eigenvalue modulus of the one-step map, from its leading growth rate
        "rho": torch.tensor([math.exp(float(mf.leading_modes(r, k=2)["mu"][0]) * dt)
                             for r in roots], dtype=torch.float64),
    }
    hlp.save_atomic(out, CACHE)
    print(f"Saved to {CACHE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"Plotting {CACHE.name}.")
    d = torch.load(CACHE, weights_only=False)

    mf_traj2, sm_traj2 = d["mf_traj2"], d["sm_traj2"]

    if True:  # standard outputs: rate and EPR in physical units
        out = {"Spin Model": an.rate_and_epr(sm_traj2, dt, POINT["t_abs"]),
               "Mean Field": an.rate_and_epr(mf_traj2, dt, POINT["t_abs"], cycles=True)}

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

        print("\nmean-field fixed points:")
        for r, rh in zip(d["roots"], d["rho"]):
            print(f"  r {float(r[0]) / dt * 1e3:9.4f} Hz   |λ| {float(rh):.6f}  "
                  f"({'stable' if rh < 1 else 'unstable'})")

    if True:  # cycle, last stretch, sigma against r, age distribution
        hz = lambda x: x.double() / dt * 1e3
        rp.mf_vs_sm(hz(mf_traj2["m"]), hz(mf_traj2["sigma"]),
                    hz(sm_traj2["m"]), hz(sm_traj2["sigma"]),
                    d["p_mf"][None], d["p_sm"][None], dt, pops=("E",),
                    fixed_points=[(hz(r), bool(rh < 1)) for r, rh in zip(d["roots"], d["rho"])],
                    title=TITLE)

    rp.show()
