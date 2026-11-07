import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_batch import RDMIsingModelBatch


# ── Parameters ────────────────────────────────────────────────────────────────

# Refraction-driven oscillation at ZERO input in a soft regime. With E = 0 the only thing
# sustaining any activity is the softness of the hazard around the threshold -- at large beta
# the network would simply be silent. So low beta is not merely tolerated here, it is what
# supplies the drive.
#
# The text below was written for route A at beta = 8; S1 is beta = 6 and the preset is what
# this now runs at, so the quoted search and the quoted Hopf at theta ~ 0.86 are the LEGACY
# sigmoid-kernel measurement and are kept as the thing the re-run is read against.
#
# The refractory kernel is what makes it oscillate rather than settle. Searched over
# J <= 320, theta <= 2, tau_int <= 40 at beta = 20, 10, 8, 5, 3: with K_ref = 0 not one of
# 358 cells has an unstable fixed point (best mu ~ -0.035/ms), and with K_ref = 25 there is
# an unsaturated oscillatory cell at every beta down to 2.
#
# theta crosses one Hopf at ~0.86: oscillatory below, stable above.
PRESET = "S1"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. Was "async", a route-A key that no longer exists; the
                           # route-B kernel with the same per-bin form is "bernoulli", so this
                           # is a CHANGE OF MODEL and not a rename. A scan wants the solver.

PARAMS: dict = pre.params(
    PRESET, hazard=HAZARD,
    J       = 15.0,
    theta   = torch.linspace(0, 1.6, 300),
    dt      = 1.0,
)

# The age grid is FITTED, not declared: the model's q_from_tau default sizes it from the
# kernels alone and is the wrong rule whenever the firing interval is comparable to Q*dt,
# where mass reaches the absorbing last bin and the run finds a different attractor. TOL
# is what goes in the cache key -- Qm is a solver setting, not a model parameter.
TOL = 1e-2   # age-grid accuracy; Qm is fitted to it

AXIS  = "theta"                      # the swept parameter: the one PARAMS gives a vector
LABEL = 'Threshold θ'

# Equilibration and recording are set in MILLISECONDS; the step counts follow dt (see
# helpers.n_steps). A step count says nothing about the system -- the same 20000 steps is a
# different physical window at every dt -- so refining dt here keeps the window and buys
# resolution, which is what refining a solver is supposed to do.
#
# 10000 + 15000 ms is the window shared by all four S1 1-D scans, so they average over the
# same stretch and can be read against each other: at the ~117 ms period of S1 that is ~85
# cycles of equilibration and ~128 recorded.
T_EQUI, T_MEAS = 10000.0, 15000.0

EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

N_LEAD     = 4      # leading eigenvalues kept per fixed point
N_THETA_FP = 100    # theta values to analyse; the cost is dominated by the eigendecomposition
N_SCAN     = 401    # resolution of the sign scan that brackets the roots

# Two things to know when reading the stability panel of this scan:
#
# |lam| sits within ~2% of the unit circle throughout, which is not a weak instability. For
# a map approximating a flow lam ~ exp(mu dt), so the dt-independent quantity is
# (|lam| - 1)/dt, and |lam| - 1 roughly halves at each refinement while mu converges. Read
# the margin that way. At theta = 0.5 on the converged age grid, mu = +0.034 /ms.
#
# The second note USED to read: the fixed point is far from the lam0 = 1/dt0 rate ceiling
# (3.5% of it at theta = 0.5), so the Hopf is not a saturation artifact, while the limit cycle
# does reach the ceiling at its peak. Under route B there is no 1/dt0 ceiling at all: the rate
# is unbounded and the only cap is 1/t_abs = 1 /ms. Whether the cycle peak reaches THAT is a
# measurement this scan has not been re-run for.
MEANS = ("sigma_tot", "H_fwd_tot", "H_rev_tot")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE_SIM = hlp.cache_path(STEM, dict(PARAMS, preset=PRESET, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                           tag="traj", root=CACHE_DIR)
CACHE_FP  = hlp.cache_path(STEM, dict(PARAMS, preset=PRESET, n_lead=N_LEAD, n_pts=N_THETA_FP, n_scan=N_SCAN, tol=TOL),
                           tag="fp", root=CACHE_DIR)

@functools.cache
def age_grid() -> list[int]:
    """ The age grid both stages use, measured once and reused. Lazy so that a run with both
    caches warm never pays for it, and shared because the simulation and the fixed points are
    the same model at the same parameters -- sizing on the attractor covers both. """
    return hlp.fit_age_grid(lambda qm: RDMIsingModelBatch(**PARAMS, Qm=qm, device="cpu"),
                            tol=TOL, equi=EQUI + STEPS)


run_sim   = True
run_fp    = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE_SIM.exists()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Qm = age_grid()

    # The EPR pass allocates O(B * Q^2), not O(B * Q): the age grid enters SQUARED, so the Q
    # that fit_age_grid just returned is what decides whether this fits on the card, and a
    # grid that came back twice as long costs four times the memory. epr_plan reads the free
    # memory and splits the scan accordingly, so there is no budget constant to keep in step
    # with the regime. One tile is the usual answer; more than one only costs the repeated
    # equilibration.
    axis_values = PARAMS[AXIS]
    B_tile, chunk = hlp.epr_plan(len(axis_values), Qm[0], device=device)
    tiles = torch.tensor_split(axis_values, max(1, -(-len(axis_values) // B_tile)))

    print(f"Running {len(axis_values)} mean-fields for {T_EQUI + T_MEAS:g} ms "
          f"({EQUI + STEPS} steps at dt={PARAMS['dt']:g}) on {device}...")
    t0 = time.time()
    parts = []
    for k, vals in enumerate(tiles):
        mf = RDMIsingModelBatch(**dict(PARAMS, **{AXIS: vals}), Qm=Qm, device=device)
        if len(tiles) > 1: print(f"[tile {k+1}/{len(tiles)}] {mf.B} mean-fields")
        mf.forward(EQUI, pb=True)
        traj = mf.entropy_trajectory(STEPS, pb=True, chunk=chunk, inplace=True)

        trunc = float(mf.truncation_error().max())
        assert trunc < TOL, (f"truncation_error = {trunc:.2e} at Qm={mf.Qm}: the "
                             f"attractor holds more old-age mass than the pilot did. "
                             f"Raise the equi passed to fit_age_grid.")

        # reduced inside the loop: everything but m_tot is only ever plotted as a mean, and
        # holding the tiles whole would defeat the tiling
        parts.append({"m_tot": traj["m_tot"].cpu()}
                     | {f"{key}_{s}": getattr(traj[key], s)(dim=-1).cpu()
                        for key in MEANS for s in ("mean", "std")})
        del mf, traj
        if device == 'cuda': torch.cuda.empty_cache()
    print(f"Done in {time.time() - t0:.2f}s")

    out: dict = {"params": PARAMS, "t_equi": T_EQUI, "t_meas": T_MEAS,
             "equi": EQUI, "steps": STEPS}
    out.update({key: torch.cat([p[key] for p in parts], dim=0) for key in parts[0]})

    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(out, CACHE_SIM)
    print(f"Saved to {CACHE_SIM} ({CACHE_SIM.stat().st_size/2**20:.1f} MiB).")


# ── Fixed points ──────────────────────────────────────────────────────────────

if run_fp and (overwrite or not CACHE_FP.exists()):

    # on CPU: the fixed-point pass is tiny and LAPACK's eig has no GPU equivalent here
    Qm_fp  = age_grid()
    mf_fp  = RDMIsingModelBatch(**PARAMS, Qm=Qm_fp, device="cpu")
    stride = max(1, mf_fp.B // N_THETA_FP)
    idx    = torch.arange(0, mf_fp.B, stride)

    print(f"Finding fixed points at {len(idx)} of {mf_fp.B} {AXIS} values (Q={mf_fp.Qm[0]})...")
    t0 = time.time()
    branches: dict = fpts.branch_scan(mf_fp, mf_fp.grid_axes[AXIS], idx, n_grid=N_SCAN, n_lead=N_LEAD)
    print(f"Done in {time.time() - t0:.2f}s, {len(branches['m_star'])} fixed points.")

    branches["params"] = PARAMS
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(branches, CACHE_FP)
    print(f"Fixed points saved to {CACHE_FP}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_SIM.exists():
        raise FileNotFoundError(f"{CACHE_SIM} not found. Run the simulation first.")

    print(f"Plotting {CACHE_SIM.name}.")
    d  = torch.load(CACHE_SIM, weights_only=False)
    fp = torch.load(CACHE_FP, weights_only=False) if CACHE_FP.exists() else None

    banner = pre.banner(PRESET, PARAMS)

    rp.sweep_figures(PARAMS[AXIS], d["m_tot"], dt=PARAMS["dt"], label=LABEL, banner=banner,
                     fp=fp, means={k: d[f"{k}_mean"] for k in MEANS})
    rp.show()
