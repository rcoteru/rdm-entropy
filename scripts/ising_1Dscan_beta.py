import functools
import time
from pathlib import Path

import torch

import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_batch import RDMIsingModelBatch

# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson"        # stated per script: the hazard is its own axis, not part of a preset

PARAMS: dict = pre.params(
    PRESET, hazard=HAZARD,
    J       = 17.0,                            # inside the oscillatory window; see the docstring
    # beta > 0 strictly: the absolute-refractory floor is sized as margin/beta
    # (kernels.abs_refractory_floor), which is +inf at beta = 0 and makes the field nan.
    # beta = 0 is a degenerate model anyway -- Phi = lam0*dt/2 at every age, no threshold.
    beta    = torch.linspace(0.0, 60.0, 301)[1:],   # 0.2 ... 60, spacing 0.2
    dt      = 0.5,
)

# The age grid is FITTED, not declared: the model's q_from_tau default sizes it from the
# kernels alone and is the wrong rule whenever the firing interval is comparable to Q*dt,
# where mass reaches the absorbing last bin and the run finds a different attractor. TOL
# is what goes in the cache key -- Qm is a solver setting, not a model parameter.
TOL = 1e-2   # age-grid accuracy; Qm is fitted to it

AXIS  = "beta"                       # the swept parameter: the one PARAMS gives a vector
LABEL = 'Inverse temperature β'

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

N_LEAD    = 4      # leading eigenvalues kept per fixed point
N_BETA_FP = 100    # beta values to analyse; the cost is dominated by the eigendecomposition
N_SCAN    = 401    # resolution of the sign scan that brackets the roots

# m_tot is cached whole and the rest only in time-average: the bifurcation scatter wants the
# tail of the activity and the spectrogram wants all of it, so this one series has to survive,
# while sigma and the path entropies are only ever plotted as means.
MEANS = ("sigma_tot", "H_fwd_tot", "H_rev_tot")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE_SIM = hlp.cache_path(STEM, dict(PARAMS, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                           tag="traj", root=CACHE_DIR)
CACHE_FP  = hlp.cache_path(STEM, dict(PARAMS, n_lead=N_LEAD, n_pts=N_BETA_FP, n_scan=N_SCAN, tol=TOL),
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
    stride = max(1, mf_fp.B // N_BETA_FP)
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
