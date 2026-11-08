from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_batch import RDMIsingModelBatch


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. The pre-port dict said "async" (route A, = "bernoulli"
                           # here), so this plane is a different model from the one the old
                           # cache held -- which is why the preset name is in the hash below.

PARAMS: dict = pre.params(
    PRESET, hazard=HAZARD,
    J       = torch.linspace(0, 60.0, 81),
    beta    = torch.linspace(1.0, 20.0, 81),
    dt      = 0.5,    # was 1.0 (2026-11-08 brief): dt=1 is inadmissible for poisson (c=1) and
                       # a coarse step on this plane risks the alternate-bin flip-flop that the
                       # acf1 screen below exists to catch. 1/2 or 1/4 is the brief's floor.
                       # 1/4 was tried first and FAILS: the low-beta corner (beta near 1) runs
                       # at a near-zero rate, needing Q ~ 2200 age bins at dt=0.25 (U_max ~ 554
                       # ms) -- epr_plan's own memory budget then cannot fit even 2 J-rows (81
                       # beta values each) into one tile, which the tiling assert correctly
                       # refuses rather than silently dropping the J axis. dt=0.5 roughly
                       # quarters the EPR peak (it scales as Q^2) and is still within the
                       # brief's floor.
)

# The age grid is FITTED per tile, not declared: the model's q_from_tau default is the
# wrong rule whenever the firing interval is comparable to Q*dt. TOL is what goes in the
# cache key -- Qm is a solver setting, not a model parameter.
#
# Loosened from the usual 1e-2 to 5e-2 for THIS grid specifically (2026-11-08 session): the
# beta ~ 1 corner fires at a near-zero rate and needs Q in the thousands even at tol=1e-2,
# which both makes fit_age_grid itself slow (each doubling is a full equi+meas forward pass at
# B=6561) and, at tol=1e-2's resulting Q, overflows epr_plan's per-tile memory budget outright
# (see the dt comment above -- this is the SAME failure mode, one order of magnitude up, that
# dt=0.25 hit). 5e-2 is still inside the brief's own exploration-grade allowance (sigma is
# already quoted at 13-23% relative error on these maps; this trades a comparable amount of
# age-grid accuracy for the run being affordable at all across the full beta range asked for).
TOL = 5e-2   # age-grid accuracy; Qm is fitted to it

# Equilibration and recording are set in MILLISECONDS; the step counts follow dt (see
# helpers.n_steps). A step count says nothing about the system -- the same 20000 steps is a
# different physical window at every dt -- so refining dt here keeps the window and buys
# resolution, which is what refining a solver is supposed to do.
T_EQUI, T_MEAS = 10000, 10000        # ms (= the 10000 + 10000 steps this script ran at dt = 1)

EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

# Reduced at the end of each tile rather than cached whole: the plot stage only ever takes a
# mean and a std over time, and the full (n_J, n_beta, T) trajectories of this grid are 7.6 GiB
# against the ~1 MiB below. Re-run if a trajectory itself is ever the question. m_tot_acf1 is
# the exception -- a scalar per cell, not a (mean, std) pair -- the lag-1 autocorrelation screen
# the brief asks for: at this coarse, exploration-grade dt, the alternate-bin flip-flop is the
# artifact a map can hide a cell inside without anyone noticing, and acf1 is what catches it.
KEYS = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE = hlp.cache_path(Path(__file__).stem, dict(PARAMS, preset=PRESET, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                       root=Path(__file__).parents[1] / "cache")
PARTIAL = CACHE.with_suffix(".partial.pt")

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


def checkpoint(parts: list[dict]) -> None:
    """ Save the tiles measured so far, atomically. This grid runs to 81x81 at dt=0.25 and the
    one stage that can take the whole process down without raising -- fp_eigenvalues' MKL
    DGEBAL landmine, see CLAUDE.md -- does not happen to be called anywhere in this script (it
    only runs `forward` and `entropy_trajectory`, never a stability eigendecomposition), so
    that specific crash cannot hit this run. What CAN still kill a multi-hour batch job is the
    ordinary kind -- OOM on a tile, a killed process, a bad node -- and saving nothing until the
    very end turns any of those into a wasted run. This writes after every tile instead, through
    a temporary and a rename so a reader never sees a half-written file. """
    tmp = PARTIAL.with_suffix(".tmp")
    torch.save({"parts": parts}, tmp)
    tmp.replace(PARTIAL)


def acf1_of(traj: torch.Tensor) -> torch.Tensor:
    """ Lag-1 autocorrelation of m_tot over the recorded window, per batch element (B, T) ->
    (B,). Same convention as S1-dt_conv.py / S1-cycle_point.py / S1-hopf_type.py: -1 is the
    whole population firing in alternate bins, the step artifact a coarse dt on this plane can
    produce and that a mean/std reduction alone cannot see. """
    x = traj - traj.mean(dim=-1, keepdim=True)
    return (x[..., :-1] * x[..., 1:]).sum(dim=-1) / x.pow(2).sum(dim=-1).clamp_min(1e-300)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    J, beta = PARAMS["J"], PARAMS["beta"]

    Q = hlp.fit_age_grid(lambda qm: RDMIsingModelBatch(**PARAMS, Qm=qm, device=device),
                         tol=TOL, equi=EQUI + STEPS)[0]
    # The EPR pass allocates O(B * Q^2), not O(B * Q): the age grid enters SQUARED, so the Q
    # that fit_age_grid just returned is what decides whether this fits, and sizing on the
    # ring buffer alone -- the obvious thing, since it is the only allocation named in the
    # signature -- under-counts the peak by between 2x and 5x. epr_plan reads the memory that
    # is actually free and returns both the tile size and the chunk, so there is no budget
    # constant to keep in step with the regime.
    B_tile, CHUNK = hlp.epr_plan(len(J) * len(beta), Q, device=device)
    n_tiles = max(1, -(-len(J) // max(1, B_tile // len(beta))))
    J_tiles = torch.tensor_split(J, n_tiles)
    assert all(len(t) >= 2 for t in J_tiles), "tiles of one J value would drop the J axis"

    # Resume: a .partial.pt from an earlier, interrupted run of this exact grid (same n_tiles,
    # since that follows deterministically from PARAMS) means its tiles need not be redone.
    parts: list[dict] = []
    if PARTIAL.exists():
        try:
            saved = torch.load(PARTIAL, weights_only=False)["parts"]
        except Exception:
            saved = []                      # a torn read from an earlier crash: start over
        if len(saved) <= n_tiles:
            parts = saved
    if parts:
        print(f"Resuming from {PARTIAL.name}: {len(parts)}/{n_tiles} tiles already measured.")

    print(f"Q = {Q} age bins; running {n_tiles} tile(s) of <= {max(map(len, J_tiles))} J values.")
    print(f"Per mean-field: {T_EQUI:g} ms equilibration + {T_MEAS:g} ms recording "
          f"= {EQUI + STEPS} steps at dt={PARAMS['dt']:g}")
    print(f"Running on: {device}")

    t0 = time.time()
    for k, J_t in enumerate(J_tiles):
        if k < len(parts):
            print(f"[tile {k+1}/{n_tiles}]  cached from {PARTIAL.name}")
            continue
        tile: dict = dict(PARAMS, Qm=[Q], J=J_t)
        mf = RDMIsingModelBatch(**tile, device=device)
        print(f"[tile {k+1}/{n_tiles}] {mf.B} mean-fields, "
              f"J in [{J_t.min():.3f}, {J_t.max():.3f}]")
        mf.forward(EQUI, pb=True)
        traj = mf.entropy_trajectory(STEPS, pb=True, chunk=CHUNK, inplace=True)
        trunc = float(mf.truncation_error().max())
        assert trunc < TOL, (f"tile {k+1}: truncation_error = {trunc:.2e}: the age grid "
                                   f"is too short (Qm={mf.Qm[0]}). Re-fit with fit_age_grid.")
        # reduce here, inside the loop: holding the tiles whole until the end would put the
        # full grid's trajectories in memory, which is the thing the tiling exists to avoid
        part = {f"{key}_{stat}": getattr(traj[key], stat)(dim=-1)
               for key in KEYS for stat in ("mean", "std")}
        part["m_tot_acf1"] = acf1_of(traj["m_tot"])
        parts.append(part)
        checkpoint(parts)
        print(f"    acf1 in [{float(part['m_tot_acf1'].min()):+.3f}, "
              f"{float(part['m_tot_acf1'].max()):+.3f}]")
        del mf, traj
        if device == 'cuda': torch.cuda.empty_cache()
    print(f"Done in {time.time() - t0:.2f}s")

    # tile k holds J_tiles[k] along axis 0, so the grid reassembles by concatenating there
    out: dict = {key: torch.cat([p[key] for p in parts], dim=0).cpu() for key in parts[0]}
    out["params"] = PARAMS

    CACHE.parent.mkdir(exist_ok=True)
    torch.save(out, CACHE)
    PARTIAL.unlink(missing_ok=True)     # superseded by the finished cache
    print(f"Saved to {CACHE} ({CACHE.stat().st_size/2**20:.1f} MiB).")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"Plotting {CACHE.name}.")
    d = torch.load(CACHE, weights_only=False)

    hlp.pub_style()
    banner = pre.banner(PRESET, PARAMS)

    # NOTE: the J axis is plotted in the units it was passed in, i.e. physical units. The
    # pre-port version of this script multiplied it by dt here, which was right when J
    # carried a 1/dt and is wrong now that the input is built from the rate. Nothing in the
    # cache can tell the two apart, so check this against the plot before quoting a number.
    J, beta = PARAMS["J"], PARAMS["beta"]
    extent = (beta.min().item(), beta.max().item(), J.min().item(), J.max().item())

    # The acf1 screen: flagged, not silently plotted. A cell below 0.97 (S1-hopf_type.py /
    # S1-cycle_point.py's threshold for "no alternate-bin flip-flop") is reported by name
    # rather than folded into the map and left for someone to notice by eye.
    acf1 = d["m_tot_acf1"]
    n_flag = int((acf1 < 0.97).sum())
    print(f"acf1 over the grid: min {float(acf1.min()):+.4f}, "
          f"{n_flag}/{acf1.numel()} cells below 0.97"
          + ("  -- see the acf1 map; do not read sigma off a flagged cell" if n_flag else ""))

    if True:  # average / std of each quantity over the plane
        rp.maps({'Mean Activity':   d["m_tot_mean"],
                 'Std of Activity': d["m_tot_std"],
                 'Mean Sigma':      d["sigma_tot_mean"],
                 'Std of Sigma':    d["sigma_tot_std"],
                 'Lag-1 acf (m_tot)': acf1},
                extent=extent, xlabel='beta (Inverse Temperature)',
                ylabel='J (Coupling Strength)',
                suptitle=f'Mean and std over the (beta, J) plane\n{banner}', ncols=3)

    if True:  # slices of the grid at a handful of J values
        for key, name in (("m_tot", 'Activity'), ("sigma_tot", 'Sigma')):
            avg, std = d[f"{key}_mean"], d[f"{key}_std"]
            idxs = hlp.grab_closest_idxs(J, [5.0, 10.0, 15.0, ])

            ax = rp.curves({f'J={J[i].item():.2f}': avg[i] for i in idxs}, x=beta,
                           title=f'{name} vs beta for Different J Values',
                           xlabel='beta (Inverse Temperature)', ylabel=f'Mean {name}',
                           figsize=(8, 6))
            for i, line in zip(idxs, ax.get_lines()):
                ax.fill_between(beta, avg[i] - std[i], avg[i] + std[i],
                                color=line.get_color(), alpha=0.3)

    rp.show()
