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
    dt      = 1.0,
)

# The age grid is FITTED per tile, not declared: the model's q_from_tau default is the
# wrong rule whenever the firing interval is comparable to Q*dt. TOL is what goes in the
# cache key -- Qm is a solver setting, not a model parameter.
TOL = 1e-2   # age-grid accuracy; Qm is fitted to it

# Equilibration and recording are set in MILLISECONDS; the step counts follow dt (see
# helpers.n_steps). A step count says nothing about the system -- the same 20000 steps is a
# different physical window at every dt -- so refining dt here keeps the window and buys
# resolution, which is what refining a solver is supposed to do.
T_EQUI, T_MEAS = 10000, 10000        # ms (= the 10000 + 10000 steps this script ran at dt = 1)

EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

# Reduced at the end of each tile rather than cached whole: the plot stage only ever takes a
# mean and a std over time, and the full (n_J, n_beta, T) trajectories of this grid are 7.6 GiB
# against the ~1 MiB below. Re-run if a trajectory itself is ever the question.
KEYS = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE = hlp.cache_path(Path(__file__).stem, dict(PARAMS, preset=PRESET, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                       root=Path(__file__).parents[1] / "cache")

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


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

    print(f"Q = {Q} age bins; running {n_tiles} tile(s) of <= {max(map(len, J_tiles))} J values.")
    print(f"Per mean-field: {T_EQUI:g} ms equilibration + {T_MEAS:g} ms recording "
          f"= {EQUI + STEPS} steps at dt={PARAMS['dt']:g}")
    print(f"Running on: {device}")

    t0 = time.time()
    parts = []
    for k, J_t in enumerate(J_tiles):
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
        parts.append({f"{key}_{stat}": getattr(traj[key], stat)(dim=-1)
                      for key in KEYS for stat in ("mean", "std")})
        del mf, traj
        if device == 'cuda': torch.cuda.empty_cache()
    print(f"Done in {time.time() - t0:.2f}s")

    # tile k holds J_tiles[k] along axis 0, so the grid reassembles by concatenating there
    out: dict = {key: torch.cat([p[key] for p in parts], dim=0).cpu() for key in parts[0]}
    out["params"] = PARAMS

    CACHE.parent.mkdir(exist_ok=True)
    torch.save(out, CACHE)
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

    if True:  # average / std of each quantity over the plane
        rp.maps({'Mean Activity':   d["m_tot_mean"],
                 'Std of Activity': d["m_tot_std"],
                 'Mean Sigma':      d["sigma_tot_mean"],
                 'Std of Sigma':    d["sigma_tot_std"]},
                extent=extent, xlabel='beta (Inverse Temperature)',
                ylabel='J (Coupling Strength)',
                suptitle=f'Mean and std over the (beta, J) plane\n{banner}')

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
