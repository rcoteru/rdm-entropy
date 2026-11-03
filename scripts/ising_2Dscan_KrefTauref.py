import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
import rdme.plots as rp
from rdme.rdm_batch import RDMIsingModelBatch


# ── Parameters ────────────────────────────────────────────────────────────────

# The refractory plane at the zero-input soft-sigmoid point of ising_1Dscan_theta: amplitude
# against timescale, with everything else held at that script's values. K_ref = 0 is kept in
# the range deliberately -- it is the null arm, and that whole row should come out dead.
#
# Std of the activity is the oscillation indicator here, not its mean: at E = 0 a stable cell
# sits at a nonzero fixed point, so only the variance separates the two.
PARAMS: dict = dict(
    J       = 40.0,
    E       = 0.0,
    beta    = 8.0,
    theta   = 0.5,
    tau_int = 10.0,
    tau_ref = torch.linspace(2.0, 60.0, 59),
    K_ref   = torch.linspace(0.0, 50.0, 51),
    dt0     = 1.0,
    dt      = 1.0,
    hazard  = "async",
    eps     = 0.01,
)

# Equilibration and recording are set in MILLISECONDS; the step counts follow dt (see
# helpers.n_steps). A step count says nothing about the system -- the same 20000 steps is a
# different physical window at every dt -- so refining dt here keeps the window and buys
# resolution, which is what refining a solver is supposed to do.
T_EQUI, T_MEAS = 20000, 30000        # ms (= the 20000 + 30000 steps this script ran at dt = 1)

EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

# The age grid is sized from the RATE as well as the kernels. q_from_tau alone truncates where
# the kernel has decayed, which is the right rule only when the firing rate is high enough that
# little mass survives to that age -- and at E = 0 it is not: these cells fire at ~0.007/ms, an
# interval of ~140 ms against a q_from_tau(30) grid only 139 ms deep. The first run of this
# script did exactly that and put 4.4% of the population in the absorbing last bin, biasing the
# rate and CV by ~8% and printing a dark stripe along the tile boundary where Q jumped.
#
# Validated here rather than taken on trust: against a 4x-Q reference on 12 cells spanning
# this grid, q_from_tau_and_rate holds CV to <= 3e-4 and the mean rate to <= 1e-6, while
# q_from_tau is wrong by 100% in CV at K_ref = 50 for every tau_ref -- at Q = 139 it leaves
# 94.5% of the population frozen in the last bin and returns a different attractor.
#
# # assertion after each tile is the real guard -- it fails loudly if the assumption breaks.
TOL = 1e-2   # age-grid accuracy; Qm is fitted to it per tile, never set by hand

KEYS = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE = hlp.cache_path(Path(__file__).stem, dict(PARAMS, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL),
                       root=Path(__file__).parents[1] / "cache")

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim and (overwrite or not CACHE.exists()):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tau_ref, K_ref = PARAMS["tau_ref"], PARAMS["K_ref"]

    # Tiled along tau_ref rather than the other axis, which is not the arbitrary choice it is
    # in the (E, J) and (beta, J) planes. A batch sizes its age grid from its SLOWEST member,
    # so one batch spanning tau_ref in [2, 60] would pay Q(60) = 553 bins for every cell,
    # including the tau_ref = 2 ones that need 30. Tiling along tau_ref lets each tile size
    # its own Q from its own slowest member -- the low-tau_ref tiles get small grids and the
    # run costs a fraction of the flat-Q version.
    @functools.cache          # the tiling search calls this repeatedly
    def q_of(tr_max):
        """ The grid this tile needs, measured. Fitted per tile rather than once for the
        grid because a batch sizes its age grid from its slowest member: one batch spanning
        tau_ref in [2, 60] would pay the tau_ref = 60 grid for every cell in it. """
        tr = PARAMS["tau_ref"][PARAMS["tau_ref"] <= tr_max]
        return hlp.fit_age_grid(
            lambda qm: RDMIsingModelBatch(**dict(PARAMS, tau_ref=tr), Qm=qm, device=device),
            tol=TOL, equi=EQUI + STEPS, verbose=False)[0]


    # A tiling is acceptable when every tile fits whole: epr_plan on a tile returns a batch
    # size at least as large as the tile. It is asked rather than arithmetic done here because
    # the EPR peak is O(B * Q^2) with a coefficient of 2 to 5 -- the ring buffer this search
    # used to size on is under half of it -- and because Q varies BETWEEN tiles, so each one
    # has to be checked against its own grid rather than against Q_max.
    Q_max = q_of(float(tau_ref.max()))
    fits  = lambda t: hlp.epr_plan(len(t) * len(K_ref), q_of(float(t.max())),
                                   device=device, verbose=False)[0] >= len(t) * len(K_ref)
    n_tiles = 1
    while True:
        tiles = torch.tensor_split(tau_ref, n_tiles)
        if all(len(t) >= 2 for t in tiles) and all(map(fits, tiles)):
            break
        n_tiles += 1
        if n_tiles > len(tau_ref) // 2:
            raise RuntimeError("cannot fit a 2-wide tau_ref tile in memory; the age grid is "
                               "too deep for this device -- loosen TOL or coarsen K_ref")
    tr_tiles = tiles

    print(f"Q ranges {q_of(float(tau_ref.min()))}..{Q_max} age bins over the grid; a flat-Q run "
          f"would be {len(tau_ref)*len(K_ref)*Q_max*Q_max*8/2**30:.1f} GiB.")
    peaks = [hlp.epr_peak_bytes(len(t) * len(K_ref), q_of(float(t.max())),
                                hlp.epr_plan(len(t) * len(K_ref), q_of(float(t.max())),
                                             device=device, verbose=False)[1])
             for t in tr_tiles]
    print(f"Running {n_tiles} tile(s) along tau_ref, largest peak "
          f"{max(peaks)/2**30:.1f} GiB of {hlp.device_budget(device)/2**30:.1f} GiB budget.")
    print(f"Per mean-field: {T_EQUI:g} ms equilibration + {T_MEAS:g} ms recording "
          f"= {EQUI + STEPS} steps at dt={PARAMS['dt']:g}")
    print(f"Running on: {device}")

    t0 = time.time()
    parts = []
    for k, tr_t in enumerate(tr_tiles):
        # Qm passed explicitly: without it the model sizes its own age grid from q_from_tau
        # and the rate-aware figure above would only ever have bounded the memory estimate.
        tile: dict = dict(PARAMS, tau_ref=tr_t, Qm=[q_of(float(tr_t.max()))])
        mf = RDMIsingModelBatch(**tile, device=device)
        print(f"[tile {k+1}/{n_tiles}] {mf.B} mean-fields, Q={mf.Qm[0]}, "
              f"tau_ref in [{tr_t.min():.2f}, {tr_t.max():.2f}]")
        # per tile, because Q is: the chunk that paced init_epr_state on the shallow grids is
        # the wrong one on the deep ones, where (chunk, Q, Q) is several times larger
        _, chunk = hlp.epr_plan(mf.B, mf.Qm[0], device=device)
        mf.forward(EQUI, pb=True)
        traj = mf.entropy_trajectory(STEPS, pb=True, chunk=chunk, inplace=True)
        # The product mass*gain*|R(U_max)|, not the mass alone: a cell at a low-rate fixed
        # point correctly holds nearly all its mass at old ages, so a mass check would fail
        # on every grid wide enough to contain a silent cell -- which is all of them here.
        trunc = float(mf.truncation_error().max())
        assert trunc < TOL, (
            f"tile {k+1}: truncation_error = {trunc:.2e} (Q={mf.Qm[0]}, "
            f"U_max={mf.Qm[0]*PARAMS['dt']:.0f} ms): the attractor holds more "
            f"old-age mass than the pilot did. Raise the equi passed to fit_age_grid.")
        parts.append({f"{key}_{stat}": getattr(traj[key], stat)(dim=-1)
                      for key in KEYS for stat in ("mean", "std")})
        del mf, traj
        if device == 'cuda': torch.cuda.empty_cache()
    print(f"Done in {time.time() - t0:.2f}s")

    # tile k holds tr_tiles[k] along axis 0, so the grid reassembles by concatenating there
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

    tau_ref, K_ref = PARAMS["tau_ref"], PARAMS["K_ref"]
    extent = (K_ref.min().item(), K_ref.max().item(),
              tau_ref.min().item(), tau_ref.max().item())

    # CV, not std: the mean rate varies by orders of magnitude across this plane, so a raw
    # std map is a map of the mean. The oscillatory region is where CV is O(1) or larger.
    cv = d["m_tot_std"] / d["m_tot_mean"].clamp_min(1e-300)

    if True:  # where the oscillation lives, and what it costs
        rp.maps({'Activity CV':      cv,
                 'Mean Activity':    d["m_tot_mean"],
                 'Mean Sigma':       d["sigma_tot_mean"],
                 'Std of Sigma':     d["sigma_tot_std"]},
                extent=extent, xlabel='K_ref (refractory amplitude)',
                ylabel='τ_ref (refractory timescale, ms)',
                suptitle='Oscillation and its cost over the (K_ref, tau_ref) plane')

    if True:  # the K_ref = 0 null arm against a few nonzero ones
        idxs = hlp.grab_closest_idxs(K_ref, [0.0, 6.0, 12.0, 25.0])
        ax = rp.curves({f'K_ref={K_ref[i].item():.0f}': cv[:, i] for i in idxs}, x=tau_ref,
                       title='Activity CV vs τ_ref  (K_ref=0 is the null arm)',
                       xlabel='τ_ref (ms)', ylabel='CV of activity', figsize=(8, 6))
        ax.set_yscale('log')

    rp.show()
