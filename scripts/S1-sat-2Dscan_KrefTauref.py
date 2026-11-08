import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_batch import RDMIsingModelBatch


# ── Parameters ────────────────────────────────────────────────────────────────

# The refractory plane through S1-sat: amplitude against timescale, with everything else held
# at the preset. K_ref = 0 is kept in the range deliberately -- it is the null arm, and that
# whole row should come out dead.
#
# Std of the activity is the oscillation indicator here, not its mean: at E = 0 a stable cell
# sits at a nonzero fixed point, so only the variance separates the two.
#
# The preset is S1 (2026-11-08 brief), not S1-sat. The pre-port dict here was E = 0, beta = 8,
# theta = 0.5, tau_int = 10, J = 40, which is S1-sat exactly -- that was the RIGHT call at the
# port (CLAUDE.md's "Ported 2026-11-07" note), since the pre-port dict is what it was. But the
# paper's figures all sit at S1, so this plane needs to as well, and the preset is now a
# variable rather than a literal: PRESET = "S1-sat" reproduces the original port's plane; this
# script runs it at S1.
PRESET = "S1"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. The pre-port dict said "async" (route A, = "bernoulli"
                           # here), so this plane is a different model from the old cache's.

# S1, unlike S1-sat, carries no J of its own -- it is "the axis the system is studied along"
# for every S1-pointed script, so a run through it has to supply one explicitly. J = 17 is the
# working cycle point (S1-cycle_point.py, 5.47 past the measured Hopf at J_c = 11.5318): the
# question this plane asks ("does the oscillation survive at a shallower refractory kernel?")
# is read against the preset's own operating point, not an arbitrary other one. J = 17 is a
# WORKING value, not a decision -- POINTS["S1/cycle"] is still unset (presets.py).
J_S1 = 17.0

PARAMS: dict = pre.params(
    PRESET, hazard=HAZARD,
    J       = J_S1,
    tau_ref = torch.linspace(2.0, 60.0, 59),
    K_ref   = torch.linspace(0.0, 50.0, 51),
    dt      = 0.25,   # was 1.0: dt=1 is inadmissible for poisson and risks the alternate-bin
                       # flip-flop the acf1 screen below exists to catch (2026-11-08 brief).
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
#
# Loosened from the usual 1e-2 to 1e-1 for THIS grid specifically (2026-11-08 session, run at
# S1 rather than S1-sat): S1's beta = 6 is weaker drive than S1-sat's beta = 8 AND the working
# J = 17 here is far below S1-sat's J = 40, so at K_ref toward 50 the firing rate is far lower
# and the tail far longer than the S1-sat original run already found pathological (see the
# comment above) -- 5e-2 (tried first) still did not return the tau_ref=60 pilot within a
# practical time even running alone on an otherwise idle GPU. 1e-1 is a further exploration-
# grade trade, in the same spirit as S1-2Dscan_betaJ.py's 5e-2, needed because this plane's
# worst corner (K_ref=50 at S1's weaker drive) is a more extreme case than that one's.
TOL = 1e-1   # age-grid accuracy; Qm is fitted to it per tile, never set by hand

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
    """ Save the tau_ref tiles measured so far, atomically. This script never calls
    fp_eigenvalues (only `forward` and `entropy_trajectory`), so the MKL DGEBAL segfault
    landmine does not apply to it -- but an ordinary crash mid-grid still should not cost a
    multi-tile run that tiles unevenly in both count and per-tile cost. """
    tmp = PARTIAL.with_suffix(".tmp")
    torch.save({"parts": parts}, tmp)
    tmp.replace(PARTIAL)


def acf1_of(traj: torch.Tensor) -> torch.Tensor:
    """ Lag-1 autocorrelation of m_tot over the recorded window, per batch element (B, T) ->
    (B,). Same convention as S1-dt_conv.py / S1-cycle_point.py / S1-hopf_type.py. """
    x = traj - traj.mean(dim=-1, keepdim=True)
    return (x[..., :-1] * x[..., 1:]).sum(dim=-1) / x.pow(2).sum(dim=-1).clamp_min(1e-300)


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
        print(f"  fitting age grid for tau_ref <= {tr_max:g} ({len(tr)} values) ...", flush=True)
        return hlp.fit_age_grid(
            lambda qm: RDMIsingModelBatch(**dict(PARAMS, tau_ref=tr), Qm=qm, device=device),
            tol=TOL, equi=EQUI + STEPS, verbose=True, max_doublings=4)[0]


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

    # Resume: a .partial.pt from an earlier, interrupted run of this exact grid (same n_tiles
    # and tau_ref boundaries, since both follow deterministically from PARAMS) means its tiles
    # need not be redone.
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

    t0 = time.time()
    for k, tr_t in enumerate(tr_tiles):
        if k < len(parts):
            print(f"[tile {k+1}/{n_tiles}]  cached from {PARTIAL.name}")
            continue
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

    # tile k holds tr_tiles[k] along axis 0, so the grid reassembles by concatenating there
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

    tau_ref, K_ref = PARAMS["tau_ref"], PARAMS["K_ref"]
    extent = (K_ref.min().item(), K_ref.max().item(),
              tau_ref.min().item(), tau_ref.max().item())

    # CV, not std: the mean rate varies by orders of magnitude across this plane, so a raw
    # std map is a map of the mean. The oscillatory region is where CV is O(1) or larger.
    cv = d["m_tot_std"] / d["m_tot_mean"].clamp_min(1e-300)

    # The acf1 screen: flagged, not silently plotted. See S1-2Dscan_betaJ.py's twin.
    acf1 = d["m_tot_acf1"]
    n_flag = int((acf1 < 0.97).sum())
    print(f"acf1 over the grid: min {float(acf1.min()):+.4f}, "
          f"{n_flag}/{acf1.numel()} cells below 0.97"
          + ("  -- see the acf1 map; do not read sigma off a flagged cell" if n_flag else ""))

    if True:  # where the oscillation lives, and what it costs
        rp.maps({'Activity CV':      cv,
                 'Mean Activity':    d["m_tot_mean"],
                 'Mean Sigma':       d["sigma_tot_mean"],
                 'Std of Sigma':     d["sigma_tot_std"],
                 'Lag-1 acf (m_tot)': acf1},
                extent=extent, xlabel='K_ref (refractory amplitude)',
                ylabel='τ_ref (refractory timescale, ms)',
                suptitle=f'Oscillation and its cost over the (K_ref, tau_ref) plane\n{banner}',
                ncols=3)

    if True:  # the K_ref = 0 null arm against a few nonzero ones
        idxs = hlp.grab_closest_idxs(K_ref, [0.0, 6.0, 12.0, 25.0])
        ax = rp.curves({f'K_ref={K_ref[i].item():.0f}': cv[:, i] for i in idxs}, x=tau_ref,
                       title='Activity CV vs τ_ref  (K_ref=0 is the null arm)',
                       xlabel='τ_ref (ms)', ylabel='CV of activity', figsize=(8, 6))
        ax.set_yscale('log')

    rp.show()
