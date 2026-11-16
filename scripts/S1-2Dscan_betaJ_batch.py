""" A slice of S1-2Dscan_betaJ.py's (beta, J) map, run as ONE batch in the rebuilt code and with
the old code's own tiling (epr_plan / chunk) in the pre-rebuild code, for cell-for-cell
agreement, wall time and peak memory (brief "P1 2026-11-13 code rebuild", Step 4).

The code version is chosen by PYTHONPATH alone: whichever `rdme` is importable decides it
(rdme.network -> "new", rdme.rdm_batch -> "old"). One process per (version, slice):

    PYTHONPATH=<src> python scripts/S1-2Dscan_betaJ_batch.py <slice>

Physics, observables and reductions are S1-2Dscan_betaJ.py's; only the age-grid fit is replaced
by a held Q (see QM below), and the plot stage is dropped. """
from pathlib import Path
import resource
import sys
import time

import torch

import rdme.helpers as hlp
import rdme.presets as pre
try:
    from rdme.network import RDMIsingModelBatch
    VERSION = "new"
except ImportError:
    from rdme.rdm_batch import RDMIsingModelBatch
    VERSION = "old"


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson_exact"

# S1-2Dscan_betaJ.py's grid, built the same way (before the float64 default is set, as there)
PARAMS: dict = pre.params(
    PRESET, hazard=HAZARD,
    J    = torch.linspace(0, 60.0, 81),
    beta = torch.linspace(1.0, 20.0, 81),
    dt   = 0.5,
)

# Held, not fitted: the Q that fit_age_grid returned at TOL = 5e-2 on the full grid in
# agents/dumps/P1_2026-11-08_S1-2Dscan_betaJ_exploration-map.md (and again in the rebuild's
# Step 3 pilot). Holding it removes the fit as a confound between versions.
QM  = 1566
TOL = 5e-2   # still asserted against truncation_error, as in S1-2Dscan_betaJ.py

T_EQUI, T_MEAS = 10000, 10000        # ms
EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

KEYS = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")

# Named slices as (J index range, beta index range) into the 81 x 81 grid. Full beta rows, so
# the old code tiles along J exactly as S1-2Dscan_betaJ.py did (3 J-rows per tile at Q = 1566).
SLICES = {
    "J27-36": (slice(27, 36), slice(0, 81)),   # J in [20, 26.25], all beta: 729 cells
    "J69-78": (slice(69, 78), slice(0, 81)),   # J in [51.75, 57.75], all beta: 729 cells
    "full":   (slice(0, 81),  slice(0, 81)),   # the whole map: 6561 cells
}
SLICE = sys.argv[1]
J_IDX, B_IDX = SLICES[SLICE]

CACHE = hlp.cache_path(Path(__file__).stem,
                       dict(PARAMS, preset=PRESET, t_equi=T_EQUI, t_meas=T_MEAS, Qm=QM,
                            slice=SLICE, code=VERSION),
                       tag=f"{VERSION}_{SLICE}", root=Path(__file__).parents[1] / "cache")
torch.set_default_dtype(torch.float64)


def acf1_of(traj: torch.Tensor) -> torch.Tensor:
    """ Lag-1 autocorrelation over the recorded window, (B, T) -> (B,); as S1-2Dscan_betaJ.py. """
    x = traj - traj.mean(dim=-1, keepdim=True)
    return (x[..., :-1] * x[..., 1:]).sum(dim=-1) / x.pow(2).sum(dim=-1).clamp_min(1e-300)


# ── Simulation ────────────────────────────────────────────────────────────────

if not CACHE.exists():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    J, beta = PARAMS["J"][J_IDX], PARAMS["beta"][B_IDX]
    B = len(J) * len(beta)

    if VERSION == "old":     # the old script's tiling, from the O(B * Q^2) EPR budget
        B_tile, CHUNK = hlp.epr_plan(B, QM, device=device)
        n_tiles = max(1, -(-len(J) // max(1, B_tile // len(beta))))
        epr_kw = dict(chunk=CHUNK, inplace=True)
    else:                    # one batch
        n_tiles, epr_kw = 1, dict(inplace=True)
    J_tiles = torch.tensor_split(J, n_tiles)
    assert all(len(t) >= 2 for t in J_tiles), "tiles of one J value would drop the J axis"
    print(f"code={VERSION} slice={SLICE}: {B} cells, Q={QM}, {n_tiles} tile(s) of "
          f"<= {max(map(len, J_tiles))} J values; {EQUI}+{STEPS} steps at dt={PARAMS['dt']:g}")

    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)
    t_equi = t_epr = 0.0
    parts: list[dict] = []
    t0 = time.time()
    for k, J_t in enumerate(J_tiles):
        mf = RDMIsingModelBatch(**dict(PARAMS, J=J_t, beta=beta), Qm=[QM], device=device)
        ta = time.time()
        mf.forward(EQUI, pb=False)
        sync(); tb = time.time()
        traj = mf.entropy_trajectory(STEPS, pb=False, **epr_kw)
        sync(); tc = time.time()
        t_equi += tb - ta; t_epr += tc - tb
        trunc = float(mf.truncation_error().max())
        assert trunc < TOL, f"tile {k+1}: truncation_error = {trunc:.2e} at Qm={QM}"
        part = {f"{key}_{stat}": getattr(traj[key], stat)(dim=-1)
                for key in KEYS for stat in ("mean", "std")}
        part["m_tot_acf1"] = acf1_of(traj["m_tot"])
        parts.append(part)
        print(f"[tile {k+1}/{n_tiles}] {mf.B} cells, equi {tb-ta:.1f} s, epr {tc-tb:.1f} s, "
              f"trunc {trunc:.2e}", flush=True)
        del mf, traj
        if device == 'cuda': torch.cuda.empty_cache()
    wall = time.time() - t0

    out: dict = {key: torch.cat([p[key] for p in parts], dim=0).cpu() for key in parts[0]}
    out["meta"] = dict(code=VERSION, slice=SLICE, J_idx=(J_IDX.start, J_IDX.stop),
                       beta_idx=(B_IDX.start, B_IDX.stop), Qm=QM, n_tiles=n_tiles,
                       wall_s=wall, equi_s=t_equi, epr_s=t_epr,
                       gpu_peak_bytes=(torch.cuda.max_memory_allocated()
                                       if device == 'cuda' else None),
                       host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                           * 1024,
                       device=torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                       torch=torch.__version__)
    out["params"] = PARAMS
    CACHE.parent.mkdir(exist_ok=True)
    torch.save(out, CACHE)
    print({k: v for k, v in out["meta"].items()})
    print(f"Saved to {CACHE}.")
else:
    print(f"{CACHE} exists; nothing to do.")
