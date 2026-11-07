"""
Where the Hopf sits, per hazard, as the solver step refines.

The companion to S1-dt_conv.py, and the reason its cycle regime is measured well above onset.
That script asks how fast each kernel's observables converge; this one asks a prior question --
whether the two operating points it measures are even the same kind of state for every kernel.

rho(J) is the spectral radius of the one-step Jacobian at the fixed point, so the Hopf is
rho = 1 and is located by bisection.

LEGACY NUMBERS (route A, kept as the thing the re-run is read against). The table below was
measured before route B, with the time-quantum construction: `async` is the kernel now called
`bernoulli`, `escape` has no route-B counterpart and is gone, and the then-`poisson` carried
lam0 = 1/dt0 = 1 against today's c = nu0*dt. At the standard Ising set (E = 0, beta = 6,
theta = 0.5, tau_int = tau_ref = K_ref = 15) with dt0 = 1:

    dt       async    escape   poisson
     1      12.0588   12.0588  12.0588     <- the control: at lam0*dt = 1 one map, one spectrum
     1/2    12.187    11.971   13.326
     1/4    12.263    11.941   13.872
     1/8      --        --     ~14.13

i.e. 12.0588 is the SIGMOID (route-A `bernoulli`-at-the-quantum) measurement, and the poisson
column was still rising at dt = 1/8 with its increments halving cleanly toward ~14.36. Refining
did not move the bifurcation in one direction: async rose, escape fell, poisson rose hard. The
consequence that mattered for any J-indexed comparison -- at J = 14 and dt = 1/4 the poisson
fixed point was still stable (rho = 0.99968) where async and escape had both lost stability --
is a route-A statement and is NOT carried over.

That is also why J = 15 was a poor choice for the old convergence study: it sat 0.64 above
poisson's own bifurcation while the other two sat 2.7-3.1 above theirs, and the amplitude of a
Hopf cycle goes like sqrt(J - Jc).

The crossing is shallow -- rho - 1 runs ~1e-3 over a unit of J -- so a scan that looks for a
cycle by its AMPLITUDE will not resolve one until well past the bifurcation. A reported onset
near J ~ 14.5 from such a scan is not in conflict with rho = 1 at 12.06.

WHAT THIS RUNS NOW: S1 under the three route-B kernels. `poisson_exact` is the one whose
dt -> 0 limit is the model's; `poisson` is the same model's per-bin discretization and must
agree with it in the limit; `bernoulli` is route A and is here for the comparison above only.

Known landmine: fp_eigenvalues/is_stable segfault inside MKL's DGEBAL (not an exception) when
the age-grid Jacobian overflows at large beta*h. S1 is beta = 6, so the bracket below is mild,
but if this dies it dies hard -- report the (J, dt, hazard) it died at.
"""

import time
from pathlib import Path

import torch

import rdme.hazards as hzd
import rdme.helpers as hlp
import rdme.presets as pre
from rdme.helpers import orders, richardson
from rdme.rdm_single import RDMIsingModel


PRESET  = "S1"
# No single hazard: the sweep IS over the three kernels, so the hazard is this script's axis
# and is passed per measurement rather than baked into PARAMS.
HAZARDS = ("bernoulli", "poisson", "poisson_exact")

# dt is supplied per measurement from DTS, so PARAMS is built at the coarsest one and the step
# is overridden below; params() requires a dt and there is no quantum to default to.
PARAMS: dict = pre.params(PRESET, hazard=HAZARDS[0], dt=1.0)
MODEL: dict = {k: v for k, v in PARAMS.items() if k not in ("dt", "hazard")}

U_MAX   = 200.0      # ms, held fixed across dt so the grid is not a second variable. Matches
                     # the window S1-cycle_point.py fits for S1 (199.5 ms at tol = 1e-7).
BRACKET = (6.0, 20.0)
N_BISECT = 22        # 14 units of J to ~3e-6; the cost is one spectrum per step
PROBE_J = (13.0, 14.0, 14.5, 15.0, 16.0)

# bernoulli needs nu0*dt <= 1 and poisson needs nu0*dt < 1 strictly (hazards.check_step), so
# with nu0 = 1 the sweep starts at dt = 1 for bernoulli and at the first halving for poisson.
# poisson_exact has no arithmetic cap. dt must also divide t_abs = 1 ms (kernels.n_abs_bins).
DTS     = (1.0, 0.5, 0.25, 0.125)

# Cached per (hazard, dt) cell, not per run. The cost of a cell is one dense eigendecomposition
# of a 2Q x 2Q Jacobian per bisection step, so it grows as dt^-3: bernoulli at dt = 1/4 is
# eight minutes where dt = 1 is eleven seconds. Adding a step to DTS should therefore cost the
# new cells and nothing else, which is what the checkpoint below buys. Keyed on the parameters
# as everywhere in this repo, so a changed preset cannot read back a stale cell.
CACHE_DIR = Path(__file__).parents[1] / "cache"

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def spectral_radius(J: float, hazard: str, dt: float) -> float:
    """ max |eig| of the linearized one-step map at the fixed point.

    Over every root rather than a chosen one: the set is a single point throughout the J range
    scanned here, but taking the max is what makes that an observation rather than an
    assumption -- a second branch appearing would show up as a discontinuity in rho instead of
    being silently picked up or dropped. """
    net = RDMIsingModel(J, **MODEL, dt=dt, hazard=hazard,
                        Qm=[max(1, round(U_MAX / dt))], device=DEVICE)
    roots = net.fixed_points(n_grid=401)
    if roots.numel() == 0:
        raise RuntimeError(f"no fixed point at J={J}, hazard={hazard}, dt={dt}")
    return max(float(net.fp_eigenvalues(r).abs().max()) for r in roots)

def hopf(hazard: str, dt: float, lo: float = BRACKET[0], hi: float = BRACKET[1]) -> float:
    """ The J at which rho crosses 1, by bisection on the bracket. """
    r_lo, r_hi = spectral_radius(lo, hazard, dt), spectral_radius(hi, hazard, dt)
    if not (r_lo < 1.0 < r_hi):
        raise ValueError(f"{hazard} at dt={dt}: bracket [{lo}, {hi}] does not straddle the "
                         f"crossing (rho = {r_lo:.6f}, {r_hi:.6f}). Widen it.")
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, hazard, dt) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


CACHE = hlp.cache_path(Path(__file__).stem,
                       dict(PARAMS, preset=PRESET, hazards=list(HAZARDS), dts=list(DTS),
                            u_max=U_MAX, bracket=list(BRACKET), n_bisect=N_BISECT,
                            probe=list(PROBE_J)),
                       root=CACHE_DIR)


def checkpoint(res: dict) -> None:
    """ Save after every cell, through a temporary and a rename, so an interrupted run keeps
    what it measured and a reader never sees a half-written file. """
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    torch.save(res, tmp)
    tmp.replace(CACHE)


if __name__ == '__main__':
    res: dict = torch.load(CACHE, weights_only=False) if CACHE.exists() else {}

    for dt in DTS:
        # poisson needs c = nu0*dt < 1 STRICTLY, so (poisson, dt = 1) at nu0 = 1 is not a
        # valid step and is skipped rather than silently clamped. bernoulli allows the
        # equality; poisson_exact has no arithmetic cap at all.
        live = []
        for hazard in HAZARDS:
            try:
                hzd.check_step(hazard, PARAMS["nu0"], dt)
            except Exception as e:
                print(f"dt = {dt:6.4f}  {hazard:14s}  skipped: {e}", flush=True)
                continue
            live.append(hazard)

        for hazard in live:
            if ("Jc", hazard, dt) in res:
                print(f"dt = {dt:6.4f}  {hazard:14s}  Hopf at J = "
                      f"{res[('Jc', hazard, dt)]:9.6f}   (cached)", flush=True)
                continue
            t0 = time.time()
            res[("Jc", hazard, dt)] = hopf(hazard, dt)
            checkpoint(res)
            print(f"dt = {dt:6.4f}  {hazard:14s}  Hopf at J = {res[('Jc', hazard, dt)]:9.6f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
        # the shape of rho(J) around it: a bisected root says nothing about how steep the
        # crossing is, and the steepness is what decides whether an amplitude scan can see it
        for hazard in live:
            if ("rho", hazard, dt) not in res:
                res[("rho", hazard, dt)] = [spectral_radius(J, hazard, dt) for J in PROBE_J]
                checkpoint(res)
            vals = " ".join(f"{v:.5f}" for v in res[("rho", hazard, dt)])
            print(f"    rho at J = {'/'.join('%g' % J for J in PROBE_J)}, "
                  f"{hazard:14s}: {vals}", flush=True)

    # ── The limits ────────────────────────────────────────────────────────────
    #
    # The question this table answers, beyond locating each kernel's own Hopf: poisson and
    # poisson_exact are ONE continuous-time model under two discretization maps, so their
    # J_c must converge to the SAME number, while bernoulli's is a different model's and need
    # not. That is the equivalence claim tested on a solved observable rather than on Phi.
    print("\n" + "=" * 72)
    print(f"J_c per kernel as dt -> 0, at {PRESET} (U_max = {U_MAX:g} ms held across dt)")
    print("=" * 72)
    print(f"    {'dt':>8} " + "".join(f"{h:>16}" for h in HAZARDS))
    for dt in DTS:
        row = f"    {dt:8.5f} "
        for h in HAZARDS:
            v = res.get(("Jc", h, dt))
            row += f"{v:16.6f}" if v is not None else f"{'--':>16}"
        print(row)
    print()
    for h in HAZARDS:
        vals = [res[("Jc", h, dt)] for dt in DTS if ("Jc", h, dt) in res]
        if len(vals) < 2:
            print(f"    {h:>14}: too few steps to extrapolate"); continue
        v = torch.tensor(vals)
        lim, pord = richardson(v)
        o = ['%.2f' % q for q in orders(v).tolist()] if len(v) > 2 else ['--']
        print(f"    {h:>14}: limit {lim:.6f} (order {pord:.2f}, orders {o})"
              if lim == lim else f"    {h:>14}: richardson REFUSED (orders {o})")
    print()
    print("    poisson and poisson_exact are the same continuous-time model under two")
    print("    discretization maps, so their limits are REQUIRED to coincide; bernoulli is a")
    print("    different rate function and its limit is its own. The legacy route-A numbers")
    print("    the docstring records (12.0588, 12.187, 12.263 for the sigmoid kernel at")
    print("    dt = 1, 1/2, 1/4) are reproduced here by `bernoulli` to every digit printed.")
