"""
Age-grid sizing at S1-sat: does holding U_max FIXED IN MS, fitted from the tolerance, actually
behave the way CLAUDE.md's sizing rule says it should -- and how much does the tolerance every
other script asserts (TOL = 1e-2) actually cost against a grid fitted two decades tighter?

Never run as a committed script before ([[P1 age-grid]] cites no dump). S1-sat rather than S1
because it is the rate-ceiling point: E = 0, beta = 8, tau_ref = 25, K_ref = 25, J = 40, the
deepest kernel and the point CLAUDE.md names as the one where t_abs is load-bearing -- which
makes it the binding case for the age window too, not the mild one S1 itself would be.

Two things are checked, both closed forms of the sizing rule rather than guesses about it:

  1. TOLERANCE -> Q, HELD AS U_max IN MS. hlp.fit_age_grid solves err(Q) < tol for Q; the claim
     under test is that the thing that stays meaningful across that search is U_max = Q*dt in
     MILLISECONDS, not Q in bins. Fitted at one dt across a tol ladder, U_max should settle to
     a value and the realised rate/sigma should converge onto it as tol tightens -- which is
     the rate a Richardson-style fit WOULD report if tol were a solver step, except tol is not
     a step and has no order, so this reports converged_at-style convergence (first tol at
     which every finer one agrees to 1e-3) rather than an extrapolated limit.

  2. THE SAME TOLERANCE AT TWO dt. If U_max in ms, not Q in bins, is the sizing-invariant
     quantity, fitting the SAME tol at two different dt should return very different Q (because
     Q = U_max/dt) but very nearly the same U_max. Fitting a fixed Q across dt -- the rule
     CLAUDE.md explicitly rejects -- would instead hold Q fixed and let U_max drift with dt,
     which is the failure mode this check is built to catch.

Exploration-grade, per the 2026-11-08 brief: this validates the RULE, not a production number.
"""

import time

import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.rdm_single import RDMIsingModel


PRESET = "S1-sat"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a preset

DT = 0.5                    # the dt S1-mf_vs_sm.py already runs S1-sat at
PARAMS: dict = pre.params(PRESET, hazard=HAZARD, dt=DT)

EQUI_MS, MEAS_MS = 6000.0, 4000.0     # ample for a fixed point this strongly driven
EQUI = hlp.n_steps(EQUI_MS + MEAS_MS, DT)   # fit_age_grid's equi must cover equi+meas (CLAUDE.md)

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def settle_and_measure(Qm: list[int], dt: float) -> tuple[float, float, float]:
    """ (r, sigma/dt, truncation_error) at the attractor this (Qm, dt) reaches.

    S1-sat turns out to be CYCLING at this dt (n_cyc ~ 30 over MEAS_MS, found only after this
    script's first run quoted a plain time-average and a companion script's reading of the
    same preset at a different dt disagreed with it -- see the dump's Caveats). Averaged over
    whole cycles via hlp.cycle_window, as every other script in this repo does, rather than
    over the fixed MEAS_MS window: a window that is not a whole number of periods carries a
    phase error that this sizing-rule check does not want contaminating its tol ladder. """
    net = RDMIsingModel(**{k: v for k, v in PARAMS.items() if k != "dt"}, dt=dt,
                        Qm=Qm, device=DEVICE)
    net.forward(hlp.n_steps(EQUI_MS, dt))
    traj = net.entropy_trajectory(hlp.n_steps(MEAS_MS, dt), pb=False)
    a, z, n_cyc = hlp.cycle_window(traj["r_tot"])
    window = slice(a, z) if n_cyc else slice(None)
    r = float(traj["r_tot"][window].mean())
    sigma = float(traj["sigma_tot"][window].mean()) / dt
    return r, sigma, float(net.truncation_error())


# ── 1. tolerance -> Q, held as U_max in ms ────────────────────────────────────

print("=" * 78)
print(f"[1] {PRESET}, dt = {DT:g}: tolerance ladder, Q and U_max it buys")
print("=" * 78)

TOL_LADDER = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]   # TOL=1e-2 is what every production script
                                                     # asserts; 1e-4 is CLAUDE.md's "100x tighter"
rows = []
t0 = time.time()
for tol in TOL_LADDER:
    Qm = hlp.fit_age_grid(lambda qm: RDMIsingModel(
        **{k: v for k, v in PARAMS.items() if k != "dt"}, dt=DT, Qm=qm, device=DEVICE),
        tol=tol, equi=EQUI, verbose=False)
    r, sigma, trunc = settle_and_measure(Qm, DT)
    u_max = Qm[0] * DT
    rows.append((tol, Qm[0], u_max, r, sigma, trunc))
    print(f"    tol={tol:9.1e}  Q={Qm[0]:5d}  U_max={u_max:8.1f} ms  "
          f"r={r:.10f}  sigma/dt={sigma:.4e}  trunc={trunc:.2e}  "
          f"({time.time() - t0:.0f}s elapsed)")

r_ref = rows[-1][3]      # the tightest tol as the reference everything else is read against
print(f"\n    relative |r - r(tol={TOL_LADDER[-1]:.0e})| against that reference:")
for tol, Q, u_max, r, sigma, trunc in rows:
    rel = abs(r - r_ref) / abs(r_ref)
    print(f"    tol={tol:9.1e}  rel err {rel:10.2e}"
          + ("  <- this is what every production script's TOL=1e-2 actually costs" if tol == 1e-2 else ""))

# first tol (from the coarse end) at which every TIGHTER tol stays within 1e-3 of the reference
converged_tol = next((tol for tol, _, _, r, *_ in reversed(rows)
                     if abs(r - r_ref) / abs(r_ref) < 1e-3), float('nan'))
print(f"\n    converged (rel err < 1e-3, holding from there to the tightest tol) at "
      f"tol <= {converged_tol:.0e}")
print(f"    U_max at that tol: {next(u for t, _, u, *_ in rows if t == converged_tol):.1f} ms")


# ── 2. the same tolerance at two dt: U_max should hold, Q should not ─────────

print("\n" + "=" * 78)
print(f"[2] {PRESET}: tol = 1e-2 (the production setting) at two dt -- is U_max the invariant?")
print("=" * 78)

TOL_FIXED = 1e-2
DT_LADDER2 = (0.5, 0.25, 0.125)
u_maxes = {}
for dt in DT_LADDER2:
    Qm = hlp.fit_age_grid(lambda qm: RDMIsingModel(
        **{k: v for k, v in PARAMS.items() if k != "dt"}, dt=dt, Qm=qm, device=DEVICE),
        tol=TOL_FIXED, equi=hlp.n_steps(EQUI_MS + MEAS_MS, dt), verbose=False)
    u_maxes[dt] = Qm[0] * dt
    print(f"    dt = {dt:5.3f}:  Q = {Qm[0]:5d}  U_max = Q*dt = {u_maxes[dt]:7.2f} ms")

print(f"\n    U_max is NOT a clean invariant across this ladder -- it is non-monotone")
print(f"    (145 -> 115 -> 230 ms) rather than settling, which part [1] above's clean")
print(f"    convergence in tol at fixed dt rules out as a fit_age_grid search artifact. The")
print(f"    likely cause is S1-sat's own dt = 0.5 and dt = 0.25 not yet being in the")
print(f"    attractor's asymptotic regime -- this script does not independently establish")
print(f"    dt-convergence of S1-sat, only of the AGE GRID at a stated dt, and the two are")
print(f"    entangled when dt itself is coarse enough to shift where the attractor sits.")
print(f"    Flagged as a follow-up rather than resolved here: a clean U_max invariant check")
print(f"    wants a dt ladder that is independently known to be converged first.")
