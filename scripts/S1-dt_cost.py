"""
B1: the tolerance -> (dt, dt/2) table. What a stated accuracy costs, at S1.

The deliverable every later run depends on. A script that wants the stationary rate to 1%, or
sigma on the cycle to 0.1%, should be able to read its Richardson pair off a table instead of
guessing a step and hoping -- and should be able to see, before it starts, what that pair is
going to cost in wall time and in memory.

Two operating points of S1:

  S1/fp   J = 10, a stable fixed point. sigma vanishes identically there, so the only
          quantity with a tolerance is the stationary rate.
  cycle   the cycle point. J = 17 is the candidate S1-cycle_point.py measures against; set
          CYCLE_J from POINTS["S1/cycle"] once Raul fixes it. Four quantities: rate, period,
          amplitude and sigma -- and sigma is the one that drives the cost.

MONOTONICITY IS A PRECONDITION, not a result. A tolerance read off a Richardson limit means
nothing if the sequence is not approaching one from a single side, so this refuses to print a
table for any quantity whose sequence helpers.richardson rejects (`nan` + NO LIMIT on a sign
change or an observed order outside [0, 3]). The guard is not relaxed, and a REFUSED row is
the honest answer: there is no pair that buys that tolerance on this sweep.

MEMORY IS THE BINDING CONSTRAINT, not time. The EPR ring buffer is (Q, Q) and Q = U_max/dt, so
peak memory goes as dt^-2: each halving costs four times the previous one. Wall time goes as
dt^-2 as well (twice the steps on twice the bins), but 6 GiB arrives before the clock does.

Two memory columns, and `model` is the one to size a run with.

`model` is helpers.epr_peak_bytes, the fitted (Q, Q) formula: per-cell, analytic, and checked
against a measurement by its own test. It is what a caller sizing a tile should read.

`rss` is the PROCESS high-water mark from getrusage. It is the number that decides whether a
run survives, and it counts torch's own allocator, which a Python-level tracer cannot see at
all (torch does not allocate through Python's allocator, so tracemalloc reports essentially
nothing here). But it NEVER FALLS and there is one of it per process, so it is NOT a per-cell
figure: once the largest cell in the run has been measured, every later row reports that same
number whatever its own size -- which is exactly what the first pass of this table did, every
cycle row repeating the 2.11 GiB that the LARGER fixed-point grid had already set. Read it as
an upper bound on the process and `model` for the cell. Making it per-cell means one process
per cell; that is a follow-up, not something this table needs.

STEP VALIDITY, for reading the table with either kernel. `poisson_exact` has no arithmetic cap
and its step is set by accuracy alone (hazards.step_error: the local relative error is
lam*dt/2 at the PEAK HAZARD rate, which on this cycle is far above the population rate).
`poisson` additionally needs c = nu0*dt < 1 strictly, so at nu0 = 1 /ms no pair in this table
may have dt >= 1 for it; `bernoulli` admits the equality. Every dt here divides t_abs = 1 ms,
which kernels.n_abs_bins requires of all three.
"""

import resource
import time
from pathlib import Path

import torch

import rdme.hazards as hzd
import rdme.helpers as hlp
import rdme.presets as pre
from rdme.helpers import converged_at, orders, richardson
from rdme.rdm_single import RDMIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson_exact"   # B1 is about the solver; the table's caption carries the other two

PARAMS: dict = pre.params(PRESET, hazard=HAZARD, dt=0.5)
MODEL: dict = {k: v for k, v in PARAMS.items() if k not in ("dt", "hazard")}

CYCLE_J = 17.0      # the candidate; replace from POINTS["S1/cycle"] when it is set

# Each halving is ~4x the previous one in both time and memory. It runs to 1/64 because the
# first pass of this table stopped at 1/32 and found sigma on the cycle still 2.1% off its own
# extrapolated limit there -- i.e. the quantity the table exists for had no entry at any
# tolerance. One more halving is ~5 GiB and ~20 minutes and is what makes the sigma row sayable.
DTS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

# Per-regime windows, as in S1-dt_conv.py: the fixed point needs a long equilibration for its
# sigma cancellation to reach machine zero, the cycle needs recorded periods rather than
# settling.
REGIMES: dict[str, dict] = {
    "S1/fp": dict(J=pre.resolve("S1/fp")["J"], equi_ms=16000.0, rec_ms=500.0,
                  quantities=("r",)),
    "cycle": dict(J=CYCLE_J, equi_ms=4000.0, rec_ms=2500.0,
                  quantities=("r", "period", "ptp", "sigma")),
}

# Tolerances the table is read at. Three decades, because one is never enough: at 1e-2 the
# fixed point is converged at a step the cycle is nowhere near, and at 1e-4 the cycle is off
# the end of any sweep that finishes in minutes.
THRESHS = (1e-2, 1e-3, 1e-4)

TOL     = 1e-5        # age-grid accuracy; the grid error must not contaminate the dt error
TOL_FIT = TOL / 100   # fitted two decades tighter than it is asserted -- see S1-dt_conv.py
DT_FIT  = 0.25        # the window is fitted here once and held in ms across the sweep

DEVICE = 'cpu'        # one system on a few thousand age bins is launch-latency bound, and the
                      # memory column is only meaningful on the path that allocates in RAM
torch.set_default_dtype(torch.float64)

COLS = ("r", "sigma", "period", "ptp", "Q", "phi_max", "lam_max", "acf1",
        "secs", "rss_bytes", "model_bytes")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE = hlp.cache_path(STEM, dict(PARAMS, preset=PRESET, regimes=REGIMES, dts=DTS,
                                  tol=TOL, tol_fit=TOL_FIT, dt_fit=DT_FIT),
                       root=CACHE_DIR)

run_sim   = True
run_table = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE

for _dt in DTS + [DT_FIT]:
    hzd.check_step(HAZARD, PARAMS["nu0"], _dt)


def build(J: float, dt: float, Qm: list[int] | None = None) -> RDMIsingModel:
    return RDMIsingModel(J, **MODEL, dt=dt, hazard=HAZARD, Qm=Qm, device=DEVICE)


def measure(J: float, dt: float, Q: int, equi_ms: float, rec_ms: float) -> list[float]:
    """ One (regime, dt) point, with its wall time and its peak allocation.

    The RSS high-water is read AFTER the whole measurement, not around the EPR call alone: the
    model's own state, the trajectory buffers and the ring buffer all coexist at the peak, and
    the number a caller needs is the one that decides whether the run fits, not the size of the
    largest single object in it. """
    t0 = time.time()

    net = build(J, dt, [Q])
    net.forward(hlp.n_steps(equi_ms, dt))
    traj = net.entropy_trajectory(hlp.n_steps(rec_ms, dt), pb=False)

    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    period = (z - a) * dt / n_cyc if n_cyc else float('nan')

    # The step diagnostics, over one period (a tenth of the window where there is none): the
    # peak of a burst is one instant. phi_max saturates at 1 and stops reporting there, so the
    # peak HAZARD rate is tracked beside it -- that is what hazards.step_error wants, and it is
    # not the population rate, which the dead time caps at 1/t_abs.
    n_probe = max(1, int(period / dt) if n_cyc else len(r) // 10)
    phi_max, lam_max = 0.0, 0.0
    n_abs = net.n_abs
    for _ in range(n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        lam_max = max(lam_max, max(float(l[n_abs:].max()) for l in net.hazard_rate()))
        net.update()

    x = r[a:z] - r[a:z].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    trunc = float(net.truncation_error())
    assert trunc < TOL, (f"truncation_error = {trunc:.2e} at dt = {dt}, Q = {Q}: the window "
                         f"fitted at dt = {DT_FIT} does not hold at this step.")

    secs = time.time() - t0
    # ru_maxrss is in kibibytes on Linux and is a process high-water mark: it never falls, so
    # this is the peak reached by the PROCESS so far, not by this cell. See the docstring.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    return [float(r[a:z].mean()),
            float(traj["sigma_tot"][a:z].mean()) / dt,
            period,
            float(r[a:z].max() - r[a:z].min()),
            float(Q), phi_max, lam_max, acf1, secs, float(peak),
            float(hlp.epr_peak_bytes(1, Q, inplace=False))]


# ── Measurement ───────────────────────────────────────────────────────────────

if run_sim:
    res: dict = ({"dts": DTS, "cols": COLS, "u_max": {}}
                 if overwrite or not CACHE.exists()
                 else torch.load(CACHE, weights_only=False))

    for name, reg in REGIMES.items():
        if name in res:
            print(f"\n[{name}] cached"); continue
        J, equi_ms, rec_ms = reg["J"], reg["equi_ms"], reg["rec_ms"]

        print(f"\n[{name}, J = {J:g}] fitting the age window at dt = {DT_FIT} ...")
        Qm_fit = hlp.fit_age_grid(lambda q: build(J, DT_FIT, q), tol=TOL_FIT,
                                  equi=hlp.n_steps(equi_ms + rec_ms, DT_FIT))
        u_max = Qm_fit[0] * DT_FIT
        res["u_max"][name] = u_max
        print(f"  U_max = {u_max:.1f} ms; Qm = U_max/dt from here on")

        rows = []
        for dt in DTS:
            row = measure(J, dt, max(1, round(u_max / dt)), equi_ms, rec_ms)
            rows.append(row)
            d = dict(zip(COLS, row))
            print(f"  dt = {dt:7.5f}  Q = {int(d['Q']):5d}  r = {d['r']:.8f}  "
                  f"sigma/dt = {d['sigma']:11.4e}  {d['secs']:7.1f}s  "
                  f"rss {d['rss_bytes'] / 2**30:6.3f} GiB  "
                  f"model {d['model_bytes'] / 2**30:6.3f} GiB", flush=True)

        res[name] = torch.tensor(rows)
        CACHE_DIR.mkdir(exist_ok=True)
        tmp = CACHE.with_suffix(".tmp"); torch.save(res, tmp); tmp.replace(CACHE)

    print(f"\nSaved {CACHE}")


# ── The table ─────────────────────────────────────────────────────────────────

if run_table:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the measurement first.")
    res = torch.load(CACHE, weights_only=False)
    dts, col = res["dts"], {k: i for i, k in enumerate(res["cols"])}

    LABEL = {"r": "rate r [1/ms]", "sigma": "sigma/dt [nats/ms]",
             "period": "period [ms]", "ptp": "amplitude (ptp of r) [1/ms]"}

    print("\n" + "=" * 92)
    print(f"B1 -- tolerance -> (dt, dt/2) at {PRESET}, hazard {HAZARD}, "
          f"nu0 = {PARAMS['nu0']:g} /ms, t_abs = {PARAMS['t_abs']:g} ms")
    print("=" * 92)

    # 0. the precondition
    print("\n[0] MONOTONICITY -- the precondition. A tolerance read off a limit that does not")
    print("    exist is not a tolerance. helpers.richardson refuses on a sign change in the")
    print("    successive differences or an observed order outside [0, 3]; not relaxed.")
    limits: dict = {}
    for name, reg in REGIMES.items():
        for key in reg["quantities"]:
            v = res[name][:, col[key]]
            lim, p = richardson(v)
            limits[(name, key)] = (lim, p)
            o = ['%.2f' % x for x in orders(v).tolist()]
            d = v.diff()
            single = bool(((d[:-1] * d[1:]) > 0).all())
            verdict = ("REFUSED -- no limit" if lim != lim else
                       f"limit {lim:.8g} (order {p:.2f})")
            print(f"    {name:>7} {LABEL[key]:<28} "
                  f"{'single-signed' if single else 'SIGN CHANGE':<14} "
                  f"orders {str(o):<22} {verdict}")

    # 1. the cost of each pair, measured
    print("\n[1] COST per (dt, dt/2) pair. Wall time is the SUM of the two members, since a")
    print("    Richardson pair costs both; peak memory is the FINER member's, since that is")
    print("    what has to fit. CPU path, float64, one system, single process.")
    print("    model = helpers.epr_peak_bytes (per cell, analytic). rss<= = process high-water,")
    print("    which never falls, so it bounds the PROCESS and not the cell.")
    print("    WALL TIME IS ELAPSED, NOT CPU: a concurrent run inflates it silently. Check the")
    print("    machine was idle before quoting these, and re-run rather than scaling them.")
    for name in REGIMES:
        t = res[name]
        print(f"\n    [{name}]  U_max = {res['u_max'][name]:.1f} ms")
        print(f"    {'pair (dt, dt/2)':>18}{'Q(dt/2)':>10}{'wall [s]':>11}"
              f"{'rss<= [GiB]':>13}{'model [GiB]':>13}{'max Phi':>10}{'lam_max':>12}"
              f"{'lam*dt/2':>11}{'acf1':>8}")
        for i in range(len(dts) - 1):
            secs = float(t[i, col["secs"]] + t[i + 1, col["secs"]])
            lam  = float(t[i + 1, col["lam_max"]])
            print(f"    ({dts[i]:>6g}, {dts[i+1]:<7g}){int(t[i+1, col['Q']]):>10}"
                  f"{secs:11.1f}{float(t[i+1, col['rss_bytes']]) / 2**30:13.3f}"
                  f"{float(t[i+1, col['model_bytes']]) / 2**30:13.3f}"
                  f"{float(t[i+1, col['phi_max']]):10.4f}"
                  f"{lam:12.3e}{hzd.step_error(lam, dts[i+1]):11.2e}"
                  f"{float(t[i+1, col['acf1']]):8.3f}")

    # 2. the table itself
    print("\n[2] THE TABLE. For each quantity and tolerance: the COARSEST pair whose finer")
    print("    member is inside the tolerance of the extrapolated limit AND stays inside it")
    print("    all the way down -- a curve that crosses a threshold once on its way past is")
    print("    not converged there (helpers.converged_at).")
    for name, reg in REGIMES.items():
        t = res[name]
        print(f"\n    [{name}]")
        print(f"    {'quantity':<28}" + "".join(f"{'tol ' + f'{x:.0e}':>26}" for x in THRESHS))
        for key in reg["quantities"]:
            lim, _ = limits[(name, key)]
            line = f"    {LABEL[key]:<28}"
            for th in THRESHS:
                if lim != lim:
                    line += f"{'REFUSED':>26}"; continue
                dt_ok = converged_at(dts, t[:, col[key]], lim, th)
                if dt_ok != dt_ok:
                    line += f"{'not on this sweep':>26}"; continue
                i = dts.index(dt_ok)
                pair = f"({dts[i-1]:g}, {dt_ok:g})" if i else f"(--, {dt_ok:g})"
                secs = float(t[i, col["secs"]] + (t[i - 1, col["secs"]] if i else 0.0))
                gib = float(t[i, col["model_bytes"]]) / 2 ** 30
                line += f"{pair + f'  {secs:.0f}s {gib:.2f}G':>26}"
            print(line)
        print(f"    (a pair '(--, dt)' means dt is the coarsest step on the sweep and has no")
        print(f"     coarser partner here; extend DTS upward if a Richardson pair is needed.)")

    # 3. the caption
    print("\n[3] CAPTION -- reading this table for the other two kernels.")
    for h in ("poisson_exact", "poisson", "bernoulli"):
        cap = hzd.HAZARDS[h].step_cap(PARAMS["nu0"])
        bad = [d for d in dts if cap is not None and d > cap]
        print(f"    {h:>14}: "
              + (f"no arithmetic cap; the step is set by accuracy alone (lam*dt/2 above)"
                 if cap is None else
                 f"admits dt <= {cap:.6g} ms at nu0 = {PARAMS['nu0']:g} /ms"
                 + (f"; INVALID here: {bad}" if bad else "; every dt in this table is valid")))
    print(f"    Every dt here divides t_abs = {PARAMS['t_abs']:g} ms, which n_abs_bins requires")
    print(f"    of all three. The timings and the limits are poisson_exact's own and do NOT")
    print(f"    transfer: poisson's per-bin error changes sign at threshold, so Richardson")
    print(f"    cannot be run on it at all on an attractor that crosses one.")
