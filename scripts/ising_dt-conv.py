"""
How fine does the solver step have to be, for each hazard, at a fixed point and on a cycle?

dt0 = 1 throughout: the quantum is the MODEL and is held fixed, dt is the SOLVER and is what
refines (see hazards.resolve_timesteps). At dt = dt0 async, escape and poisson ARE the sigmoid,
bit for bit, so those three curves start from the same point and separate toward their own
continuous-time limits. exp_surv does not: it is not calibrated to the quantum at all, being
poisson's continuum model under exact survival rather than under maximum caliber. It is here
as a SOLVER for that model, and the two must agree in the limit while differing at every
finite step -- which is the sharpest check in this script, since it is the one pair whose
limits are required to coincide. Two operating points of the standard Ising set, either side of the Hopf: J = 10, a
stable fixed point, where sigma vanishes identically and only the rate converges; and J = 17,
the limit cycle.

Where the Hopf is, and why the cycle is measured well above it. At the quantum all three
kernels share one spectrum and the fixed point loses stability at J = 12.0588. Refining moves
it, and not in one direction: async rises to ~12.3, escape falls to ~11.9, and poisson rises
to ~14.36 (measured at dt = 1, 1/2, 1/4, 1/8 as 12.06, 13.33, 13.87, 14.13, the increments
halving cleanly). J = 15 therefore sat only 0.64 above poisson's OWN bifurcation while the
other two sat 2.7-3.1 above theirs, and since the amplitude of a Hopf cycle goes like
sqrt(J - Jc) that halved poisson's peak-to-peak and left it the worst-converging row in the
table -- a small amplitude measured in the sensitive neighbourhood of a bifurcation that is
itself still moving with dt. At J = 17 every kernel sits 2.7 or more above its own Hopf and
the three amplitudes come within 30% of each other.

Not further, though, and that bound is poisson's: its rate is exponential, so the drive that
buys margin also buys stiffness. Peak max(rate)*dt at dt = 1/4 runs 0.08, 0.32, 1.70, 11.7 at
J = 15, 17, 20, 24 -- resolvable at 17 and hopeless by 24, where the step needed is ~1e-3.
The strongly-driven regime where the exponential rate takes over is a separate experiment.

Converging: r = m/dt [1/ms], sigma/dt [nats/ms], the period, the amplitude. Diverging, and kept
as the control: H_fwd/dt and H_rev/dt, which grow as -r ln dt at the predicted rate.

Two things are held fixed so that only the step varies, and both bite if they are not:

  the age window.  U_max = Qm*dt is fitted once per (hazard, regime) at DT_FIT and then held in
      ms. Refitting per dt lays up to `tol` of grid error -- 1.4e-4 on the rate at the usual
      setting -- on top of dt differences that run from 1.8e-4 down to 1.4e-5. It is fitted two
      decades tighter than it is asserted; see TOL_FIT, which records what happened when it
      was not.
  the averaging window.  Taken between the first and last upward mean-crossing of r, so it is a
      whole number of cycles by construction (helpers.cycle_window). A fixed-length window left
      a phase error that moved the mean in the third digit; this one is flat in the
      equilibration to eight.

What came out of it -- the ordering of the kernels, the hierarchy of observables, and the
peak-rate diagnostic that explains both -- is in the companion write-up, beside the tables and
figures below that it is drawn from.
"""

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
import rdme.plots as rp
from rdme.rdm_single import RDMIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS: dict = dict(
    E       = 0.0,    # no external drive: the activity is self-sustained
    beta    = 6.0,
    theta   = 0.5,
    tau_int = 15.0,
    tau_ref = 15.0,   # long: the refractory delay sets the oscillation period
    K_ref   = 15.0,
    dt0     = 1.0,    # the MODEL's time quantum, fixed; dt is what this script sweeps
)

# J either side of the Hopf, with the recording window each needs. The cycle gets 35 periods
# because the cycle-aligned average only removes the PHASE error, not the sampling noise of a
# short run; the fixed point needs no more than its own settling.
REGIMES: dict[str, dict] = {
    # equi_ms is per-regime because the two regimes ask different questions of it. The cycle
    # averages are flat in it from 4000 ms on, which is what it was originally verified
    # against. sigma at the FIXED POINT is not: it is a cancellation that vanishes only when
    # the state has actually stopped moving, so it reports the residual transient rather than
    # a floor, and it does so at a scale eight orders below the cycle. Measured at dt = 0.25,
    # J = 10, as the equilibration is lengthened:
    #
    #     equi      async      escape     poisson    exp_surv
    #     1000 ms   1.7e-04    2.4e-04    -5.4e-06   2.0e-06
    #     4000 ms   5.5e-08    4.6e-08     2.3e-13  -2.0e-15
    #    16000 ms  -6.5e-18   -9.7e-18    -1.9e-18  -5.2e-18
    #
    # i.e. every kernel reaches machine zero, as the recurrence-time argument says it must,
    # and 4000 ms is simply not settled for async and escape. Reading that 5e-8 as a property
    # of a kernel would have been a claim about nothing.
    "fixed point": dict(J=10.0, rec_ms=500.0, equi_ms=16000.0),
    "limit cycle": dict(J=17.0, rec_ms=2500.0, equi_ms=4000.0),
}

HAZARDS = ("async", "escape", "poisson", "exp_surv")
COLORS  = {"async": "#2e6fb7", "escape": "#b4453c", "poisson": "#3f8f5c",
           "exp_surv": "#8a6bbf"}

# lam0*dt <= 1 caps async at dt <= dt0, so the sweep starts at the quantum and halves. Each
# halving costs about four times the previous one -- twice the steps on twice the age bins, and
# the EPR's ring buffer is (Q, Q) -- so the last entry is most of the wall time and most of the
# memory (~3 min and ~6 GiB at 1/32, on the cycle). Dropping it costs a factor of two in the
# resolved error and leaves the whole sweep at about three minutes.
DTS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]

EQUI_MS = 4000.0   # the default; REGIMES overrides it per regime, and says why
DT_FIT  = 0.25     # step the age window is fitted at; U_max in ms is dt-independent

# Age-grid accuracy. Tighter than the 1e-2 the scans use, because here the grid error is not
# the quantity being reported but a contaminant of one: 1e-2 on the rate is 1.4e-4 in absolute
# terms, which is the size of the dt differences themselves.
TOL = 1e-5

# The window is FITTED at TOL_FIT and ASSERTED at TOL, with two decades between them. The gap
# is not caution for its own sake: fit_age_grid aims its interpolation at tol/3 and so lands
# just inside the threshold by construction, while the attractor -- and with it the boundary
# mass -- moves with dt, which is the whole point of the sweep. Fitted at TOL the window
# measured 3.9e-6 at DT_FIT and 4.9e-5 one halving later, i.e. it failed its own assertion.
# The margin is nearly free: the error falls about a decade per 40 ms of extra window here.
TOL_FIT = TOL / 100

# Accuracies, relative to the Richardson limit, that a step is asked to reach. Two of them,
# because one is never enough: at 1e-2 the fixed point is converged at a step the cycle is
# nowhere near, and at 1e-3 the cycle is off the end of any sweep that finishes in minutes.
# Reported for r and for sigma/dt separately -- they are not converged at the same step, and
# sigma is the harder of the two, being a functional of the hazard rather than a moment of it.
THRESHS = (1e-2, 1e-3)

DEVICE = 'cpu'     # one system stepping a few thousand age bins is launch-latency bound, so
                   # there is little for a GPU to do here; the batched scans are where it pays

# Column layout of the per-(hazard, regime) result tensor, (len(DTS), len(COLS)).
COLS = ("r", "sigma", "H_fwd", "H_rev", "period", "ptp", "Q", "phi_max", "rate_max",
        "trunc", "secs")

# The converging observables and the regimes each is reported in, shared by the tables and the
# figures so the two cannot drift apart. Everything but the rate belongs to the cycle: sigma
# vanishes identically at a fixed point, and a settled state has no period or amplitude.
# H_fwd/H_rev are deliberately absent -- they diverge, and are handled as the control.
REPORTED: dict[str, tuple[str, ...]] = {
    "r":      tuple(REGIMES),
    "sigma":  ("limit cycle",),
    "period": ("limit cycle",),
    "ptp":    ("limit cycle",),
}


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE     = hlp.cache_path(STEM, dict(PARAMS, regimes=REGIMES, hazards=list(HAZARDS),
                                      dts=DTS, equi=EQUI_MS, dt_fit=DT_FIT, tol=TOL,
                                      tol_fit=TOL_FIT),
                           root=CACHE_DIR)

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


def checkpoint(res: dict) -> None:
    """ Write the cache after every (hazard, regime) block rather than once at the end.

    A single save at the end means an interrupted run leaves nothing on disk and a run in
    progress is invisible to anyone else -- who then starts a second copy of it. Through a
    temporary and a rename, so a reader sees either the old version or the new one and never
    the half-written middle. The finished blocks are skipped on the next run. """
    CACHE_DIR.mkdir(exist_ok=True)
    # Fold in whatever is on disk before writing. Two runs of this script -- mine and yours --
    # each hold their own `res` in memory, so a plain save makes the last writer win and
    # silently discard every unit the other measured. setdefault keeps what THIS process just
    # computed and adopts any unit it does not have, which makes concurrent runs additive
    # instead of destructive: they divide the work by racing for it rather than redoing it.
    if CACHE.exists():
        try:
            disk = torch.load(CACHE, weights_only=False)
        except Exception:
            disk = {}                      # a torn read from another writer: ours still stands
        adopted = [k for k in disk if k not in res]
        for k in adopted:
            res[k] = disk[k]
        if adopted:
            print(f"  (+{len(adopted)} units from another run of this script)")
    tmp = CACHE.with_suffix(".tmp")
    torch.save(res, tmp)
    tmp.replace(CACHE)


# ── Measurement ───────────────────────────────────────────────────────────────

def build(J: float, hazard: str, dt: float, Qm: list[int] | None = None) -> RDMIsingModel:
    return RDMIsingModel(J, **PARAMS, dt=dt, hazard=hazard, Qm=Qm, device=DEVICE)


def measure(J: float, hazard: str, dt: float, Qm: list[int], rec_ms: float,
            equi_ms: float = None) -> list[float]:
    """ One (hazard, regime, dt) point: equilibrate, record, average over whole cycles. """
    t0  = time.time()
    equi_ms = EQUI_MS if equi_ms is None else equi_ms
    net = build(J, hazard, dt, Qm)
    net.forward(int(equi_ms / dt))
    traj = net.entropy_trajectory(int(rec_ms / dt), pb=False)

    r = traj["r_tot"]
    a, b, n_cyc = hlp.cycle_window(r)
    period = (b - a) * dt / n_cyc if n_cyc else float('nan')

    # The grid is sized on the attractor the run actually found, so the bound is read off the
    # end of the run rather than trusted from the pilot.
    trunc = net.truncation_error()

    # max_step_prob is ONE INSTANT, and on a burst the peak is an order of magnitude above the
    # trough, so it is tracked over a period (a tenth of the window where there is no period).
    # This is the diagnostic that says whether dt resolves the fastest hazard present, which is
    # not the same question as whether the averages have stopped moving.
    # Tracked beside it: the peak of the RATE the hazard represents, which is what says
    # whether a step is small enough for a kernel whose rate is unbounded. phi_max saturates
    # at 1 and so stops reporting once it is there; rate_max*dt keeps counting, and the two
    # together separate "this step resolves the hazard" from "this step has given up".
    n_probe = max(1, int(period / dt) if n_cyc else len(r) // 10)
    phi_max, rate_max = 0.0, 0.0
    for _ in _stepping(net, n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        rate_max = max(rate_max, max(float(l.max()) for l in net.hazard_rate()))

    return [float(r[a:b].mean()),
            float(traj["sigma_tot"][a:b].mean()) / dt,
            float(traj["H_fwd_tot"][a:b].mean()) / dt,
            float(traj["H_rev_tot"][a:b].mean()) / dt,
            period,
            float(r[a:b].max() - r[a:b].min()),
            float(Qm[0]), float(phi_max), float(rate_max), float(trunc), time.time() - t0]


def _stepping(net: RDMIsingModel, n: int):
    """ Yields n times, advancing the model between yields: the sketch in max_step_prob's
    docstring, written out so the generator expression above reads as the max over a window. """
    for _ in range(n):
        yield
        net.update()


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    # Resume: the hash pins the parameters, so an existing file is this run, finished or not.
    res: dict = ({"params": PARAMS, "regimes": REGIMES, "dts": DTS, "u_max": {}}
                 if overwrite or not CACHE.exists()
                 else torch.load(CACHE, weights_only=False))
    t_all = time.time()

    for hazard in HAZARDS:
        for name, reg in REGIMES.items():
            if (hazard, name) in res:
                print(f"\n[{hazard}, {name}] cached")
                continue
            J, rec_ms = reg["J"], reg["rec_ms"]
            equi_ms = reg.get("equi_ms", EQUI_MS)

            # One fit per (hazard, regime), in ms. Sized on the attractor at EQUI+REC, as
            # fit_age_grid requires: the boundary mass keeps settling after equilibration ends.
            print(f"\n[{hazard}, {name}, J = {J:g}] fitting the age window at dt = {DT_FIT} ...")
            Qm_fit = hlp.fit_age_grid(lambda q: build(J, hazard, DT_FIT, q), tol=TOL_FIT,
                                      equi=int((equi_ms + rec_ms) / DT_FIT))
            u_max = Qm_fit[0] * DT_FIT
            res["u_max"][(hazard, name)] = u_max
            print(f"  U_max = {u_max:.1f} ms; Qm = U_max/dt from here on")

            rows = []
            for dt in DTS:
                Qm = [max(1, round(u_max / dt))]
                row = measure(J, hazard, dt, Qm, rec_ms, equi_ms)
                rows.append(row)
                d = dict(zip(COLS, row))
                print(f"  dt = {dt:6.4f}  Q = {int(d['Q']):5d}  r = {d['r']:.8f}  "
                      f"sigma/dt = {d['sigma']:11.4e}  max Phi = {d['phi_max']:.2e}  "
                      f"({d['secs']:.0f}s)")
                assert d["trunc"] < TOL, (
                    f"truncation_error = {d['trunc']:.2e} at dt = {dt}, Q = {int(d['Q'])}: the "
                    f"age window fitted at dt = {DT_FIT} does not hold at this step.")

            res[(hazard, name)] = torch.tensor(rows)
            checkpoint(res)

    checkpoint(res)
    print(f"\nDone in {time.time() - t_all:.0f}s. {CACHE}.")


# ── Analysis ──────────────────────────────────────────────────────────────────

def orders(vals: torch.Tensor) -> torch.Tensor:
    """ Observed order of convergence from successive differences, dt halving each step:
    p_k = log2(|x_k - x_{k-1}| / |x_{k+1} - x_k|). A first-order method gives 1. """
    d = vals.diff().abs()
    return torch.log2(d[:-1] / d[1:].clamp_min(1e-300))


def richardson(vals: torch.Tensor) -> tuple[float, float]:
    """ (extrapolated limit, order used), from the two finest steps and the order the last
    triple shows. Falls back to first order when the differences are too noisy to read one off
    -- near the limit they are differences of nearly equal numbers and the log2 of their ratio
    is meaningless, which is exactly when the extrapolation is also a small correction.

    REFUSES, returning nan, when the sequence is not in its asymptotic regime at all. Two
    symptoms, either of which is disqualifying:

      the successive differences change sign.  The sequence has a turning point, so it is not
          approaching the limit from one side and `vals[-1] - vals[-2]` is not an error
          estimate. Measured on poisson's sigma at J = 17, which falls to dt ~ 1/3 and then
          climbs: the old fallback read an "order" off the turning point and returned a limit
          with four digits, every one of them meaningless.
      an observed order above ~3 or below zero.  Same disease, caught one step earlier -- a
          first-order method cannot show p = 8, so a triple that says so is measuring
          something other than the discretization error.

    Returning nan rather than a cautious number is the point. The caller formats it as a
    refusal and the convergence verdicts fall through to "not resolved", which is the honest
    answer; a plausible-looking limit with a relative-error column beside it is not. """
    if len(vals) > 2:
        o = orders(vals)
        d = vals.diff()
        turning = bool(((d[:-1] * d[1:]) < 0).any())
        wild = bool((o.abs() > 3.0).any()) or bool((o < 0).any())
        if turning or wild:
            why = "differences change sign" if turning else "observed order out of range"
            warnings.warn(f"richardson refused: {why}; orders "
                          f"{['%.2f' % v for v in o.tolist()]}", RuntimeWarning, stacklevel=2)
            return float('nan'), float('nan')
    p = float(orders(vals)[-1]) if len(vals) > 2 else 1.0
    if not (0.2 < p < 4.0):
        p = 1.0
    return float(vals[-1] + (vals[-1] - vals[-2]) / (2 ** p - 1)), p


def converged_at(dts: list[float], vals: torch.Tensor, limit: float, thresh: float) -> float:
    """ Coarsest dt from which the relative error stays under `thresh` all the way down.
    'And stays': a curve crossing a threshold once on its way past is not converged there. """
    err = (vals - limit).abs() / max(abs(limit), 1e-300)
    ok = float('nan')
    for k in range(len(dts) - 1, -1, -1):
        if float(err[k]) < thresh: ok = dts[k]
        else: break
    return ok


if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"\nReading {CACHE.name}.")
    res = torch.load(CACHE, weights_only=False)
    dts = res["dts"]
    col = {k: i for i, k in enumerate(COLS)}

    # Which observables are converged where; REPORTED says where each one means anything.
    for name in REGIMES:
        for key in (k for k, rows in REPORTED.items() if name in rows):
            print(f"\n[{name}] {key}, against the Richardson limit of each hazard:")
            print(f"    {'dt':>8} " + "".join(f"{h:^25}" for h in HAZARDS))
            print(f"    {'':>8} " + "".join(f"{'value':>14}{'rel err':>11}" for _ in HAZARDS))
            for j, dt in enumerate(dts):
                line = f"    {dt:8.4f} "
                for h in HAZARDS:
                    v = res[(h, name)][:, col[key]]
                    lim, _ = richardson(v)
                    line += f"{float(v[j]):14.8g}{abs(float(v[j]) - lim) / abs(lim):11.2e}"
                print(line)

            lims = {h: richardson(res[(h, name)][:, col[key]]) for h in HAZARDS}
            for h in HAZARDS:
                lim, p = lims[h]
                shown_orders = ['%.2f' % o for o in orders(res[(h, name)][:, col[key]]).tolist()]
                if lim != lim:
                    print(f"    {h:>8}: NO LIMIT -- the sequence is not in its asymptotic "
                          f"regime; orders {shown_orders}. Read the values, not a limit.")
                else:
                    print(f"    {h:>8}: limit {lim:.8g}, orders {shown_orders} "
                          f"(extrapolated with {p:.2f})")

            for t in THRESHS:
                reach = {h: converged_at(dts, res[(h, name)][:, col[key]], lims[h][0], t)
                         for h in HAZARDS}
                got = {h: v for h, v in reach.items() if v == v}
                if not got:
                    verdict = "not resolved on this sweep"
                elif len(set(got.values())) == 1 and len(got) == len(HAZARDS):
                    verdict = "all converge at the same step"
                else:
                    best = max(got.values())
                    lead = [h for h in HAZARDS if got.get(h) == best]
                    missing = [h for h in HAZARDS if h not in got]
                    verdict = f"{'/'.join(lead)} converges at the COARSEST step"
                    if missing:
                        verdict += f" ({'/'.join(missing)} not resolved on this sweep)"
                shown = ",  ".join(f"{h} " + (f"{reach[h]:g}" if reach[h] == reach[h] else "--")
                                   for h in HAZARDS)
                print(f"    rel err < {t:.0e} from dt = {shown}   ->  {verdict}")

    # The control: H is supposed to diverge, and by a known amount per halving.
    print(f"\n[control] H/dt per halving of dt, against the -r ln 2 a binned entropy "
          f"against a counting measure predicts:")
    for name in REGIMES:
        for h in HAZARDS:
            pred = float(res[(h, name)][-1, col["r"]]) * float(torch.log(torch.tensor(2.0)))
            for key in ("H_fwd", "H_rev"):
                v = res[(h, name)][:, col[key]]
                print(f"    {h:>8} {name:>12} {key:>6}: "
                      f"{['%.5f' % d for d in v.diff().tolist()]}  (predicts {pred:.5f})")

    # The age window each (hazard, regime) was fitted to, held in ms across the sweep.
    print(f"\n[age window] U_max = Qm*dt, fitted once at dt = {DT_FIT} to "
          f"truncation_error < {TOL_FIT:.0e}:")
    for h in HAZARDS:
        for name in REGIMES:
            print(f"    {h:>8} {name:>12}: {res['u_max'][(h, name)]:7.1f} ms")

    # Resolution diagnostic, which is a different question from the averages settling.
    print(f"\n[diagnostic] peak Phi over one period -- the largest fraction of an age bin that "
          f"fires in one step:")
    print(f"    {'dt':>8} " + "".join(f"{h + ' / ' + n:>22}"
                                      for h in HAZARDS for n in REGIMES))
    for j, dt in enumerate(dts):
        print(f"    {dt:8.4f} " + "".join(f"{float(res[(h, n)][j, col['phi_max']]):22.3e}"
                                          for h in HAZARDS for n in REGIMES))

    # The same question asked of the rate rather than of the probability. Phi saturates at 1
    # and stops reporting there; max(rate)*dt is the quantity resolve_timesteps names as the
    # criterion for an unbounded kernel, and it keeps counting past the point where Phi has
    # flattened -- so it says how far a step is from resolving the hazard, not merely that it
    # does not.
    print(f"\n[diagnostic] peak rate [1/ms] over one period, and max(rate)*dt beside it:")
    print(f"    {'dt':>8} " + "".join(f"{h + ' / ' + n:>26}"
                                      for h in HAZARDS for n in REGIMES))
    for j, dt in enumerate(dts):
        line = f"    {dt:8.4f} "
        for h in HAZARDS:
            for n in REGIMES:
                lam = float(res[(h, n)][j, col["rate_max"]])
                line += f"{lam:14.3e}{lam * dt:12.2e}"
        print(line)


# ── Plots ─────────────────────────────────────────────────────────────────────

if run_plot:

    def dt_axis(ax) -> None:
        """ The shared x axis: log, refining to the right, labelled at the steps actually run.
        Explicit ticks because the sweep is six halvings inside one decade and the log
        locator's minor labels collide into illegibility there. """
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xticks(dts)
        ax.set_xticklabels([f'{d:g}' if d >= 1 else f'1/{round(1 / d)}' for d in dts])
        ax.minorticks_off()
        ax.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
        ax.grid(alpha=0.3)

    # Values on the left, the error against the extrapolated limit on the right. The error
    # panel is the one that answers the question -- on a log-log axis a first-order method is a
    # straight line of slope 1, and the hazard that reaches a given accuracy first is the one
    # whose line sits lower, not the one whose value panel looks flatter.
    #
    # `rows` is where a quantity exists at all and `blank` is where it exists but says nothing,
    # which are different cases and are drawn differently: the cycle quantities simply have no
    # fixed-point row, while sigma keeps a sliver of one, because "identically zero here" is
    # the substantive statement about the fixed point and not a gap in the figure.
    SIGMA_NOTE = ('$\\sigma \\equiv 0$ at a fixed point\n'
                  '(measured $|\\sigma/\\Delta_t| \\leq$ {worst:.0e} nats/ms:\n'
                  'the rounding of a cancellation)')
    PANELS = (
        ("r",      r'mean $r$ [1/ms]',             'Rate',                    None),
        ("sigma",  r'$\sigma/\Delta_t$ [nats/ms]', 'Entropy production rate', SIGMA_NOTE),
        ("period", r'period [ms]',                 'Cycle period',            None),
        ("ptp",    r'peak-to-peak of $r$ [1/ms]',  'Cycle amplitude',         None),
    )

    for key, ylab, title, note in PANELS:
        rows    = REPORTED[key]
        shown   = [n for n in REGIMES if n in rows or note]
        heights = [1.0 if n in rows else 0.3 for n in shown]
        fig, axs = plt.subplots(len(shown), 2, squeeze=False,
                                figsize=(11, 4.4 * sum(heights)),
                                gridspec_kw=dict(height_ratios=heights))
        fig.set_label(key)          # the figure's filename slug; the suptitle is a sentence
        for i, name in enumerate(shown):
            ax_v, ax_e = axs[i]

            # Nothing is drawn for sigma at the fixed point. There is a series there and it is
            # not zero, but it is the rounding of a cancellation at 1e-9 nats/ms -- six orders
            # below the cycle -- and a curve through it reads as a convergence that is not
            # happening. The number is in the table; the panel says what it is.
            if name not in rows:
                worst = max(float(res[(h, name)][:, col[key]].abs().max()) for h in HAZARDS)
                for ax in (ax_v, ax_e):
                    ax.text(0.5, 0.5, note.format(worst=worst),
                            transform=ax.transAxes, ha='center', va='center', fontsize=10,
                            color='0.35')
                    ax.set_axis_off()
                continue

            errs = {}
            for h in HAZARDS:
                v = res[(h, name)][:, col[key]]
                lim, _ = richardson(v)
                errs[h] = (v - lim).abs() / max(abs(lim), 1e-300)
                ax_v.plot(dts, v, 'o-', color=COLORS[h], label=h, markersize=6)
                ax_v.axhline(lim, color=COLORS[h], linestyle=':', linewidth=1)
                ax_e.plot(dts, errs[h], 'o-', color=COLORS[h], label=h, markersize=6)

            # anchored on the coarsest error actually plotted, so the guide sits next to the
            # curves rather than at an offset of its own choosing
            anchor = max(float(errs[h][0]) for h in HAZARDS)
            ax_e.plot(dts, anchor * torch.tensor(dts) / dts[0], 'k--', linewidth=0.8,
                      label=r'slope 1 ($\propto \Delta_t$)')
            for t in THRESHS:
                ax_e.axhline(t, color='0.4', linewidth=0.8)
            ax_e.set_yscale('log')

            for ax in (ax_v, ax_e):
                dt_axis(ax); ax.legend(fontsize=8)
            ax_v.set_ylabel(ylab)
            ax_v.set_title(f'{title} -- {name}')
            ax_e.set_ylabel('relative error vs extrapolated limit')
            ax_e.set_title(f'Convergence -- {name}')

        fig.suptitle(f'{title} vs the solver step, at $\\Delta t_0 = {PARAMS["dt0"]:g}$ '
                     f'(all three kernels are the sigmoid at $\\Delta_t = \\Delta t_0$)')
        fig.tight_layout()

    # The divergent control, which is the one figure here that is NOT a convergence. H is the
    # entropy of a binned process measured against a counting measure, so refining the bin adds
    # ln(1/dt) of resolution per spike and H/dt grows without bound as -r ln dt + const. On a
    # log-x axis that is a straight line of slope r, and the increment per halving is r ln 2 --
    # a prediction with no fitted constant in it, which is why it is worth drawing next to four
    # curves that converge: it says the EPR machinery is reporting what the theory says it
    # should, in the one place where "stops moving" would have been the wrong answer.
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.set_label('H-control')
    styles = {"fixed point": (':', 's'), "limit cycle": ('-', 'o')}

    for h in HAZARDS:
        for name in REGIMES:
            ls, mk = styles[name]
            v = res[(h, name)][:, col["H_fwd"]]
            r_lim, _ = richardson(res[(h, name)][:, col["r"]])
            ax_h.plot(dts, v, ls, marker=mk, color=COLORS[h], markersize=5,
                      label=f'{h} / {name}')

            # the prediction, anchored on the coarsest point: nothing is fitted, the offset is
            # the one measured value and the slope is the converged rate
            pred = float(v[0]) + r_lim * torch.log(torch.tensor(dts[0]) / torch.tensor(dts))
            ax_h.plot(dts, pred, '-', color='0.35', linewidth=0.8, zorder=0)

            # increments against r ln 2, which is the same statement with the offset divided
            # out: every curve should sit on 1, and the finest steps are where it gets there
            for ckey, fill in (("H_fwd", True), ("H_rev", False)):
                ratio = res[(h, name)][:, col[ckey]].diff() / (r_lim * float(torch.log(
                    torch.tensor(2.0))))
                ax_q.plot(dts[1:], ratio, ls, marker=mk, color=COLORS[h], markersize=5,
                          markerfacecolor=COLORS[h] if fill else 'none',
                          label=f'{h} / {name}' if fill else None)

    ax_h.plot([], [], '-', color='0.35', linewidth=0.8,
              label=r'$-r\ln\Delta_t$ + const (no fit)')
    ax_h.set_ylabel(r'$H_{\mathrm{fwd}}/\Delta_t$ [nats/ms]')
    ax_h.set_title('The entropies do not converge, and are not supposed to')

    ax_q.axhline(1.0, color='k', linewidth=0.8)
    ax_q.set_ylabel(r'$\Delta(H/\Delta_t)$ per halving, in units of $r\ln 2$')
    ax_q.set_title('Filled: $H_{\\mathrm{fwd}}$.  Open: $H_{\\mathrm{rev}}$.  Both $\\to 1$')

    for ax in (ax_h, ax_q):
        dt_axis(ax); ax.legend(fontsize=8)
    fig.suptitle(r'Control: $H$ diverges as $-r\ln\Delta_t$, at the predicted rate')
    fig.tight_layout()

    rp.show()
