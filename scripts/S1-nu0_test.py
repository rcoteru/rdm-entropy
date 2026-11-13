"""
The nu0 re-anchoring test, step A: the shift-gauge control, and re-tuning the operating point.

From the brief `agents/briefs/P1 2026-11-08 nu0 re-anchoring test.md`. The question the brief
exists for is whether anchoring nu0 on the escape rate at threshold (nu0 = 0.01 ms^-1, 10 Hz)
instead of on 1 ms^-1 lowers the hazard peak that sets the dt floor. This script is the first
two stages of that: the control that says the implementation does what the algebra says, and
the re-tuning that has to come before any cost comparison can be read.

THE OPERATING POINT UNDER TEST IS PROVISIONAL AND LIVES HERE ONLY. Call it `S1-nu0test`:

    preset S1, hazard poisson_exact, nu0 = 0.01 ms^-1, t_abs = 1 ms, theta = 0.5 HELD.

It is NOT in presets.py and must not be added there: the brief is explicit that this is a test,
not a re-parametrization, and that no preset changes because of it.

t_abs = 1 ms, NOT the 4 ms that would put nu0*t_abs at the cortical 0.04. That is a decision
taken by the brief-holder and carried here: at nu0 = 0.01 ms^-1 the invariant this script runs
at is nu0*t_abs = 0.01, a different point on the same axis, not a failed run. The grounds are
`P1_2026-11-08_S1-sat-refractory_window_t_abs-not-load-bearing.md`, which found t_abs inert to
every printed digit from 0 to 4 ms at S1-sat. A t_abs = 4 ms spot check at the re-anchored
point is a later step, not this one.

THE TRAP, which stage 1 is the control on. (nu0, theta) -> (nu0 e^{beta delta}, theta + delta)
leaves the hazard nu0 e^{beta(S + R - theta)} unchanged -- it is a pure relabel, not a
re-anchoring. What makes the re-anchoring real is holding theta at the preset's value and
moving nu0 alone, which is what stage 2 does.

E IS NOT THE SAME MOVE, and this script asserted that it was until the control measured it.
The argument for degeneracy -- E enters only through I = sum w r + E, so raising E by -delta
should shift h exactly as lowering theta by delta does -- fails on the age resolution of the
synaptic term: rdm.update_synaptic_term carries S per AGE with S(0) = 0, so a constant E
reaches the field as E*(1 - (1-alpha)^n), zero for a neuron that has just fired and E only
asymptotically, while theta shifts every age by the same amount. The two twins are therefore
NOT the same system, and the control's E arm measures the difference: ~1% in r, period,
phi_max and lam_max at both operating points, against 1e-14 for the theta arm. E is a genuine
second knob, which is what the brief assumed when it said to restore the rate with J and E
and never with theta.

  stage `gauge`   nu0 = 0.01 WITH theta = 0.5 - ln(100)/beta. Must reproduce S1 to solver
                  error at both operating points (J = 10, J = 17). If it does not, the finding
                  is the implementation and nothing below it means anything.

  stage `retune`  nu0 = 0.01 with theta = 0.5 HELD. The fixed-point branches of r(J), the
                  linear stability of each, the attractor actually reached from a warm start
                  on the active branch, and the J at which the active branch carries the S1
                  operating rate.

  stage `erest`   nu0 = 0.01 with theta = 0.5 HELD and J held on S1's own axis: the E that
                  restores S1's operating rate at each point, the stability of the restored
                  state, the re-anchored Hopf, and lam_max / max Phi against S1's at dt = 0.25.

  stage `ladder`  the dt ladder (1/2 -> 1/32) at the restored point: lam_max, max Phi,
                  lam*dt/2, acf1, r, sigma, period and ptp per dt, re-anchored against plain
                  S1 at the SAME dt, same structure both sides; and the observed orders on
                  every quantity, reported as S1-dt_conv.py reports them.

  stage `cost`    the dt and the memory that buy 1% sigma, through helpers.converged_at, so
                  the table reads as S1-dt_cost.py's does, against S1's own column.

E IS TUNED ONCE AT dt = 0.25 AND HELD DOWN THE LADDER (decision carried into this script, not
taken by it). A dt -> 0 limit is taken at fixed model parameters: re-tuning E per rung would
change the model at every rung and make Richardson meaningless, and CLAUDE.md is explicit that
model parameters do not move when dt does. The realised r therefore drifts along the ladder --
so does S1's own at fixed J, which is what makes the comparison like for like, and both drifts
are reported.
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.epr as epr
import rdme.hazards as hzd
import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.helpers import converged_at, orders, richardson
from rdme.rdm_single import RDMIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET  = "S1"
HAZARD  = "poisson_exact"   # the solver; the hazard is its own axis and is stated here
NU0_NEW = 0.01              # ms^-1: the escape rate at threshold, against S1's 1 ms^-1

_S1   = pre.resolve(PRESET)
BETA  = _S1["beta"]
# The shift that takes nu0 = 1 to nu0 = 0.01: nu0_new = nu0 e^{beta*delta}. Computed, not
# typed, so the control cannot be fitted to its own answer by a rounded decimal.
DELTA       = math.log(NU0_NEW / _S1["nu0"]) / BETA        # = -ln(100)/6
THETA_GAUGE = _S1["theta"] + DELTA

DT_REF = 0.25     # every rate measurement and every root scan is at this step. Divides t_abs
DT_EIG = 1.0      # the spectral radius only. The Jacobian is (2Q, 2Q) with Q = U_max/dt, so a
                  # dense eigensolve at DT_REF is hours; at DT_EIG it is seconds. Stage 2
                  # spot-checks the CLASSIFICATION at DT_EIG/2 so this is a cost choice and
                  # not a silent assumption
DT_EIG_CHECK = 0.5

PARAMS_S1    = pre.params(PRESET, hazard=HAZARD, dt=DT_REF)
PARAMS_GAUGE = pre.params(PRESET, hazard=HAZARD, dt=DT_REF,
                          nu0=NU0_NEW, theta=THETA_GAUGE)
PARAMS_NU0   = pre.params(PRESET, hazard=HAZARD, dt=DT_REF, nu0=NU0_NEW)   # "S1-nu0test"
# The same relabel wearing E's clothes. E enters the field only through I = sum w r + E, so
# raising E by -DELTA shifts h exactly as lowering theta by DELTA does, with theta left at the
# preset's value. The brief allows "restore the rate with J and E"; this arm is what says E is
# not a second knob but the shift gauge again, measured rather than argued.
E_GAUGE      = _S1["E"] - DELTA
PARAMS_GAUGE_E = pre.params(PRESET, hazard=HAZARD, dt=DT_REF, nu0=NU0_NEW, E=E_GAUGE)

# The two operating points of S1 the whole comparison is against. The targets are MEASURED
# here at DT_REF rather than quoted from a dump, so the two sides are compared like for like.
J_FP, J_CYCLE = 10.0, 17.0      # S1/fp, and the working cycle point
REGIMES = {"fp": J_FP, "cycle": J_CYCLE}

# The branch scan of the re-anchored point. Runs from below the fold that creates the active
# branch to well past it; 5 apart, which resolves the structure without paying for 100 cells.
J_SCAN  = torch.arange(60.0, 160.1, 5.0)
R_SCAN_MAX = 0.1        # 1/ms: the window the root scan covers, as a rate. Above this the
                        # model is in the alternate-bin regime (acf1 -> -1) and the roots
                        # there are not operating points
N_GRID  = 4001          # root-scan resolution in overlaps over [0, R_SCAN_MAX*dt]
SN_BRACKET, N_BISECT = (60.0, 80.0), 24   # the fold, by bisection on "is there an active root"

# Stage 3: restoring the rate with E, at J held on S1's own axis values. The scan is what says
# whether r(E) has one root or several; the bisection runs inside whichever scan cell brackets
# the target, so a second crossing would be seen rather than stepped over.
E_SCAN   = torch.arange(0.0, 2.01, 0.1)
E_BISECT = 20          # halvings of one 0.1-wide scan cell -> 1e-7 in E
# The re-anchored Hopf, as a cut in J at the tuned E. J, not E, because S1's own reference
# (J_c = 11.534, so J = 17 sits 5.466 past onset) is a distance in J, and a distance past onset
# is only comparable if it is measured along the same axis.
J_ONSET_BRACKET, N_BISECT_J = (4.0, 17.0), 20
J_C_S1   = 11.534      # S1's Hopf under poisson_exact, as a dt -> 0 limit (CLAUDE.md)
DELTA_S1 = J_CYCLE - J_C_S1

T_EQUI, T_MEAS = 8000.0, 4000.0   # ms
TOL     = 1e-5        # age-grid accuracy
TOL_FIT = TOL / 100   # fitted two decades tighter than it is asserted -- see S1-dt_conv.py

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)

COLS = ("r", "sigma", "period", "ptp", "Q", "phi_max", "lam_max", "acf1")

# ── The ladder and the cost (stages 4 and 5) ──────────────────────────────────

DTS_L  = [0.5, 0.25, 0.125, 0.0625, 0.03125]   # every one divides t_abs = 1 ms
DT_FIT = DT_REF        # the age window is fitted here and then HELD IN MS down the ladder
SIDES  = ("S1", "re")  # the control and the re-anchored point, same structure both sides

# What each regime has a tolerance on. sigma vanishes identically at a fixed point and a
# settled state has no period or amplitude, so only the rate converges to anything there.
QUANTITIES = {"fp": ("r",), "cycle": ("r", "period", "ptp", "sigma")}
THRESHS = (1e-2, 1e-3, 1e-4)
SIGMA_TARGET = 1e-2    # the brief's deliverable: the dt that buys 1% sigma

# ptp is REPORTED BUT NEVER EXTRAPOLATED (decided 2026-11-08, S1-dt_conv.py's NO_LIMIT): the
# cycle amplitude has no dt -> 0 limit, and `richardson` does not catch that on its own --
# its refusal tests are a sign change in the differences and an observed order outside [0, 3],
# and a sequence whose order is still CLIMBING trips neither. So it returns a number here, and
# that number is not a limit. Quoted at its stated dt on both sides, and the climb is the
# finding.
NO_LIMIT = ("ptp",)

COLS_L = ("r", "sigma", "H_fwd", "H_rev", "H_rel_fwd", "H_rel_rev", "period", "ptp",
          "Q", "phi_max", "lam_max", "acf1", "trunc", "secs", "model_bytes")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
KEY = dict(PARAMS_NU0, preset=PRESET, point="S1-nu0test", theta_gauge=THETA_GAUGE,
           dt_ref=DT_REF, dt_eig=DT_EIG, dt_eig_check=DT_EIG_CHECK,
           j_regimes=REGIMES, j_scan=(60.0, 160.0, len(J_SCAN)), r_scan_max=R_SCAN_MAX,
           n_grid=N_GRID, sn_bracket=list(SN_BRACKET), n_bisect=N_BISECT,
           t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL, tol_fit=TOL_FIT)
CACHE_GAUGE  = hlp.cache_path(STEM, KEY | {"e_gauge": E_GAUGE}, tag="gauge",  root=CACHE_DIR)
CACHE_RETUNE = hlp.cache_path(STEM, KEY, tag="retune", root=CACHE_DIR)
CACHE_EREST  = hlp.cache_path(STEM, KEY | {
    "e_scan": (float(E_SCAN[0]), float(E_SCAN[-1]), len(E_SCAN)), "e_bisect": E_BISECT,
    "j_onset_bracket": list(J_ONSET_BRACKET), "n_bisect_j": N_BISECT_J,
    "delta_s1": DELTA_S1}, tag="erest", root=CACHE_DIR)
# The ladder and the cost get their OWN tags, so the three stages above keep their caches.
_LKEY = KEY | {"dts_l": DTS_L, "dt_fit": DT_FIT, "sides": list(SIDES),
               "e_bisect": E_BISECT,
               "e_scan": (float(E_SCAN[0]), float(E_SCAN[-1]), len(E_SCAN))}
CACHE_LADDER = hlp.cache_path(STEM, _LKEY, tag="ladder", root=CACHE_DIR)
CACHE_COST   = hlp.cache_path(STEM, _LKEY | {"threshs": list(THRESHS),
                                             "sigma_target": SIGMA_TARGET},
                              tag="cost", root=CACHE_DIR)

run_gauge  = True
run_retune = True
run_erest  = True
run_ladder = True
run_cost   = True
run_report = True
overwrite  = False   # the hash covers every parameter, so this is only for a change to the CODE

for _dt in (DT_REF, DT_EIG, DT_EIG_CHECK, *DTS_L):
    for _nu0 in (PARAMS_S1["nu0"], NU0_NEW):
        hzd.check_step(HAZARD, _nu0, _dt)


# ── Measurement machinery (as S1-dt_cost.py's build/measure) ──────────────────

def model(P: dict) -> dict:
    return {k: v for k, v in P.items() if k not in ("dt", "hazard")}


def build(P: dict, J: float, dt: float, Q: int | None = None) -> RDMIsingModel:
    return RDMIsingModel(J, **model(P), dt=dt, hazard=HAZARD,
                         Qm=None if Q is None else [Q], device=DEVICE)


def age_window(P: dict, J: float, dt: float) -> float:
    """ U_max = Qm*dt in MILLISECONDS, fitted at `dt` and then held in ms.

    A physical length rather than a bin count, so the same window is used on both sides of a
    comparison and at every step: refitting per run lays up to `tol` of grid error on top of
    the difference being measured. """
    Qm = hlp.fit_age_grid(lambda q: build(P, J, dt, None if q is None else q[0]),
                          tol=TOL_FIT, equi=hlp.n_steps(T_EQUI + T_MEAS, dt))
    return Qm[0] * dt


def measure(P: dict, J: float, dt: float, Q: int, *, ic_m: float | None = None) -> list[float]:
    """ One (parameter set, J) point: the COLS row, averaged over whole cycles.

    `ic_m` places the model on the stationary state of that overlap before equilibrating --
    the warm start the re-anchored point needs, where the active branch is not reachable from
    the default initial condition and "which attractor" is itself the question. """
    net = build(P, J, dt, Q)
    if ic_m is not None:
        net.set_state(torch.tensor([ic_m], device=DEVICE))
    net.forward(hlp.n_steps(T_EQUI, dt))
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, dt), pb=False)

    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    period = (z - a) * dt / n_cyc if n_cyc else float('nan')

    # The step diagnostics over one period (a tenth of the record where there is none): the
    # peak of a burst is one instant. phi_max saturates at 1 and stops reporting there, so the
    # peak HAZARD rate is tracked beside it -- that is what hazards.step_error wants, and it
    # is not the population rate, which the dead time caps at 1/t_abs.
    n_probe = max(1, hlp.n_steps(period, dt) if n_cyc else len(r) // 10)
    phi_max, lam_max, n_abs = 0.0, 0.0, net.n_abs
    for _ in range(n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        lam_max = max(lam_max, max(float(l[n_abs:].max()) for l in net.hazard_rate()))
        net.update()

    x = r[a:z] - r[a:z].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    trunc = float(net.truncation_error())
    assert trunc < TOL, (f"truncation_error = {trunc:.2e} at J = {J}, dt = {dt}, Q = {Q}: "
                         f"the fitted age window does not hold here.")

    return [float(r[a:z].mean()),
            float(traj["sigma_tot"][a:z].mean()) / dt,
            period,
            float(r[a:z].max() - r[a:z].min()),
            float(Q), phi_max, lam_max, acf1]


# ── 1. The shift-gauge control ────────────────────────────────────────────────

if run_gauge and (overwrite or not CACHE_GAUGE.exists()):
    print(f"\n[gauge] (nu0, theta) = (1, {_S1['theta']:g}) against "
          f"({NU0_NEW:g}, {THETA_GAUGE:.12f}), and against "
          f"(nu0, theta, E) = ({NU0_NEW:g}, {_S1['theta']:g}, {E_GAUGE:.12f}), at dt = {DT_REF:g}")
    print(f"        delta = ln(nu0_new/nu0)/beta = {DELTA:.12f}")
    out: dict = {"delta": DELTA, "theta_gauge": THETA_GAUGE, "cols": COLS, "u_max": {}}
    for name, J in REGIMES.items():
        # The window is fitted on the S1 side and used on BOTH: the two are the same system
        # under this move, so a window refitted per side would be a difference of its own.
        u_max = age_window(PARAMS_S1, J, DT_REF)
        Q = max(1, round(u_max / DT_REF))
        out["u_max"][name] = u_max
        print(f"  [{name}, J = {J:g}] U_max = {u_max:.1f} ms, Q = {Q}")
        for side, P in (("S1", PARAMS_S1), ("gauge", PARAMS_GAUGE),
                        ("gaugeE", PARAMS_GAUGE_E)):
            t0 = time.time()
            out[(name, side)] = measure(P, J, DT_REF, Q)
            d = dict(zip(COLS, out[(name, side)]))
            print(f"    {side:>5}: r = {d['r']:.12f}  sigma/dt = {d['sigma']:.6e}  "
                  f"lam_max = {d['lam_max']:.8e}  ({time.time() - t0:.0f}s)", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(out, CACHE_GAUGE)
    print(f"  saved {CACHE_GAUGE}")


# ── 2. Re-tuning at nu0 = 0.01 with theta held ────────────────────────────────

def branches(J: float, dt: float, Q: int) -> torch.Tensor:
    """ Every fixed point of the re-anchored model at this J, as RATES in 1/ms.

    Scanned over overlaps in [0, R_SCAN_MAX*dt] rather than the whole [0, 1]: the roots that
    matter sit at O(dt * 0.01), and a scan that has to cover saturation as well cannot resolve
    the fold where the active branch is born. """
    net = build(PARAMS_NU0, J, dt, Q)
    roots = net.fixed_points(n_grid=N_GRID, lo=0.0, hi=R_SCAN_MAX * dt)
    return torch.tensor(sorted(float(x) / dt for x in roots))


def spectral_radius(J: float, m: float, dt: float, Q: int) -> float:
    """ max |eig| of the linearized one-step map at the fixed point with overlap m.

    At ONE named root, not the max over all of them. S1-cycle_point.py takes the max because
    S1 has a single root there and the max is then an observation that it does; here the
    structure is bistable and the unstable middle branch would pin the max above 1 at every J,
    which would say nothing about the branch being classified. """
    net = build(PARAMS_NU0, J, dt, Q)
    return float(net.fp_eigenvalues(torch.tensor([m], device=DEVICE)).abs().max())


def n_active(J: float, dt: float, Q: int) -> int:
    """ How many fixed points sit above the quiescent one: 0 below the fold, 2 above it. """
    return max(0, len(branches(J, dt, Q)) - 1)


def active_root(J: float, m0: float, dt: float, Q: int) -> float:
    """ The active branch's rate at J, by Newton from a seed overlap m0.

    Newton rather than a fresh scan: the branch is being followed in J from a point already on
    it, so the seed is good, and a full bracket scan costs a (N_GRID, Q) evaluation per call --
    which a root find in J would pay forty times over for no extra information. """
    net = build(PARAMS_NU0, J, dt, Q)
    roots = net.fixed_points(method='newton',
                             guesses=torch.tensor([[m0]], device=DEVICE))
    return float(roots[-1]) / dt if roots.numel() else float('nan')


if run_retune and (overwrite or not CACHE_RETUNE.exists()):
    print(f"\n[retune] nu0 = {NU0_NEW:g} /ms, theta = {PARAMS_NU0['theta']:g} HELD, "
          f"E = {PARAMS_NU0['E']:g}, dt = {DT_REF:g}")
    # The window, in ms. Two fits and the LARGER of them: the re-anchored model's own
    # attractor at the bottom of the scan (the quiescent branch, the lowest rate on the scan),
    # and S1 at the cycle point, which is the honest proxy for the ACTIVE branch -- the branch
    # the whole exercise is about is not reachable from the default initial condition, so
    # fit_age_grid cannot size a window on it directly, and it carries the S1 operating rate
    # by construction. Taking the max also means both sides of the comparison share one
    # window, which is what makes their Q comparable at all.
    u_quiet  = age_window(PARAMS_NU0, float(J_SCAN[0]), DT_REF)
    u_active = age_window(PARAMS_S1, J_CYCLE, DT_REF)
    u_max = max(u_quiet, u_active)
    print(f"  U_max: {u_quiet:.1f} ms (quiescent, J = {float(J_SCAN[0]):g}) vs "
          f"{u_active:.1f} ms (S1 at the cycle point) -> {u_max:.1f} ms")
    Q     = max(1, round(u_max / DT_REF))
    Q_eig = max(1, round(u_max / DT_EIG))
    Q_chk = max(1, round(u_max / DT_EIG_CHECK))
    print(f"  U_max = {u_max:.1f} ms -> Q = {Q} at dt = {DT_REF:g}, "
          f"Q = {Q_eig} at dt = {DT_EIG:g}")

    res: dict = {"u_max": u_max, "u_quiet": u_quiet, "u_active": u_active,
                 "Q": Q, "Q_eig": Q_eig, "cols": COLS, "J": J_SCAN,
                 "roots": {}, "rho": {}, "rho_check": {}, "attractor": {}}
    for J in J_SCAN.tolist():
        t0 = time.time()
        rts = branches(J, DT_REF, Q)
        res["roots"][J] = rts
        # the spectral radius of the lowest (quiescent) and the highest (active) branch
        res["rho"][J] = {"low": spectral_radius(J, float(rts[0]) * DT_EIG, DT_EIG, Q_eig),
                         "high": (spectral_radius(J, float(rts[-1]) * DT_EIG, DT_EIG, Q_eig)
                                  if len(rts) > 1 else float('nan'))}
        # what the system actually does when started ON the active branch
        res["attractor"][J] = measure(PARAMS_NU0, J, DT_REF, Q,
                                      ic_m=float(rts[-1]) * DT_REF)
        d = dict(zip(COLS, res["attractor"][J]))
        print(f"  J = {J:6.1f}  roots {[f'{float(v):.6f}' for v in rts]}  "
              f"rho(low) = {res['rho'][J]['low']:.4f}  rho(high) = {res['rho'][J]['high']:.4f}"
              f"  -> attractor r = {d['r']:.6f}, ptp = {d['ptp']:.6f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # The classification at DT_EIG is a cost choice; check it does not move at DT_EIG/2 on the
    # two J either side of the middle of the scan that carry an active branch.
    for J in [j for j in J_SCAN.tolist() if len(res["roots"][j]) > 1][:2]:
        rts = res["roots"][J]
        res["rho_check"][J] = {
            "low":  spectral_radius(J, float(rts[0]) * DT_EIG_CHECK, DT_EIG_CHECK, Q_chk),
            "high": spectral_radius(J, float(rts[-1]) * DT_EIG_CHECK, DT_EIG_CHECK, Q_chk)}
        print(f"  [rho check at dt = {DT_EIG_CHECK:g}] J = {J:g}: {res['rho_check'][J]}",
              flush=True)

    # The fold, by bisection on whether an active branch exists at all.
    lo, hi = SN_BRACKET
    if not (n_active(lo, DT_REF, Q) == 0 < n_active(hi, DT_REF, Q)):
        raise ValueError(f"bracket {SN_BRACKET} does not straddle the fold")
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if n_active(mid, DT_REF, Q) == 0: lo = mid
        else:                             hi = mid
    res["J_sn"] = 0.5 * (lo + hi)
    print(f"  fold (active branch appears) at J = {res['J_sn']:.6f}", flush=True)

    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(res, CACHE_RETUNE)
    print(f"  saved {CACHE_RETUNE}")


# ── 3. Restoring the operating rate with E, at J held ─────────────────────────

def P_E(E: float) -> dict:
    """ The re-anchored point with E as the free knob. Through `presets.params` so a typo in a
    parameter name raises here rather than appearing in the figure. """
    return pre.params(PRESET, hazard=HAZARD, dt=DT_REF, nu0=NU0_NEW, E=E)


def r_of(E: float, J: float, Q: int) -> float:
    """ The rate of the attractor ACTUALLY REACHED at (J, E) from the default initial condition.

    Not a branch solved for: the whole point of stage 2 was that the re-anchored model has a
    fixed point carrying the right rate which the system cannot sit on. """
    return dict(zip(COLS, measure(P_E(E), J, DT_REF, Q)))["r"]


def fp_rho(P: dict, J: float, dt: float, Q: int) -> tuple[float, float, int]:
    """ (spectral radius, rate of that fixed point in 1/ms, number of fixed points found).

    The LOWEST root: with E restoring the drive the model is no longer bistable on the window
    scanned, and the root count is returned so that assumption is reported rather than made.
    """
    net = build(P, J, dt, Q)
    rts = net.fixed_points(n_grid=N_GRID, lo=0.0, hi=R_SCAN_MAX * dt)
    if rts.numel() == 0:
        return float('nan'), float('nan'), 0
    m = float(sorted(float(x) for x in rts)[0])
    rho = float(net.fp_eigenvalues(torch.tensor([m], device=DEVICE)).abs().max())
    return rho, m / dt, len(rts)


if run_erest and (overwrite or not CACHE_EREST.exists()):
    print(f"\n[erest] nu0 = {NU0_NEW:g} /ms, theta = {PARAMS_NU0['theta']:g} HELD, "
          f"J on S1's own axis ({J_FP:g}, {J_CYCLE:g}), E free. dt = {DT_REF:g}")
    G0 = torch.load(CACHE_GAUGE, weights_only=False)   # S1's own targets, measured, not typed
    targets = {n: dict(zip(COLS, G0[(n, "S1")]))["r"] for n in REGIMES}

    EO: dict = {"cols": COLS, "E_scan": E_SCAN, "targets": targets,
                "scan": {}, "u": {}, "tuned": {}, "onset": {}}

    for name, J in REGIMES.items():
        tgt = targets[name]
        # The window, in ms, HELD across the whole scan and the whole bisection at this J:
        # the larger of S1's own fit and the re-anchored model's at the quietest E on the scan
        # (low E is the long window). Refitting per E would lay grid error on top of the r(E)
        # being root-found.
        u_s1 = age_window(PARAMS_S1, J, DT_REF)
        u_lo = age_window(P_E(float(E_SCAN[0])), J, DT_REF)
        u_max = max(u_s1, u_lo)
        Q = max(1, round(u_max / DT_REF))
        EO["u"][name] = {"s1": u_s1, "lo_E": u_lo, "held": u_max, "Q": Q}
        print(f"  [{name}, J = {J:g}] target r = {tgt:.12f} /ms; U_max {u_s1:.1f} ms (S1) vs "
              f"{u_lo:.1f} ms (E = {float(E_SCAN[0]):g}) -> held {u_max:.1f} ms, Q = {Q}",
              flush=True)

        rs = []
        for E in E_SCAN.tolist():
            t0 = time.time()
            rs.append(r_of(E, J, Q))
            print(f"    E = {E:5.2f}  r = {rs[-1]:.8f}  ({time.time() - t0:.0f}s)", flush=True)
        rs_t = torch.tensor(rs)
        EO["scan"][name] = rs_t

        # every crossing of the target on the scan, so a second root is seen and not stepped over
        d = rs_t - tgt
        cross = [i for i in range(len(rs) - 1) if float(d[i]) * float(d[i + 1]) <= 0]
        print(f"    crossings of the target on the scan: {len(cross)} "
              f"at E in {[(float(E_SCAN[i]), float(E_SCAN[i+1])) for i in cross]}")
        if not cross:
            raise ValueError(f"the target r = {tgt:g} is not bracketed by the E scan at J = {J}")

        lo, hi = float(E_SCAN[cross[0]]), float(E_SCAN[cross[0] + 1])
        for _ in range(E_BISECT):
            mid = 0.5 * (lo + hi)
            if r_of(mid, J, Q) < tgt: lo = mid
            else:                     hi = mid
        E_star = 0.5 * (lo + hi)
        row = measure(P_E(E_star), J, DT_REF, Q)
        rho, r_fp, n_fp = fp_rho(P_E(E_star), J, DT_EIG, max(1, round(u_max / DT_EIG)))
        rho_c, _, n_fp_c = fp_rho(P_E(E_star), J, DT_EIG_CHECK,
                                  max(1, round(u_max / DT_EIG_CHECK)))
        # the window a fit would ask for AT the restored state: the comparable grid demand,
        # against S1's own. Not the window the row above was measured on, which is the held one.
        u_star = age_window(P_E(E_star), J, DT_REF)
        EO["tuned"][name] = {"E": E_star, "row": row, "n_cross": len(cross),
                             "rho": rho, "r_fp": r_fp, "n_fp": n_fp,
                             "rho_check": rho_c, "n_fp_check": n_fp_c,
                             "u_fit": u_star, "Q_fit": max(1, round(u_star / DT_REF))}
        dd = dict(zip(COLS, row))
        print(f"    -> E = {E_star:.9f}: r = {dd['r']:.12f} (target {tgt:.12f}, "
              f"rel {abs(dd['r']-tgt)/tgt:.2e}), ptp = {dd['ptp']:.6e}, "
              f"lam_max = {dd['lam_max']:.6g}, phi_max = {dd['phi_max']:.6g}", flush=True)
        print(f"       rho = {rho:.6f} at dt = {DT_EIG:g} ({n_fp} fixed point(s)); "
              f"{rho_c:.6f} at dt = {DT_EIG_CHECK:g} ({n_fp_c}); "
              f"U_max refitted at this E = {u_star:.1f} ms", flush=True)

    # The re-anchored Hopf, as a cut in J at the CYCLE-tuned E.
    E_c = EO["tuned"]["cycle"]["E"]
    u_on = EO["u"]["cycle"]["held"]
    Q_on = max(1, round(u_on / DT_EIG))
    lo, hi = J_ONSET_BRACKET
    r_lo, r_hi = fp_rho(P_E(E_c), lo, DT_EIG, Q_on)[0], fp_rho(P_E(E_c), hi, DT_EIG, Q_on)[0]
    print(f"\n  [onset] rho at E = {E_c:.9f}: {r_lo:.4f} at J = {lo:g}, {r_hi:.4f} at J = {hi:g}")
    EO["onset"]["bracket_rho"] = (r_lo, r_hi)
    if not (r_lo < 1.0 < r_hi):
        EO["onset"]["J_c"] = float('nan')
        print("  [onset] the bracket does not straddle rho = 1; no onset located")
    else:
        for _ in range(N_BISECT_J):
            mid = 0.5 * (lo + hi)
            if fp_rho(P_E(E_c), mid, DT_EIG, Q_on)[0] < 1.0: lo = mid
            else:                                            hi = mid
        J_c = 0.5 * (lo + hi)
        EO["onset"]["J_c"] = J_c
        EO["onset"]["rho_check"] = fp_rho(P_E(E_c), J_c, DT_EIG_CHECK,
                                          max(1, round(u_on / DT_EIG_CHECK)))[0]
        print(f"  [onset] re-anchored Hopf at J = {J_c:.6f} (rho = 1, dt = {DT_EIG:g}); "
              f"J = {J_CYCLE:g} sits {J_CYCLE - J_c:.3f} past it (S1: {DELTA_S1:.3f})")
        J_m = J_c + DELTA_S1
        EO["onset"]["J_matched"] = J_m
        EO["onset"]["row_matched"] = measure(P_E(E_c), J_m, DT_REF,
                                             max(1, round(u_on / DT_REF)))
        dm = dict(zip(COLS, EO["onset"]["row_matched"]))
        print(f"  [onset] matched-Delta point J = {J_m:.4f} at the same E: r = {dm['r']:.8f}, "
              f"ptp = {dm['ptp']:.6f}, lam_max = {dm['lam_max']:.6g}, "
              f"phi_max = {dm['phi_max']:.6g}", flush=True)

    # the rho(J) profile at the tuned E, for the figure and for reading the onset off a curve
    J_PROF = torch.arange(4.0, 20.1, 1.0)
    EO["onset"]["J_prof"] = J_PROF
    EO["onset"]["rho_prof"] = [fp_rho(P_E(E_c), float(j), DT_EIG, Q_on)[0]
                               for j in J_PROF]

    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(EO, CACHE_EREST)
    print(f"  saved {CACHE_EREST}")


# ── 4. The dt ladder: lam_max, max Phi, and the orders on ptp ─────────────────
#
# E IS TUNED ONCE AT DT_REF = 0.25 AND HELD ACROSS THE WHOLE LADDER. A dt -> 0 limit is taken
# at FIXED model parameters: re-tuning E at every rung would change the model at every rung
# and make Richardson meaningless, and CLAUDE.md is explicit that model parameters do not move
# when dt does. The price is that the realised r drifts along the ladder -- so does S1's own,
# which is the like-for-like comparison, and both drifts are reported.
#
# The age window is fitted ONCE per (side, regime) at DT_FIT and HELD IN MS down the ladder:
# refitting per dt would lay grid error on top of the dt differences being measured.

def measure_full(P: dict, J: float, dt: float, Q: int) -> list[float]:
    """ One rung: the COLS_L row, averaged over whole cycles.

    `measure` above with H carried TWICE -- raw, which diverges as r ln(1/dt) and is the
    control (increment per halving -> r ln 2), and relative to the Poisson reference, which is
    finite as dt -> 0 and is what compares across steps. sigma is identical either way, both
    halves subtracting the same H_ref, which is the check on the subtraction. """
    t0  = time.time()
    net = build(P, J, dt, Q)
    net.forward(hlp.n_steps(T_EQUI, dt))
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, dt), pb=False)

    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    period = (z - a) * dt / n_cyc if n_cyc else float('nan')
    href = epr.poisson_reference_entropy(traj["m_tot"][a:z], P["nu0"], dt)

    n_probe = max(1, hlp.n_steps(period, dt) if n_cyc else len(r) // 10)
    phi_max, lam_max, n_abs = 0.0, 0.0, net.n_abs
    for _ in range(n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        lam_max = max(lam_max, max(float(l[n_abs:].max()) for l in net.hazard_rate()))
        net.update()

    x = r[a:z] - r[a:z].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    trunc = float(net.truncation_error())
    assert trunc < TOL, (f"truncation_error = {trunc:.2e} at J = {J}, dt = {dt}, Q = {Q}: "
                         f"the window fitted at dt = {DT_FIT} does not hold at this step.")

    return [float(r[a:z].mean()),
            float(traj["sigma_tot"][a:z].mean()) / dt,
            float(traj["H_fwd_tot"][a:z].mean()) / dt,
            float(traj["H_rev_tot"][a:z].mean()) / dt,
            float((traj["H_fwd_tot"][a:z] - href).mean()) / dt,
            float((traj["H_rev_tot"][a:z] - href).mean()) / dt,
            period,
            float(r[a:z].max() - r[a:z].min()),
            float(Q), phi_max, lam_max, acf1, trunc, time.time() - t0,
            float(hlp.epr_peak_bytes(1, Q, inplace=False))]


def side_params(side: str, E_star: float) -> dict:
    """ The parameter dict of one side of the comparison. `S1` is the control, `re` is the
    re-anchored point at the E tuned at DT_REF in the `erest` stage and held. """
    return PARAMS_S1 if side == "S1" else P_E(E_star)


if run_ladder and (overwrite or not CACHE_LADDER.exists()):
    if not CACHE_EREST.exists():
        raise FileNotFoundError(f"{CACHE_EREST} not found: the ladder runs at the E that "
                                f"stage 3 tunes. Run the erest stage first.")
    EO0 = torch.load(CACHE_EREST, weights_only=False)
    E_TUNED = {n: EO0["tuned"][n]["E"] for n in REGIMES}
    print(f"\n[ladder] dt in {DTS_L}, both regimes, both sides.")
    print(f"         E tuned at dt = {DT_REF:g} and HELD: "
          + ", ".join(f"{n} {E_TUNED[n]:.9f}" for n in REGIMES))

    L: dict = {"cols": COLS_L, "dts": DTS_L, "E": E_TUNED, "u_max": {}, "Q_fit": {}}
    for name, J in REGIMES.items():
        for side in SIDES:
            P = side_params(side, E_TUNED[name])
            u_max = age_window(P, J, DT_FIT)
            L["u_max"][(side, name)] = u_max
            L["Q_fit"][(side, name)] = max(1, round(u_max / DT_FIT))
            print(f"\n  [{side}, {name}, J = {J:g}] U_max fitted at dt = {DT_FIT:g}, "
                  f"tol = {TOL_FIT:g}: {u_max:.1f} ms -> held in ms down the ladder")
            rows = []
            for dt in DTS_L:
                Q = max(1, round(u_max / dt))
                row = measure_full(P, J, dt, Q)
                rows.append(row)
                d = dict(zip(COLS_L, row))
                print(f"    dt = {dt:8.5f}  Q = {int(d['Q']):5d}  r = {d['r']:.8f}  "
                      f"sigma = {d['sigma']:.6e}  ptp = {d['ptp']:.6f}  "
                      f"maxPhi = {d['phi_max']:.4f}  lam = {d['lam_max']:.4f}  "
                      f"({d['secs']:.0f}s, {d['model_bytes'] / 2**30:.2f} GiB)", flush=True)
            L[(side, name)] = torch.tensor(rows)
            CACHE_DIR.mkdir(exist_ok=True)
            tmp = CACHE_LADDER.with_suffix(".tmp"); torch.save(L, tmp); tmp.replace(CACHE_LADDER)
    print(f"  saved {CACHE_LADDER}")


# ── 5. The cost: the dt and the memory that buy 1% sigma ──────────────────────
#
# Derived from the ladder, through the same helpers.converged_at machinery S1-dt_cost.py uses
# so the two tables read the same way: the COARSEST dt whose relative error against the
# Richardson limit stays inside the tolerance ALL THE WAY DOWN. Where the ladder does not
# reach the tolerance, the step is EXTRAPOLATED by a log-log fit of the relative error against
# dt, and is labelled as an extrapolation wherever it is printed.

def power_fit(x: list[float], y: list[float]) -> tuple[float, float]:
    """ (C, p) of y = C x^p by least squares in logs. Two points give the line through them. """
    lx = torch.tensor([math.log(v) for v in x])
    ly = torch.tensor([math.log(v) for v in y])
    n  = len(x)
    p  = float(((lx * ly).mean() - lx.mean() * ly.mean())
               / ((lx * lx).mean() - lx.mean() ** 2))
    return float(math.exp(float(ly.mean()) - p * float(lx.mean()))), p


def dt_for(err: list[float], dts: list[float], target: float, n_fit: int = 3
           ) -> tuple[float, float]:
    """ (dt that reaches `target` relative error, the fitted order), from the finest `n_fit`
    rungs. Used only when the ladder itself does not reach the target. """
    C, p = power_fit(dts[-n_fit:], err[-n_fit:])
    return (target / C) ** (1.0 / p), p


if run_cost and (overwrite or not CACHE_COST.exists()):
    if not CACHE_LADDER.exists():
        raise FileNotFoundError(f"{CACHE_LADDER} not found. Run the ladder stage first.")
    L = torch.load(CACHE_LADDER, weights_only=False)
    cl = {k: i for i, k in enumerate(L["cols"])}
    C: dict = {"limits": {}, "orders": {}, "converged": {}, "sigma_cost": {}}
    for name in REGIMES:
        for side in SIDES:
            t = L[(side, name)]
            for key in QUANTITIES[name]:
                v = t[:, cl[key]]
                C["limits"][(side, name, key)] = richardson(v)
                C["orders"][(side, name, key)] = orders(v).tolist()
                lim = C["limits"][(side, name, key)][0]
                C["converged"][(side, name, key)] = {
                    th: (converged_at(DTS_L, v, lim, th) if lim == lim else float('nan'))
                    for th in THRESHS}
            # the deliverable: the dt, the grid and the memory that buy SIGMA_TARGET on sigma
            if "sigma" not in QUANTITIES[name]:
                continue
            v = t[:, cl["sigma"]]
            lim, p_rich = C["limits"][(side, name, "sigma")]
            u_max = L["u_max"][(side, name)]
            entry: dict = {"limit": lim, "p_richardson": p_rich, "u_max": u_max}
            if lim == lim:
                err = [abs(float(x) - lim) / abs(lim) for x in v]
                entry["err"] = err
                on_ladder = converged_at(DTS_L, v, lim, SIGMA_TARGET)
                entry["on_ladder"] = on_ladder
                if on_ladder == on_ladder:
                    entry["dt"], entry["extrapolated"], entry["p_err"] = on_ladder, False, \
                        power_fit(DTS_L, err)[1]
                else:
                    dt_e, p_err = dt_for(err, DTS_L, SIGMA_TARGET)
                    # report it on the admissible ladder: the coarsest 1/2^k at or below it
                    k = math.ceil(math.log2(1.0 / dt_e))
                    entry["dt_raw"], entry["dt"] = dt_e, 2.0 ** (-k)
                    entry["extrapolated"], entry["p_err"] = True, p_err
                Q = max(1, round(u_max / entry["dt"]))
                entry["Q"] = Q
                entry["bytes"] = int(hlp.epr_peak_bytes(1, Q, inplace=False))
                # wall time, extrapolated from the ladder's own timings (secs ~ C dt^-q)
                cs, q = power_fit(DTS_L, [float(x) for x in t[:, cl["secs"]]])
                entry["secs"], entry["q_time"] = cs * entry["dt"] ** q, q
            C["sigma_cost"][(side, name)] = entry
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(C, CACHE_COST)
    print(f"\n[cost] saved {CACHE_COST}")



# ── Report ────────────────────────────────────────────────────────────────────

def rel(a: float, b: float, floor: float = 1e-300) -> float:
    """ |a - b| / |b| where b is big enough to divide by, and the absolute difference where it
    is not -- sigma at a fixed point is identically zero and has no relative error. """
    return abs(a - b) / abs(b) if abs(b) > floor else abs(a - b)


if run_report:
    for f in (CACHE_GAUGE, CACHE_RETUNE, CACHE_EREST, CACHE_LADDER, CACHE_COST):
        if not f.exists(): raise FileNotFoundError(f"{f} not found. Run the stages first.")
    G = torch.load(CACHE_GAUGE, weights_only=False)
    R = torch.load(CACHE_RETUNE, weights_only=False)
    EO = torch.load(CACHE_EREST, weights_only=False)
    col = {k: i for i, k in enumerate(COLS)}

    print("\n" + "=" * 92)
    print(f"S1 nu0 re-anchoring, step A -- preset {PRESET}, hazard {HAZARD}, "
          f"t_abs = {PARAMS_S1['t_abs']:g} ms, dt = {DT_REF:g}")
    print("=" * 92)

    # 1. the control
    print(f"\n[1] SHIFT-GAUGE CONTROL. (nu0, theta) = (1, {_S1['theta']:g}) against "
          f"({NU0_NEW:g}, {G['theta_gauge']:.12f}),")
    print(f"    delta = {G['delta']:.12f}. The hazard is unchanged by the move, so every")
    print(f"    column must agree to solver error. A disagreement is the implementation.")
    keys = ("r", "sigma", "period", "ptp", "phi_max", "lam_max")
    # A mixed criterion, not a relative one. sigma at a fixed point is identically zero and is
    # realised at the cancellation floor (~1e-13 nats/ms here), where a RELATIVE difference
    # divides one rounding by another and means nothing -- it read 1.8e-06 on an absolute
    # difference of 1.2e-18 and called a passing control a failure. ATOL is set at the floor,
    # a decade above the largest |sigma| either side realises at a fixed point.
    RTOL, ATOL = 1e-10, 1e-12
    worst_rel = worst_abs = 0.0
    failures: list[str] = []
    for side in ("gauge", "gaugeE"):
        label = {"gauge": "theta-shift twin", "gaugeE": "E-shift twin"}[side]
        print(f"\n    --- {label} ---")
        for name, J in REGIMES.items():
            a = dict(zip(COLS, G[(name, "S1")]))
            b = dict(zip(COLS, G[(name, side)]))
            print(f"\n    [{name}, J = {J:g}]  U_max = {G['u_max'][name]:.1f} ms, "
                  f"Q = {int(a['Q'])}")
            print(f"    {'column':>10}{'S1':>22}{'twin':>22}{'rel. diff':>14}{'abs. diff':>14}")
            for k in keys:
                dr, da = rel(b[k], a[k]), abs(b[k] - a[k])
                ok = (da <= ATOL) or (dr == dr and dr < RTOL)
                # Only the theta arm is the gauge; the E arm is expected to differ and is
                # reported, not judged. A theta-arm failure is the implementation.
                if not ok and side == "gauge": failures.append(f"{side}/{name}/{k}")
                worst_rel = max(worst_rel, dr if dr == dr else 0.0)
                worst_abs = max(worst_abs, da if da == da else 0.0)
                print(f"    {k:>10}{a[k]:>22.14g}{b[k]:>22.14g}{dr:>14.3e}{da:>14.3e}"
                      f"{'' if ok else '   <-- FAIL'}")
    print(f"\n    worst relative difference: {worst_rel:.3e}   worst absolute: {worst_abs:.3e}")
    print(f"    criterion (theta arm only): |a-b| <= {ATOL:g} OR rel < {RTOL:g}, per column "
          f"-- {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    print("    The theta arm is the gauge and must pass. The E arm is NOT the gauge and is")
    print("    EXPECTED to fail: S is carried per age with S(0) = 0, so a constant E reaches")
    print("    the field as E*(1-(1-alpha)^n) and does not shift it uniformly as theta does.")
    print("    That is what makes E a legitimate knob for restoring the rate; see the docstring.")

    # 2. the re-anchored point
    print(f"\n[2] RE-ANCHORED POINT `S1-nu0test`: nu0 = {NU0_NEW:g} /ms, "
          f"theta = {PARAMS_NU0['theta']:g} HELD, E = {PARAMS_NU0['E']:g}.")
    print(f"    AGE GRID, both sides at tol = {TOL_FIT:g}, dt = {DT_REF:g}:")
    for n, J in REGIMES.items():
        print(f"      S1         [{n}, J = {J:g}]  U_max = {G['u_max'][n]:7.1f} ms  "
              f"Q = {int(dict(zip(COLS, G[(n, 'S1')]))['Q'])}")
    print(f"      S1-nu0test [quiescent, J = {float(R['J'][0]):g}]  "
          f"U_max = {R['u_quiet']:7.1f} ms")
    print(f"      used for the whole retune stage: U_max = {R['u_max']:.1f} ms, "
          f"Q = {R['Q']} (the larger of the two fits; see the stage).")
    print(f"    Fixed-point branches, their linear stability (rho at dt = {DT_EIG:g}), and the")
    print(f"    attractor reached from a warm start ON the active branch.")
    print(f"\n    {'J':>7}{'r_quiet':>11}{'r_mid':>11}{'r_active':>11}"
          f"{'rho_quiet':>11}{'rho_active':>12}{'attractor r':>14}{'ptp':>11}{'acf1':>8}")
    Js = R["J"].tolist()
    for J in Js:
        rts = R["roots"][J].tolist()
        a = dict(zip(COLS, R["attractor"][J]))
        cells = [f"{rts[0]:>11.6f}",
                 f"{rts[1]:>11.6f}" if len(rts) > 2 else f"{'--':>11}",
                 f"{rts[-1]:>11.6f}" if len(rts) > 1 else f"{'--':>11}"]
        print(f"    {J:>7.1f}" + "".join(cells)
              + f"{R['rho'][J]['low']:>11.4f}{R['rho'][J]['high']:>12.4f}"
              + f"{a['r']:>14.6f}{a['ptp']:>11.6f}{a['acf1']:>8.3f}")
    print(f"\n    fold: the active branch appears at J = {R['J_sn']:.4f} "
          f"(bisection on root count, dt = {DT_REF:g})")
    for J, v in R["rho_check"].items():
        print(f"    rho at dt = {DT_EIG_CHECK:g}, J = {J:g}: quiet {v['low']:.4f}, "
              f"active {v['high']:.4f}  (classification unchanged from dt = {DT_EIG:g})")

    # 3. the match
    print(f"\n[3] MATCHING THE S1 OPERATING RATE. Targets are S1's own r measured in this")
    print(f"    script at dt = {DT_REF:g}, not quoted from a dump.")
    targets = {n: dict(zip(COLS, G[(n, "S1")]))["r"] for n in REGIMES}
    match: dict = {}
    for name, J in REGIMES.items():
        tgt = targets[name]
        # (i) can the ATTRACTOR be brought to the target by J alone?
        att = torch.tensor([dict(zip(COLS, R["attractor"][j]))["r"] for j in Js])
        bracketed = bool(((att - tgt)[:-1] * (att - tgt)[1:] < 0).any())
        # (ii) the J at which the ACTIVE BRANCH's own fixed-point rate equals the target --
        #      well defined whatever that branch's stability, by linear interpolation on the
        #      scan and then bisection on the branch rate.
        act = [(j, float(R["roots"][j][-1])) for j in Js if len(R["roots"][j]) > 1]
        J_b, v_b = float('nan'), float('nan')
        for (j1, v1), (j2, v2) in zip(act[:-1], act[1:]):
            if (v1 - tgt) * (v2 - tgt) <= 0:
                lo, hi = j1, j2
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    vm = active_root(mid, v2 * DT_REF, DT_REF, R["Q"])
                    if (v1 - tgt) * (vm - tgt) <= 0: hi = mid
                    else:                            lo = mid
                J_b = 0.5 * (lo + hi)
                v_b = active_root(J_b, v2 * DT_REF, DT_REF, R["Q"])
                break
        match[name] = {"target": tgt, "J_branch": J_b, "r_branch": v_b,
                       "bracketed": bracketed}
        print(f"\n    [{name}] S1 r = {tgt:.8f} /ms at J = {J:g}")
        print(f"      attractor r at nu0 = {NU0_NEW:g} over J in "
              f"[{Js[0]:g}, {Js[-1]:g}]: {float(att.min()):.6f} to {float(att.max()):.6f} /ms")
        print(f"      target bracketed by the attractor rate? "
              f"{'YES' if bracketed else 'NO -- J alone does not reach it'}")
        if J_b == J_b:
            print(f"      the ACTIVE BRANCH carries r = {tgt:.8f} at J = {J_b:.4f} "
                  f"(branch r there = {v_b:.8f}, residual {rel(v_b, tgt):.2e})")
            print(f"      rho of that branch = "
                  f"{spectral_radius(J_b, v_b * DT_EIG, DT_EIG, R['Q_eig']):.4f}"
                  f"   (> 1 means it is NOT an operating point)")
            print(f"      distance past the fold: J - J_sn = {J_b - R['J_sn']:.4f}  "
                  f"(S1: J - J_c = {J - 11.534:.3f} past its Hopf at 11.534)")
        else:
            print(f"      the active branch does not carry this rate anywhere on the scan")

    # 4. the gauge invariants
    print(f"\n[4] GAUGE-INVARIANT COMBINATIONS (presets.invariants). lam_op is the PEAK hazard")
    print(f"    rate read off the attractor (COLS 'lam_max'), not a mean.")
    for name, J in REGIMES.items():
        s1 = dict(zip(COLS, G[(name, "S1")]))
        print(f"\n    S1          [{name}, J = {J:g}]")
        print(f"      {pre.invariants_line(dict(PARAMS_S1), lam_op=s1['lam_max'])}")
    for name in REGIMES:
        J_b = match[name]["J_branch"]
        if J_b != J_b: continue
        a = dict(zip(COLS, R["attractor"][min(Js, key=lambda j: abs(j - J_b))]))
        print(f"\n    S1-nu0test  [{name}-matched branch, J = {J_b:.4f}; lam_op from the")
        print(f"                 attractor actually reached at the nearest scanned J]")
        print(f"      {pre.invariants_line(dict(PARAMS_NU0), lam_op=a['lam_max'])}")

    # 5. step diagnostics on the re-anchored scan
    print(f"\n[5] STEP DIAGNOSTICS on the re-anchored attractors at dt = {DT_REF:g}")
    phis = [dict(zip(COLS, R["attractor"][j]))["phi_max"] for j in Js]
    lams = [dict(zip(COLS, R["attractor"][j]))["lam_max"] for j in Js]
    acfs = [dict(zip(COLS, R["attractor"][j]))["acf1"] for j in Js]
    print(f"    max over J of max_step_prob: {max(phis):.3e}")
    print(f"    max HAZARD rate lam:         {max(lams):.4g} /ms  -> lam*dt/2 = "
          f"{hzd.step_error(max(lams), DT_REF):.3e}")
    print(f"    min lag-1 autocorrelation:   {min(acfs):.4f}  "
          f"(-1 would be the alternate-bin artifact)")


    # 6. restoring the rate with E, at J held
    print(f"\n[6] RESTORING THE OPERATING RATE WITH E, at J held on S1's own axis values.")
    print(f"    theta = {PARAMS_NU0['theta']:g} is NEVER moved; E is the knob, and the control")
    print(f"    in [1] is what says E is not that relabel. r(E) is the rate of the attractor")
    print(f"    ACTUALLY REACHED, root-found by bisection inside the scan cell that brackets.")
    for name, J in REGIMES.items():
        t = EO["tuned"][name]; u = EO["u"][name]
        s1 = dict(zip(COLS, G[(name, "S1")]))
        re_ = dict(zip(COLS, t["row"]))
        tgt = EO["targets"][name]
        print(f"\n    [{name}, J = {J:g}]  E = {t['E']:.9f}   "
              f"(crossings of the target on the E scan: {t['n_cross']})")
        print(f"      r = {re_['r']:.12f} /ms vs target {tgt:.12f} -> "
              f"residual {rel(re_['r'], tgt):.2e}")
        print(f"      age grid: held U_max = {u['held']:.1f} ms, Q = {u['Q']} "
              f"(max of S1's {u['s1']:.1f} ms and the E = {float(EO['E_scan'][0]):g} fit "
              f"{u['lo_E']:.1f} ms);")
        print(f"                refitted AT the restored state: U_max = {t['u_fit']:.1f} ms, "
              f"Q = {t['Q_fit']}  <- the comparable grid demand")
        print(f"      rho = {t['rho']:.6f} at dt = {DT_EIG:g} ({t['n_fp']} fixed point(s) on "
              f"the scan window), {t['rho_check']:.6f} at dt = {DT_EIG_CHECK:g} "
              f"-> {'STABLE fixed point' if t['rho'] < 1 else 'fixed point UNSTABLE (cycle)'}")
        print(f"      {'column':>10}{'S1':>22}{'S1-nu0test + E':>22}{'ratio':>12}")
        for k in COLS:
            ratio = (re_[k] / s1[k]) if abs(s1[k]) > 1e-300 else float('nan')
            print(f"      {k:>10}{s1[k]:>22.12g}{re_[k]:>22.12g}{ratio:>12.4g}")

    print(f"\n    HEADLINE (the number the brief exists for), at the same dt = {DT_REF:g}:")
    for name, J in REGIMES.items():
        s1 = dict(zip(COLS, G[(name, "S1")])); re_ = dict(zip(COLS, EO["tuned"][name]["row"]))
        for k, lbl in (("lam_max", "peak hazard lam_max [1/ms]"), ("phi_max", "max Phi")):
            verdict = ("LOWER" if re_[k] < s1[k] * (1 - 1e-6) else
                       "HIGHER" if re_[k] > s1[k] * (1 + 1e-6) else "EQUAL")
            print(f"      [{name:>5}] {lbl:>26}: S1 {s1[k]:.6g}  ->  re-anchored "
                  f"{re_[k]:.6g}   ({re_[k] / s1[k]:.4g}x, {verdict})")

    print(f"\n    OSCILLATION. ptp and the cycle count say whether the restored state moves;")
    print(f"    rho says whether its fixed point is a repeller.")
    for name, J in REGIMES.items():
        re_ = dict(zip(COLS, EO["tuned"][name]["row"]))
        s1 = dict(zip(COLS, G[(name, "S1")]))
        print(f"      [{name:>5}, J = {J:g}] ptp = {re_['ptp']:.6e} /ms "
              f"(S1 {s1['ptp']:.6e}), period = {re_['period']:.4g} ms "
              f"(S1 {s1['period']:.4g}), rho = {EO['tuned'][name]['rho']:.4f}")

    O = EO["onset"]
    print(f"\n    THE RE-ANCHORED ONSET, as a cut in J at the cycle-tuned "
          f"E = {EO['tuned']['cycle']['E']:.9f}.")
    print(f"    J and not E, because S1's own reference (J_c = {J_C_S1:g}, so J = {J_CYCLE:g}")
    print(f"    sits {DELTA_S1:.3f} past onset) is a distance along the J axis.")
    if O["J_c"] == O["J_c"]:
        print(f"      rho = 1 at J = {O['J_c']:.6f} (dt = {DT_EIG:g}; "
              f"rho at that J and dt = {DT_EIG_CHECK:g} is {O['rho_check']:.4f})")
        print(f"      J = {J_CYCLE:g} sits Delta = {J_CYCLE - O['J_c']:.3f} past it, "
              f"against S1's Delta = {DELTA_S1:.3f}")
        m = dict(zip(COLS, O["row_matched"]))
        print(f"      matched-Delta point J = {O['J_matched']:.4f} at the same E "
              f"(so r is NOT the restored one there):")
        print(f"        r = {m['r']:.8f} /ms, ptp = {m['ptp']:.6f}, period = {m['period']:.4g} "
              f"ms, lam_max = {m['lam_max']:.6g}, max Phi = {m['phi_max']:.6g}, "
              f"acf1 = {m['acf1']:.3f}")
    else:
        print(f"      the bracket {J_ONSET_BRACKET} does not straddle rho = 1 "
              f"(rho = {O['bracket_rho'][0]:.4f} and {O['bracket_rho'][1]:.4f}); "
              f"no onset located")

    print(f"\n    GAUGE INVARIANTS of the restored points (E != 0, so this is no longer")
    print(f"    `{PRESET}` + one override -- it is {PRESET} + nu0 + E, stated as such).")
    for name, J in REGIMES.items():
        s1 = dict(zip(COLS, G[(name, "S1")])); t = EO["tuned"][name]
        re_ = dict(zip(COLS, t["row"]))
        print(f"      S1         [{name}, J = {J:g}]        "
              f"{pre.invariants_line(dict(PARAMS_S1), lam_op=s1['lam_max'])}")
        print(f"      +nu0,+E    [{name}, J = {J:g}, E = {t['E']:.6f}]  "
              f"{pre.invariants_line(dict(P_E(t['E'])), lam_op=re_['lam_max'])}")

    print(f"\n    STEP DIAGNOSTICS at dt = {DT_REF:g} (admissible steps divide t_abs = 1 ms).")
    for name, J in REGIMES.items():
        re_ = dict(zip(COLS, EO["tuned"][name]["row"]))
        s1 = dict(zip(COLS, G[(name, "S1")]))
        print(f"      [{name:>5}] max Phi = {re_['phi_max']:.4f} (S1 {s1['phi_max']:.4f})   "
              f"lam*dt/2 = {hzd.step_error(re_['lam_max'], DT_REF):.4f} "
              f"(S1 {hzd.step_error(s1['lam_max'], DT_REF):.4f})   "
              f"acf1 = {re_['acf1']:.4f} (S1 {s1['acf1']:.4f})")

    # 7. the dt ladder
    L = torch.load(CACHE_LADDER, weights_only=False)
    CO = torch.load(CACHE_COST, weights_only=False)
    cl = {k: i for i, k in enumerate(L["cols"])}

    print(f"\n[7] THE dt LADDER. E was tuned ONCE at dt = {DT_REF:g} and is HELD at every rung")
    print(f"    (fp {L['E']['fp']:.9f}, cycle {L['E']['cycle']:.9f}): a dt -> 0 limit is taken")
    print(f"    at fixed model parameters. The realised r therefore DRIFTS down the ladder on")
    print(f"    both sides, and both drifts are in the tables -- that is the like-for-like")
    print(f"    comparison. The age window is fitted once per (side, regime) at dt = {DT_FIT:g}")
    print(f"    and held in ms, so grid error is not laid on top of the dt differences.")
    for name, J in REGIMES.items():
        print(f"\n    [{name}, J = {J:g}]  U_max held: "
              + "; ".join(f"{s} {L['u_max'][(s, name)]:.1f} ms" for s in SIDES))
        print(f"    {'dt':>9}{'Q(S1)':>7}{'Q(re)':>7}"
              f"{'lam_max S1':>12}{'lam_max re':>12}{'ratio':>8}"
              f"{'maxPhi S1':>11}{'maxPhi re':>11}{'ratio':>8}{'verdict':>9}")
        for i, dt in enumerate(L["dts"]):
            a = dict(zip(L["cols"], L[("S1", name)][i].tolist()))
            b = dict(zip(L["cols"], L[("re", name)][i].tolist()))
            v = ("LOWER" if b["lam_max"] < a["lam_max"] * (1 - 1e-6) else
                 "HIGHER" if b["lam_max"] > a["lam_max"] * (1 + 1e-6) else "EQUAL")
            print(f"    {dt:>9.5f}{int(a['Q']):>7}{int(b['Q']):>7}"
                  f"{a['lam_max']:>12.5f}{b['lam_max']:>12.5f}"
                  f"{b['lam_max'] / a['lam_max']:>8.4f}"
                  f"{a['phi_max']:>11.5f}{b['phi_max']:>11.5f}"
                  f"{b['phi_max'] / a['phi_max']:>8.4f}{v:>9}")
        print(f"\n    {'dt':>9}{'side':>5}{'lam*dt/2':>11}{'acf1':>9}{'r [1/ms]':>13}"
              f"{'sigma':>13}{'period [ms]':>13}{'ptp':>11}{'secs':>9}")
        for i, dt in enumerate(L["dts"]):
            for side in SIDES:
                d = dict(zip(L["cols"], L[(side, name)][i].tolist()))
                print(f"    {dt:>9.5f}{side:>5}{hzd.step_error(d['lam_max'], dt):>11.4f}"
                      f"{d['acf1']:>9.4f}{d['r']:>13.8f}{d['sigma']:>13.6e}"
                      f"{d['period']:>13.5f}{d['ptp']:>11.6f}{d['secs']:>9.1f}")
        for side in SIDES:
            rr = L[(side, name)][:, cl["r"]].tolist()
            print(f"    r drift over the ladder, {side:>2}: {rr[0]:.8f} -> {rr[-1]:.8f} "
                  f"({(rr[-1] / rr[0] - 1) * 100:+.3f}%)   [E held, so r is not pinned]")

    # 8. the orders on ptp -- the question of item 3
    print(f"\n[8] DOES THE AMPLITUDE ACQUIRE A dt -> 0 LIMIT? Observed orders on each")
    print(f"    quantity (helpers.orders) and whether helpers.richardson accepts the")
    print(f"    sequence. ptp is the one the brief asks about; `REFUSED` is a legitimate")
    print(f"    answer and is not worked around.")
    for name in REGIMES:
        print(f"\n    [{name}]")
        print(f"    {'quantity':>10}{'side':>5}  {'values at dt = ' + ', '.join('%g' % d for d in L['dts']):<60}")
        for key in QUANTITIES[name]:
            for side in SIDES:
                v = L[(side, name)][:, cl[key]]
                lim, p = CO["limits"][(side, name, key)]
                o = ['%.2f' % x for x in CO["orders"][(side, name, key)]]
                vals = ", ".join(f"{float(x):.6g}" for x in v)
                climbing = all(y > x for x, y in zip(CO["orders"][(side, name, key)][:-1],
                                                     CO["orders"][(side, name, key)][1:]))
                verdict = ("REFUSED -- no limit" if lim != lim
                           else ("NOT EXTRAPOLATED (NO_LIMIT): orders still "
                                 + ("CLIMBING" if climbing else "unsettled")
                                 + f"; richardson would say {lim:.6g} and that is not a limit"
                                 if key in NO_LIMIT
                                 else f"limit {lim:.8g} (order {p:.2f})"))
                print(f"    {key:>10}{side:>5}  {vals}")
                print(f"    {'':>10}{'':>5}  orders {str(o):<26} {verdict}")

    # 9. the H control
    print(f"\n[9] EPR. The RAW H pair is the control: against a counting measure H/dt grows")
    print(f"    as r ln(1/dt), so its increment per halving must come out at r ln 2. The")
    print(f"    POISSON-REFERENCED pair (epr.poisson_reference_entropy) is the finite one.")
    print(f"    sigma is identical either way -- both halves subtract the same H_ref.")
    for name in REGIMES:
        print(f"\n    [{name}]")
        print(f"    {'dt pair':>16}{'side':>5}{'d(H_fwd/dt)':>14}{'d(H_rev/dt)':>14}"
              f"{'r ln2':>12}{'ratio fwd':>11}{'ratio rev':>11}")
        for i in range(len(L["dts"]) - 1):
            for side in SIDES:
                t = L[(side, name)]
                dF = float(t[i + 1, cl["H_fwd"]] - t[i, cl["H_fwd"]])
                dR = float(t[i + 1, cl["H_rev"]] - t[i, cl["H_rev"]])
                rl2 = float(t[i + 1, cl["r"]]) * math.log(2.0)
                print(f"    ({L['dts'][i]:>6g},{L['dts'][i+1]:>7g}){side:>5}"
                      f"{dF:>14.6e}{dR:>14.6e}{rl2:>12.6e}"
                      f"{dF / rl2:>11.4f}{dR / rl2:>11.4f}")
        print(f"    {'dt':>9}{'side':>5}{'H_rel_fwd':>14}{'H_rel_rev':>14}{'sigma':>14}"
              f"{'H_rel_rev - H_rel_fwd':>24}")
        for i, dt in enumerate(L["dts"]):
            for side in SIDES:
                d = dict(zip(L["cols"], L[(side, name)][i].tolist()))
                print(f"    {dt:>9.5f}{side:>5}{d['H_rel_fwd']:>14.6e}"
                      f"{d['H_rel_rev']:>14.6e}{d['sigma']:>14.6e}"
                      f"{d['H_rel_rev'] - d['H_rel_fwd']:>24.6e}")

    # 10. the cost comparison -- the brief's deliverable
    print(f"\n[10] THE COST. Read exactly as S1-dt_cost.py's table: for each quantity and")
    print(f"     tolerance, the COARSEST dt whose relative error against the Richardson limit")
    print(f"     stays inside the tolerance all the way down (helpers.converged_at).")
    for name in REGIMES:
        for side in SIDES:
            print(f"\n     [{name}, {side}]  U_max = {L['u_max'][(side, name)]:.1f} ms")
            print(f"     {'quantity':<22}" + "".join(f"{'tol ' + f'{x:.0e}':>22}"
                                                     for x in THRESHS))
            t = L[(side, name)]
            for key in QUANTITIES[name]:
                lim, _ = CO["limits"][(side, name, key)]
                line = f"     {key:<22}"
                for th in THRESHS:
                    if key in NO_LIMIT:
                        line += f"{'no limit to read':>22}"; continue
                    if lim != lim:
                        line += f"{'REFUSED':>22}"; continue
                    dt_ok = CO["converged"][(side, name, key)][th]
                    if dt_ok != dt_ok:
                        line += f"{'not on this ladder':>22}"; continue
                    i = L["dts"].index(dt_ok)
                    gib = float(t[i, cl["model_bytes"]]) / 2 ** 30
                    line += f"{f'dt {dt_ok:g}  {gib:.2f}G':>22}"
                print(line)

    print(f"\n     THE DELIVERABLE: the dt that buys {SIGMA_TARGET:.0%} sigma, and what it costs.")
    print(f"     Memory is helpers.epr_peak_bytes(1, Q, inplace=False) at Q = U_max/dt -- the")
    print(f"     same analytic per-cell figure S1-dt_cost.py sizes a run with.")
    for name in REGIMES:
        if (("S1", name) not in CO["sigma_cost"]): continue
        print(f"\n     [{name}]")
        print(f"     {'side':>5}{'sigma limit':>14}{'dt':>12}{'Q':>9}{'peak mem':>12}"
              f"{'wall [s]':>11}{'from':>16}")
        for side in SIDES:
            e = CO["sigma_cost"][(side, name)]
            if e["limit"] != e["limit"]:
                print(f"     {side:>5}{'REFUSED':>14}"); continue
            src = ("extrapolated" if e["extrapolated"] else "on the ladder")
            print(f"     {side:>5}{e['limit']:>14.8g}{e['dt']:>12g}{e['Q']:>9}"
                  f"{e['bytes'] / 2**30:>11.1f}G{e['secs']:>11.0f}{src:>16}")
        a, b = CO["sigma_cost"][("S1", name)], CO["sigma_cost"][("re", name)]
        if a["limit"] == a["limit"] and b["limit"] == b["limit"]:
            print(f"\n     NET, re-anchored / S1: dt {b['dt'] / a['dt']:.3g}x, "
                  f"Q {b['Q'] / a['Q']:.3g}x, memory {b['bytes'] / a['bytes']:.3g}x, "
                  f"wall {b['secs'] / a['secs']:.3g}x")
            print(f"     -> the re-anchored point is "
                  f"{'CHEAPER' if b['bytes'] < a['bytes'] else 'DEARER'} in memory and "
                  f"{'CHEAPER' if b['secs'] < a['secs'] else 'DEARER'} in wall clock "
                  f"at {SIGMA_TARGET:.0%} sigma.")
            for side, e in (("S1", a), ("re", b)):
                tag = ("err ~ dt^%.2f over the finest three rungs, Richardson order %.2f"
                       % (e["p_err"], e["p_richardson"]))
                print(f"       [{side}] {tag}; "
                      + (f"raw extrapolated dt = {e['dt_raw']:.5g} ms, reported on the "
                         f"admissible 1/2^k ladder" if e["extrapolated"]
                         else "reached on the ladder itself"))
            print(f"     Wall clock is extrapolated from this ladder's own timings "
                  f"(secs ~ dt^{a['q_time']:.2f} on the S1 side, dt^{b['q_time']:.2f} on the "
                  f"re-anchored one), elapsed and not CPU.")


# ── Plots ─────────────────────────────────────────────────────────────────────

if run_report:
    hlp.pub_style()
    banner = pre.banner(PRESET, dict(PARAMS_NU0, dt=DT_REF))

    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.set_label('nu0test-branches')

    Js = R["J"].tolist()
    quiet  = [float(R["roots"][j][0]) for j in Js]
    active = [float(R["roots"][j][-1]) if len(R["roots"][j]) > 1 else float('nan') for j in Js]
    middle = [float(R["roots"][j][1]) if len(R["roots"][j]) > 2 else float('nan') for j in Js]
    attr   = [dict(zip(COLS, R["attractor"][j]))["r"] for j in Js]

    ax_b.plot(Js, quiet, 'o-', color='C0', markersize=3, label='quiescent branch')
    ax_b.plot(Js, middle, 'x--', color='0.6', markersize=4, label='middle branch (unstable)')
    ax_b.plot(Js, active, 's-', color='C1', markersize=3, label='active branch')
    ax_b.plot(Js, attr, '.', color='C3', markersize=7,
              label='attractor from a warm start on the active branch')
    for name, J in REGIMES.items():
        ax_b.axhline(dict(zip(COLS, G[(name, "S1")]))["r"], color='C2', linewidth=0.9,
                     linestyle=':', label=f'S1 {name} rate (J = {J:g})')
    ax_b.axvline(R["J_sn"], color='k', linewidth=0.8, linestyle='--',
                 label=f'fold at J = {R["J_sn"]:.2f}')
    ax_b.set_yscale('log')
    ax_b.set_ylabel('rate $r$ [1/ms]'); ax_b.set_title('Fixed-point branches and the attractor')
    ax_b.legend(fontsize=7)

    ax_r.plot(Js, [R["rho"][j]["low"] for j in Js], 'o-', color='C0', markersize=3,
              label='quiescent branch')
    ax_r.plot(Js, [R["rho"][j]["high"] for j in Js], 's-', color='C1', markersize=3,
              label='active branch')
    ax_r.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label=r'$\rho = 1$')
    ax_r.set_ylabel(r'spectral radius $\rho$')
    ax_r.set_title(f'Linear stability (at $\\Delta_t$ = {DT_EIG:g})')
    ax_r.legend(fontsize=8)

    for ax in (ax_b, ax_r):
        ax.set_xlabel('Recurrent coupling  $J$'); ax.grid(alpha=0.3)
    fig.suptitle(f'nu0 re-anchoring: $\\nu_0$ = {NU0_NEW:g} /ms, '
                 r'$\theta$ held — ' + banner, fontsize=8)
    fig.tight_layout()

    # The E restore: r(E) at both J, and the stability cut in J at the tuned E.
    fig2, (ax_e, ax_j) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig2.set_label('nu0test-erestore')
    Es = EO["E_scan"].tolist()
    for i, (name, J) in enumerate(REGIMES.items()):
        c = f'C{i}'
        ax_e.plot(Es, EO["scan"][name].tolist(), 'o-', color=c, markersize=3,
                  label=f'{name}, J = {J:g}')
        ax_e.axhline(EO["targets"][name], color=c, linewidth=0.9, linestyle=':',
                     label=f'S1 {name} rate')
        ax_e.plot([EO["tuned"][name]["E"]], [EO["targets"][name]], '*', color=c, markersize=11,
                  label=f'E = {EO["tuned"][name]["E"]:.4f}')
    ax_e.set_xlabel('External drive  $E$'); ax_e.set_ylabel('attractor rate $r$ [1/ms]')
    ax_e.set_title('Restoring the operating rate with $E$'); ax_e.legend(fontsize=7)

    ax_j.plot(O["J_prof"].tolist(), O["rho_prof"], 'o-', color='C1', markersize=3,
              label=f'$\\rho$ at E = {EO["tuned"]["cycle"]["E"]:.4f}')
    ax_j.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label=r'$\rho = 1$')
    if O["J_c"] == O["J_c"]:
        ax_j.axvline(O["J_c"], color='C3', linewidth=0.9, linestyle='-',
                     label=f'onset J = {O["J_c"]:.3f}')
    ax_j.axvline(J_C_S1, color='C2', linewidth=0.9, linestyle=':',
                 label=f'S1 Hopf J = {J_C_S1:g}')
    ax_j.axvline(J_CYCLE, color='0.5', linewidth=0.8, linestyle='-.',
                 label=f'J = {J_CYCLE:g}')
    ax_j.set_ylabel(r'spectral radius $\rho$')
    ax_j.set_title(f'Re-anchored onset (at $\\Delta_t$ = {DT_EIG:g})')
    ax_j.legend(fontsize=7)
    for ax in (ax_e, ax_j):
        ax.set_xlabel('Recurrent coupling  $J$' if ax is ax_j else ax.get_xlabel())
        ax.grid(alpha=0.3)
    fig2.suptitle(r'$\nu_0$ re-anchoring: the rate restored with $E$, $\theta$ held — '
                  + banner, fontsize=8)
    fig2.tight_layout()

        # The ladder: the hazard peak per dt, and the amplitude that may or may not converge.
    fig3, (ax_p, ax_s, ax_a) = plt.subplots(1, 3, figsize=(14, 4.4))
    fig3.set_label('nu0test-ladder')
    STY = {"S1": dict(color='C0', marker='o'), "re": dict(color='C3', marker='s')}
    LBL = {"S1": r'$S1$ ($\nu_0$ = 1, E = 0)', "re": r'$\nu_0$ = 0.01, E restored'}
    for name in REGIMES:
        ls = '-' if name == "cycle" else '--'
        for side in SIDES:
            t = L[(side, name)]
            kw = dict(STY[side], linestyle=ls, markersize=4,
                      label=f'{LBL[side]}, {name}')
            ax_p.plot(L["dts"], t[:, cl["phi_max"]].tolist(), **kw)
            ax_s.plot(L["dts"], [hzd.step_error(float(v), d)
                                 for v, d in zip(t[:, cl["lam_max"]], L["dts"])], **kw)
            if name == "cycle":
                ax_a.plot(L["dts"], t[:, cl["ptp"]].tolist(), **kw)
    ax_p.set_ylabel(r'max $\Phi$ over a cycle'); ax_p.set_title('Hazard peak per step')
    ax_s.set_ylabel(r'$\lambda\,\Delta_t/2$ at the peak rate')
    ax_s.axhline(1.0, color='k', linewidth=0.8, linestyle=':')
    ax_s.set_title('Step-error bound (hazards.step_error)')
    ax_a.set_ylabel('amplitude  ptp($r$) [1/ms]')
    ax_a.set_title('Cycle amplitude: no limit on either side')
    for ax in (ax_p, ax_s, ax_a):
        ax.set_xscale('log'); ax.set_xlabel(r'solver step  $\Delta_t$ [ms]')
        ax.grid(alpha=0.3); ax.legend(fontsize=6)
    ax_p.set_yscale('log'); ax_s.set_yscale('log')
    # The banner's own dt is DT_REF, which is where E was tuned and NOT where these
    # figures are drawn, so the swept range is stated in place of it.
    banner_l = banner.replace(f'dt={DT_REF:g}',
                              f'dt={DTS_L[0]:g}..{DTS_L[-1]:g} (E tuned at dt={DT_REF:g})')
    fig3.suptitle(r'$\nu_0$ re-anchoring, the $\Delta_t$ ladder — ' + banner_l, fontsize=8)
    fig3.tight_layout()

    # The cost: how far sigma is from its own limit at each step, and what that step costs.
    fig4, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig4.set_label('nu0test-cost')
    for side in SIDES:
        e = CO["sigma_cost"][(side, "cycle")]
        if e["limit"] != e["limit"]: continue
        ax_e.plot(L["dts"], e["err"], **dict(STY[side], markersize=4, label=LBL[side]))
        ax_e.plot([e["dt"]], [SIGMA_TARGET], '*', color=STY[side]["color"], markersize=12,
                  label=f'$\\Delta_t$ = {e["dt"]:g} ({e["bytes"] / 2**30:.1f} GiB)')
        Qs = [max(1, round(e["u_max"] / d)) for d in L["dts"] + [e["dt"]]]
        ax_m.plot(L["dts"] + [e["dt"]],
                  [hlp.epr_peak_bytes(1, q, inplace=False) / 2 ** 30 for q in Qs],
                  **dict(STY[side], markersize=4, label=f'{LBL[side]}  (U_max '
                                                        f'{e["u_max"]:.0f} ms)'))
    ax_e.axhline(SIGMA_TARGET, color='k', linewidth=0.8, linestyle=':',
                 label=f'{SIGMA_TARGET:.0%} of the limit')
    ax_e.set_ylabel(r'$|\sigma(\Delta_t) - \sigma^\ast| / |\sigma^\ast|$')
    ax_e.set_title(r'Distance from the Richardson limit, cycle')
    ax_m.set_ylabel('peak memory [GiB]  (epr_peak_bytes, Q = $U_{max}/\\Delta_t$)')
    ax_m.set_title('What that step costs')
    for ax in (ax_e, ax_m):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'solver step  $\Delta_t$ [ms]'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig4.suptitle(r'$\nu_0$ re-anchoring: the cost of 1% $\sigma$ on the cycle — '
                  + banner_l, fontsize=8)
    fig4.tight_layout()

    rp.show()
