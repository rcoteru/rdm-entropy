"""
Is the Hopf at S1 super- or subcritical? Up-and-down continuation in J.

Never checked, at any operating point, under either route. The question is which way the
branch of cycles leaves the bifurcation: forward, so that the amplitude grows continuously
from zero as J passes J_c (supercritical), or backward, so that a branch of large-amplitude
cycles already exists below J_c and the system jumps onto it (subcritical). The two are
distinguished by what the system does on the way back down.

THE AMPLITUDE ROUTE IS CLOSED. The textbook test -- fit ptp ~ (J - J_c)^(1/2) and see whether
the exponent holds -- cannot be run here, because the cycle amplitude has no dt -> 0 limit in
this model: the observed order is still 0.76 at dt = 1/64 and the value is still climbing
(S1-cycle_point.py; decided 2026-11-08). Every amplitude in this script is therefore quoted at
ONE STATED dt and never extrapolated, never fitted to a power and never passed to richardson.

CONTINUATION NEEDS NO CONVERGED AMPLITUDE, which is why it is the route taken. Sweep J up from
10 to 20, then back down to 10, each step starting from the previous step's FINAL STATE rather
than from a fresh initial condition. A supercritical Hopf has exactly one attractor at every J
and the two passes retrace each other. A subcritical one has a window where a stable cycle and
a stable fixed point coexist, and in that window the upward pass -- arriving on the fixed-point
branch -- stays settled while the downward pass -- arriving on the cycle branch -- keeps
cycling. That difference IS the measurement:

    hysteresis, i.e. a bistable window  =>  SUBCRITICAL
    neither pass differing from the other  =>  SUPERCRITICAL

Expectation, recorded here before the run so that agreement or surprise can be read off:
SUPERCRITICAL. sigma rises continuously from zero with exponent ~1.11 near onset
(S1-cycle_point.py), where a subcritical Hopf would jump discontinuously. If this script finds
hysteresis it CONTRADICTS that measurement and is a stop-and-flag, not a result to file.

What continuation requires, and what it is easy to get wrong:

  the state must actually carry over. A fresh model is rebuilt per J -- the refractory floor
      R_abs depends on the coupling, so J cannot simply be assigned in place -- and (p, S, m)
      are copied across. The age grid is fitted ONCE, at the most demanding J, and held fixed
      for the whole sweep, which is what makes the copy well defined.
  the per-step settling must be short enough to be affordable and long enough to leave the
      transient behind. Too long and the two passes converge on the same attractor even where
      two exist, erasing the very hysteresis being looked for; too short and the state
      reported is the previous J's, which MANUFACTURES hysteresis. Both failure modes point
      the same way, so SETTLE_MS is swept as a control at the three J nearest onset.
  a verdict per J needs a criterion. "Cycling" is ptp above a floor set by the settled
      fixed point's own numerical noise, confirmed by cycle_window finding whole periods.
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.hazards as hzd
import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_single import RDMIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. The solver kernel, since this is a dt-fixed study of the
                           # branch structure and not a comparison of discretizations.

# dt is FIXED and stated, not swept. The object of study is the shape of the branch, which is
# a property of the continuous-time model; 1/4 resolves the cycle at J = 17 (max Phi 0.75,
# S1-dt_conv.py) and is the step the brief sets for this run. Every amplitude below is an
# amplitude AT dt = 1/4 and is labelled so.
DT = 0.25

PARAMS: dict = pre.params(PRESET, hazard=HAZARD, dt=DT)
MODEL: dict = {k: v for k, v in PARAMS.items() if k not in ("dt", "hazard", "J")}
hzd.check_step(HAZARD, PARAMS["nu0"], DT)

# The measured Hopf under this hazard, as a dt -> 0 limit (S1-cycle_point.py / S1-hopf_dt.py).
# Used only to place the ladder and to report distances; nothing is fitted to it.
J_C = 11.5318

# The ladder. 10 to 20 brackets the Hopf with 1.53 of margin below and 8.47 above, and is
# traversed in both directions. The spacing is uniform rather than concentrated at onset: a
# bistable window is an INTERVAL, and a uniform ladder is what measures its width without
# assuming where it is. 0.125 in J gives 81 points, i.e. 162 settle-and-record units.
J_LO, J_HI, J_STEP = 10.0, 20.0, 0.125
J_UP = torch.arange(J_LO, J_HI + J_STEP / 2, J_STEP)
J_DOWN = J_UP.flip(0)

# Per-step durations. SETTLE_MS is the transient allowed after each J change and is the one
# number this method turns on (see the docstring); REC_MS is the window the verdict is read
# from, long enough for cycle_window to find several whole periods at the ~70 ms period here.
SETTLE_MS = 20000.0
REC_MS    = 4000.0

# The control on SETTLE_MS, run at the three J nearest onset on both passes. If the verdict at
# a J moves when the settling is lengthened, the verdict was the previous J's state and not
# this one's -- which is the failure mode that manufactures hysteresis out of nothing.
SETTLE_CONTROL = (20000.0, 60000.0)
N_CONTROL      = 3

# The age grid is fitted ONCE and held across the sweep, so that the state copied from one J
# to the next lives on the same grid. Fitted at the J that demands the most of it: the LOW end,
# where the rate is smallest and mass survives furthest into the tail.
TOL     = 1e-5
J_FIT   = J_LO

# A J counts as cycling when the rate is modulated by more than this fraction of its mean,
# i.e. ptp/r > CYCLE_MOD. A RELATIVE criterion, and deliberately not an absolute floor scaled
# off the most settled cell: below onset the state relaxes to its fixed point to machine
# precision (ptp ~ 1e-16 /ms here), so a floor read from there is machine epsilon and a
# threshold built on it calls every cell in the sweep a cycle -- which is a verdict that
# cannot fail and therefore says nothing. The modulation depth has a scale of its own: it is
# ~1e-14 on a settled cell and ~10 on the cycle at J = 17, so anything in between separates
# them, and 1e-2 is twelve orders above the floor and three below the signal.
CYCLE_MOD = 1e-2

DEVICE = 'cpu'     # one population on a few hundred age bins is launch-latency bound

COLS = ("r", "ptp", "period", "n_cyc", "acf1", "phi_max", "trunc")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE     = hlp.cache_path(STEM, dict(PARAMS, preset=PRESET, j=[J_LO, J_HI, J_STEP],
                                      settle=SETTLE_MS, rec=REC_MS, tol=TOL,
                                      ctrl=list(SETTLE_CONTROL), n_ctrl=N_CONTROL),
                           root=CACHE_DIR)

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


# ── Measurement ───────────────────────────────────────────────────────────────

def build(J: float, Qm: list[int] | None = None):
    return RDMIsingModel(float(J), **MODEL, dt=DT, hazard=HAZARD, Qm=Qm, device=DEVICE)


def measure(J: float, Qm: list[int], state, settle_ms: float) -> tuple[list[float], tuple]:
    """ Settle at J from `state`, record, and return the row plus the final state.

    `state` is (p, S, m) from the previous J, or None to start from the model's own initial
    condition. The model is REBUILT rather than re-coupled: the absolute-refractory floor
    R_abs is computed from w at construction, so assigning J in place would leave the floor at
    the old coupling -- a silent inconsistency that would look like a branch. The grid is the
    same at every J, which is what makes copying the state across well defined. """
    net = build(J, Qm)
    if state is not None:
        p, S, m = state
        net.p = [t.clone() for t in p]
        net.S = [t.clone() for t in S]
        net.m = m.clone()

    net.forward(hlp.n_steps(settle_ms, DT))
    traj = net.trajectory(hlp.n_steps(REC_MS, DT), r_tot=True, pb=False)

    r = traj["r_tot"]
    a, b, n_cyc = hlp.cycle_window(r)
    period = (b - a) * DT / n_cyc if n_cyc else float('nan')

    # Over the cycle-aligned window, so the amplitude is not read across a partial period.
    x = r[a:b] - r[a:b].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))

    phi_max = 0.0
    n_probe = max(1, int(period / DT) if n_cyc else len(r) // 10)
    for _ in range(n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        net.update()

    row = [float(r[a:b].mean()),
           float(r[a:b].max() - r[a:b].min()),
           period,
           float(n_cyc),
           acf1,
           float(phi_max),
           float(net.truncation_error())]
    final = ([t.clone() for t in net.p], [t.clone() for t in net.S], net.m.clone())
    return row, final


def sweep(Js: torch.Tensor, Qm: list[int], settle_ms: float, label: str) -> torch.Tensor:
    """ One pass of the continuation: each J started from the previous J's final state.

    The FIRST J of a pass is started from the model's own initial condition, which is the only
    honest choice -- a pass has to begin somewhere, and beginning it from the other pass's end
    would make the two passes one long sweep instead of two. The up pass therefore enters at
    J = 10 cold (below onset, where there is one attractor and the entry does not matter) and
    the down pass enters at J = 20 cold (far above onset, same). """
    rows, state = [], None
    t0 = time.time()
    for k, J in enumerate(Js):
        row, state = measure(float(J), Qm, state, settle_ms)
        rows.append(row)
        d = dict(zip(COLS, row))
        print(f"  [{label}] J = {float(J):6.3f}  r = {d['r']:.6f}  ptp = {d['ptp']:.6f}  "
              f"period = {d['period']:7.2f}  n_cyc = {int(d['n_cyc']):3d}  "
              f"acf1 = {d['acf1']:+.3f}")
        assert d["trunc"] < TOL, (
            f"truncation_error = {d['trunc']:.2e} at J = {float(J):g}: the age window fitted "
            f"at J = {J_FIT} does not hold here.")
    print(f"  [{label}] {len(Js)} points in {time.time() - t0:.0f}s")
    return torch.tensor(rows)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    CACHE_DIR.mkdir(exist_ok=True)
    res: dict = ({"params": PARAMS, "j_up": J_UP, "dt": DT}
                 if overwrite or not CACHE.exists()
                 else torch.load(CACHE, weights_only=False))
    t_all = time.time()

    if "Qm" not in res:
        # Sized on the attractor at the FULL per-J budget, as fit_age_grid requires, and at
        # the low end of the ladder where the rate is smallest and the tail longest.
        print(f"Fitting the age window at J = {J_FIT}, dt = {DT} ...")
        Qm = hlp.fit_age_grid(lambda q: build(J_FIT, q), tol=TOL / 100,
                              equi=hlp.n_steps(SETTLE_MS + REC_MS, DT))
        res["Qm"] = Qm
        print(f"  Qm = {Qm}, U_max = {Qm[0] * DT:.1f} ms")
    Qm = res["Qm"]

    for key, Js, label in (("up", J_UP, "up  "), ("down", J_DOWN, "down")):
        if key in res:
            print(f"\n[{key}] cached")
            continue
        print(f"\n[{key}] {len(Js)} points, J = {float(Js[0]):g} -> {float(Js[-1]):g}, "
              f"settle = {SETTLE_MS:g} ms each")
        res[key] = sweep(Js, Qm, SETTLE_MS, label)
        torch.save(res, CACHE)

    # The settling control, at the N_CONTROL J nearest onset on each pass. Re-run from the
    # same entry as the pass itself would reach them from, which means re-walking the pass up
    # to that point -- there is no shortcut, the whole claim is that the state is inherited.
    if "control" not in res:
        near = torch.argsort((J_UP - J_C).abs())[:N_CONTROL].sort().values
        print(f"\n[control] settling sweep at J = "
              f"{[round(float(J_UP[i]), 3) for i in near]}")
        ctrl = {}
        for settle in SETTLE_CONTROL:
            for key, Js in (("up", J_UP), ("down", J_DOWN)):
                idx = near if key == "up" else (len(J_UP) - 1 - near).flip(0)
                state, out = None, {}
                for k, J in enumerate(Js):
                    if k > int(idx.max()):
                        break
                    # walked at the production settling until the control window is reached
                    s = settle if k in idx.tolist() else SETTLE_MS
                    row, state = measure(float(J), Qm, state, s)
                    if k in idx.tolist():
                        out[float(J)] = row
                ctrl[(key, settle)] = out
                print(f"  [{key}, settle = {settle:g} ms] "
                      + ",  ".join(f"J={j:g}: ptp={r[1]:.6f}" for j, r in out.items()))
        res["control"] = ctrl
        torch.save(res, CACHE)

    torch.save(res, CACHE)
    print(f"\nDone in {time.time() - t_all:.0f}s. {CACHE}.")


# ── Analysis ──────────────────────────────────────────────────────────────────

if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    res = torch.load(CACHE, weights_only=False)
    col = {k: i for i, k in enumerate(COLS)}
    up, down = res["up"], res["down"].flip(0)      # down re-ordered onto the same J axis
    Js = J_UP

    # Modulation depth, which is what the cycling verdict is read from.
    mod_up   = up[:, col["ptp"]] / up[:, col["r"]].clamp_min(1e-300)
    mod_down = down[:, col["ptp"]] / down[:, col["r"]].clamp_min(1e-300)
    print(f"\n[floor] modulation depth ptp/r over the ladder: "
          f"{float(mod_up.min()):.2e} at the settled end to {float(mod_up.max()):.2e} at "
          f"J = {J_HI:g}. A J counts as cycling above {CYCLE_MOD:g}.")
    print(f"    the settled end relaxes to MACHINE ZERO (smallest ptp "
          f"{float(torch.cat([up[:, col['ptp']], down[:, col['ptp']]]).min()):.1e} /ms), "
          f"which is why the criterion is relative and not a floor scaled off it.")

    cyc_up   = mod_up > CYCLE_MOD
    cyc_down = mod_down > CYCLE_MOD
    disagree = cyc_up != cyc_down

    print(f"\n[onset] first cycling J on each pass:")
    for name, mask in (("up", cyc_up), ("down", cyc_down)):
        k = int(mask.nonzero()[0]) if bool(mask.any()) else None
        print(f"    {name:>4}: J = {float(Js[k]):.3f}" if k is not None else
              f"    {name:>4}: never cycles")

    # The threshold-free half of the same statement, and the more trustworthy one: a
    # bistable window is where the two passes land on DIFFERENT states, which is visible as a
    # relative difference in ptp without any cycling threshold having to be chosen. On one
    # attractor the two passes agree to the settling error; on two they differ by order one.
    rel_gap = ((up[:, col["ptp"]] - down[:, col["ptp"]]).abs()
               / torch.maximum(up[:, col["ptp"]], down[:, col["ptp"]]).clamp_min(1e-300))
    k_worst = int(rel_gap.argmax())
    print(f"\n[gap] |ptp_up - ptp_down| / max, over the ladder: "
          f"max {float(rel_gap.max()):.3f} at J = {float(Js[k_worst]):.3f}, "
          f"median {float(rel_gap.median()):.3e}")
    big = rel_gap > 0.5
    if bool(big.any()):
        print(f"    J with a gap above 50%: "
              f"{[round(float(j), 3) for j in Js[big]]}")
        print(f"    ⚠ check these against the settling control before reading them as "
              f"bistability: within ~0.1 of J_c = {J_C} the relaxation time diverges, so "
              f"{SETTLE_MS:g} ms leaves a point unsettled and the two passes arrive at "
              f"different points of the SAME approach.")

    # Which J the settling control says are not settled, so the verdict can tell a bistable
    # window from an unconverged one. These are two different things that look identical in
    # the ptp column: near onset the relaxation time diverges, so a point that has not
    # finished approaching its single attractor is reached from above and from below at
    # different distances along the SAME approach, and the passes differ for a reason that
    # has nothing to do with branch structure. A drift above DRIFT_TOL between the two
    # control settlings means the point is still moving and carries no verdict.
    DRIFT_TOL = 0.05
    unsettled = torch.zeros(len(Js), dtype=torch.bool)
    if "control" in res:
        ctrl = res["control"]
        for k, J in enumerate(Js):
            j = float(J)
            if j not in ctrl[("up", SETTLE_CONTROL[0])]:
                continue
            for key in ("up", "down"):
                v = [ctrl[(key, s)][j][col["ptp"]] for s in SETTLE_CONTROL]
                if abs(v[-1] - v[0]) / max(abs(v[0]), 1e-300) > DRIFT_TOL:
                    unsettled[k] = True

    # The verdict. A bistable window is a set of J at which the two passes disagree about
    # whether the system cycles; its width is what distinguishes the two cases -- but only
    # where the points carry a verdict at all.
    print(f"\n[verdict]")
    checked = disagree & ~unsettled
    if bool(checked.any()):
        w = Js[checked]
        print(f"    HYSTERESIS: the passes disagree at {int(checked.sum())} of {len(Js)} J, "
              f"from J = {float(w.min()):.3f} to {float(w.max()):.3f} "
              f"(width {float(w.max() - w.min()) + J_STEP:.3f}), at points the settling "
              f"control calls settled.")
        print(f"    => SUBCRITICAL. This CONTRADICTS the sigma exponent of ~1.11 measured "
              f"near onset, which requires a continuous onset. Stop and flag; do not file "
              f"this as a result without reconciling the two.")
    elif bool(disagree.any()):
        w = Js[disagree & unsettled]
        print(f"    The passes disagree at {int(disagree.sum())} J, but every one of them is "
              f"a point the settling control calls UNSETTLED "
              f"({[round(float(j), 3) for j in w]}): the ptp there still moves by more than "
              f"{DRIFT_TOL:.0%} between {SETTLE_CONTROL[0]:g} and {SETTLE_CONTROL[-1]:g} ms "
              f"of settling.")
        print(f"    => INCONCLUSIVE AT THOSE J, supercritical everywhere else. A diverging "
              f"relaxation time at onset is expected and is not bistability; separating the "
              f"two there needs settling scaled as 1/(J - J_c), not a longer fixed window.")
    else:
        print(f"    No J at which the two passes disagree about cycling.")
        print(f"    => SUPERCRITICAL, as expected from the continuous sigma onset.")
    if bool(unsettled.any()):
        print(f"    (unsettled by the control: "
              f"{[round(float(j), 3) for j in Js[unsettled]]})")

    # Agreement between the passes where both cycle, which is the quantitative half of the
    # same statement: one attractor means the same amplitude from either direction.
    # Where both passes cycle AND the control does not call the point unsettled: on one
    # attractor the two passes must agree there to the settling error, and that agreement is
    # the quantitative half of "no hysteresis". Points the control cannot vouch for are
    # excluded rather than averaged in -- they are the ones whose disagreement means
    # something other than branch structure.
    both = cyc_up & cyc_down & ~unsettled
    if bool(both.any()):
        print(f"    agreement where both passes cycle and the control calls them settled "
              f"({int(both.sum())} J, from {float(Js[both][0]):.3f} to "
              f"{float(Js[both][-1]):.3f}):")
        for key in ("r", "ptp", "period"):
            a, b = up[both, col[key]], down[both, col[key]]
            ok = ~(a.isnan() | b.isnan())
            rel = ((a[ok] - b[ok]).abs() / a[ok].abs().clamp_min(1e-300)).max()
            print(f"        worst relative up-vs-down difference in {key:>6}: "
                  f"{float(rel):.2e}   ({int(ok.sum())} of {int(both.sum())} with a period)")

    # The step artifact, which at a coarse dt is the thing to rule out before reading any of
    # the above: acf1 = -1 is the whole population firing in alternate bins.
    print(f"\n[screen] lag-1 autocorrelation of r, minimum over the ladder: "
          f"up {float(up[:, col['acf1']].min()):+.4f}, "
          f"down {float(down[:, col['acf1']].min()):+.4f}  "
          f"(-1 would be an alternate-bin flip-flop, not a cycle)")
    # Peak Phi says whether dt resolves the hazard, which is a different question from
    # whether the averages have settled. It grows steeply with J here -- the drive that buys
    # margin past onset also buys stiffness -- so the top of the ladder is where a fixed dt
    # runs out, and the J at which it does has to be named rather than hidden in a maximum.
    PHI_WARN = 0.95
    phi = up[:, col["phi_max"]]
    hot = phi > PHI_WARN
    print(f"[screen] peak Phi over the ladder: {float(phi.min()):.3f} at J = {J_LO:g} to "
          f"{float(phi.max()):.3f} at J = {J_HI:g}")
    if bool(hot.any()):
        print(f"    ⚠ Phi > {PHI_WARN} from J = {float(Js[hot][0]):.3f} upward "
              f"({int(hot.sum())} of {len(Js)} points): at dt = {DT:g} the step barely "
              f"resolves the hazard there, so the cycle SHAPE at the top of the ladder is "
              f"not converged. The onset region carries the verdict and sits at Phi ~ "
              f"{float(phi[int((Js - J_C).abs().argmin())]):.3f}, which is fully resolved.")

    # The settling control.
    if "control" in res:
        print(f"\n[control] ptp at the three J nearest onset, as the settling is lengthened. "
              f"A verdict that moves here was the previous J's, not this one's:")
        ctrl = res["control"]
        for key in ("up", "down"):
            for j in sorted(ctrl[(key, SETTLE_CONTROL[0])]):
                vals = [ctrl[(key, s)][j][col["ptp"]] for s in SETTLE_CONTROL]
                drift = abs(vals[-1] - vals[0]) / max(abs(vals[0]), 1e-300)
                print(f"    {key:>4} J = {j:6.3f}: "
                      + ",  ".join(f"{s:g} ms -> {v:.6e}" for s, v in zip(SETTLE_CONTROL, vals))
                      + f"   (drift {drift:.1%})")


# ── Plots ─────────────────────────────────────────────────────────────────────

if run_plot:

    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    fig.set_label('continuation')

    for ax, key, ylab in ((axs[0], "ptp", r'peak-to-peak of $r$ [1/ms]'),
                          (axs[1], "r",   r'mean $r$ [1/ms]'),
                          (axs[2], "period", r'period [ms]')):
        ax.plot(Js, up[:, col[key]], 'o-', color="#2e6fb7", markersize=4,
                label=r'up: $J$ increasing')
        ax.plot(Js, down[:, col[key]], 's--', color="#c2562a", markersize=4,
                markerfacecolor='none', label=r'down: $J$ decreasing')
        ax.axvline(J_C, color='0.4', linewidth=0.8)
        ax.text(J_C, 0.96, r'  $J_c$', transform=ax.get_xaxis_transform(), va='top',
                fontsize=9, color='0.4')
        ax.set_xlabel(r'$J$'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # The amplitude panel carries the warning on its face: this is a value at one dt, and a
    # reader who fits a square root to it is fitting to something with no limit.
    axs[0].set_title(r'Amplitude at $\Delta_t = %g$ ms (no $\Delta_t \rightarrow 0$ limit)' % DT)
    axs[1].set_title('Mean rate')
    axs[2].set_title('Cycle period')

    fig.suptitle('Up-and-down continuation in $J$: the two passes retrace, or they do not\n'
                 + pre.banner(PRESET, PARAMS))
    fig.tight_layout()

    rp.show()
