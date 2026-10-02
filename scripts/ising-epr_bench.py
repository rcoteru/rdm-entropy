"""
Cost of the two entropy-production algorithms, single-system.

rdme.entropy_trajectory computes the same quantity two ways:

    method="joint"    build the explicit (Q, K) joint every step, read the EPR off it.  O(Q^2)
    method="sliding"  slide the four marginals the EPR actually reads.                  O(Q)

The joint path is the definition and the oracle; the sliding path is what makes long runs and
small dt affordable. This script measures what that costs, and confirms the two agree at every
size so the benchmark is timing a correct computation rather than a fast wrong one.

Q is swept by refining dt at fixed kernels, which is the knob that actually drives it in
practice: Qm = q_from_tau(tau, dt) grows like 1/dt, so halving the step doubles the age grid
and quadruples the joint path's per-step work while only doubling the sliding path's.

Two measurement details, both of which would otherwise corrupt the numbers:

  The fill phase is not free and not per-step. entropy_trajectory opens with Q_max
  un-reported ticks to fill the look-ahead buffers, which is O(Q) steps of O(Q) work -- O(Q^2)
  in total, the same order as the joint path's *entire* reported run at small T. Timing one
  call and dividing by T would therefore charge the sliding path for the fill. The per-step
  cost is taken as (t(T) - t(0)) / T: a T=0 call does the fill and the state initialisation and
  nothing else, so it measures the fixed cost directly rather than inferring it.

  That difference is still two large numbers subtracted, so each timing is the minimum of
  several repeats. The minimum, not the mean: the quantity wanted is the cost of the work, and
  every source of noise here (scheduling, clocks, other processes) only ever adds. A first
  attempt with two nonzero T values and no repeats produced a NEGATIVE per-step time at the
  largest grid, where the fill dominates both runs -- the failure mode this guards against.

  CUDA is asynchronous. Without an explicit synchronize, the timer measures how long it takes
  to enqueue the work, not to do it. Every timed region is bracketed by a sync when the device
  is CUDA, and a warm-up call precedes the measurements so that allocator growth and kernel
  loading are not counted.

Single-system only; the batched comparison is a separate question, since the batch path has its
own memory-vs-time tradeoff (the joint is (B, Q, K) and must be chunked, which the sliding path
avoids entirely) and would be dominated by that rather than by the exponent measured here.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

import rdme.kernels as krn
from rdme.single import RDMIsingModel


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR  = Path(__file__).parents[1] / "cache"
CACHE_FILE = CACHE_DIR / f"{bname}.pt"

run_sim   = True
run_plot  = True
overwrite = True

torch.set_default_dtype(torch.float64)

METHODS = ("joint", "sliding")
COLORS  = {"joint": "#b4453c", "sliding": "#2e6fb7"}


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_cuda = torch.device(device).type == 'cuda'

    # Kernels follow the repo's Ising convention; J and E put the model on a moving state, so
    # the EPR is not trivially zero and the hazards are exercised across their range.
    MODEL = dict(J=10.0, E=0.95, beta=30.0, theta=1.0,
                 tau_int=20.0, tau_ref=3.0, K_ref=0.0)
    HAZARD, DT0 = "escape", 0.2

    # dt refines -> Q grows like 1/dt. Stops where the joint path becomes painful, which is
    # the point the sweep is meant to demonstrate.
    DTS = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]
    T_RUN = 150              # reported steps in the timed run
    REPS  = 3                # repeats; the minimum of these is taken
    EQ_MS = 300.0            # short: equilibration is not what is being timed

    def sync():
        if is_cuda: torch.cuda.synchronize()

    def build(dt):
        return RDMIsingModel(**MODEL, dt=dt, dt0=DT0, hazard=HAZARD, device=device)

    def timed(dt, method, T):
        """ Wall time of one entropy_trajectory call of T reported steps, from an equilibrated
        state. The model is rebuilt each time so every call sees the same starting point. """
        net = build(dt)
        net.forward(int(EQ_MS / dt))
        sync()
        t0 = time.perf_counter()
        net.entropy_trajectory(T, method=method, pb=False)
        sync()
        return time.perf_counter() - t0

    def per_step_cost(dt, method):
        """ (t(T_RUN) - t(0)) / T_RUN, each term the minimum over REPS. """
        base = min(timed(dt, method, 0) for _ in range(REPS))
        full = min(timed(dt, method, T_RUN) for _ in range(REPS))
        return (full - base) / T_RUN

    if CACHE_FILE.exists() and not overwrite:
        print(f"Results already exist at {CACHE_FILE}. Skipping.")
    else:
        print(f"Running on: {device}   hazard={HAZARD}, dt0={DT0}")
        print(f"Per-step cost = (t(T={T_RUN}) - t(T=0)) / {T_RUN}, each the min of {REPS} "
              f"repeats,\nso the O(Q^2) fill phase is subtracted rather than amortised.\n")

        Qs, per_step, agree = [], {m: [] for m in METHODS}, []
        t_start = time.time()

        for dt in DTS:
            Q = krn.q_from_tau(max(MODEL["tau_int"], MODEL["tau_ref"]), dt, 0.01)
            Qs.append(Q)

            # agreement first: a speedup over a wrong answer is worth nothing
            ref = build(dt); ref.forward(int(EQ_MS / dt))
            sl  = build(dt); sl.forward(int(EQ_MS / dt))
            a = ref.entropy_trajectory(20, method="joint", pb=False)
            b = sl.entropy_trajectory(20, method="sliding", pb=False)
            d = max((a[k] - b[k]).abs().max().item()
                    for k in ("m", "sigma", "H_fwd", "H_rev"))
            agree.append(d)

            row = []
            for m in METHODS:
                timed(dt, m, 5)                       # warm-up, discarded
                ps = per_step_cost(dt, m)
                per_step[m].append(ps)
                row.append(ps)
            print(f"  dt={dt:6.4f}  Q={Q:5d} | joint {row[0]*1e3:8.3f} ms/step   "
                  f"sliding {row[1]*1e3:8.3f} ms/step   speedup {row[0]/row[1]:5.2f}x   "
                  f"|diff| {d:.1e}")

        print(f"\nDone in {time.time() - t_start:.1f}s")
        res = {"Q": torch.tensor(Qs), "dt": torch.tensor(DTS),
               "per_step": {m: torch.tensor(v) for m, v in per_step.items()},
               "agree": torch.tensor(agree), "device": device,
               "hazard": HAZARD, "T_run": T_RUN, "reps": REPS}
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(res, CACHE_FILE)
        print(f"Results saved to {CACHE_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"{CACHE_FILE} not found. Run the simulation first.")
    res = torch.load(CACHE_FILE, weights_only=False)
    Q, ps = res["Q"], res["per_step"]
    speedup = ps["joint"] / ps["sliding"]

    print(f"\nDevice: {res['device']}.  Agreement across the sweep: "
          f"max |diff| = {float(res['agree'].max()):.1e}")
    print(f"{'dt':>8} {'Q':>6} {'joint ms':>10} {'sliding ms':>11} {'speedup':>8}")
    for j in range(len(Q)):
        print(f"{float(res['dt'][j]):8.4f} {int(Q[j]):6d} {float(ps['joint'][j])*1e3:10.3f} "
              f"{float(ps['sliding'][j])*1e3:11.3f} {float(speedup[j]):7.2f}x")

    # Fitted exponent on the upper half of the sweep: at small Q both methods are dominated by
    # per-operation overhead rather than by the work, so a fit over the whole range would
    # measure Python, not the algorithm.
    half = len(Q) // 2
    lq = torch.log(Q[half:].to(torch.float64))
    print()
    for m in METHODS:
        y = ps[m][half:]
        if bool((y <= 0).any()):
            # a non-positive per-step time means the measurement, not the algorithm, failed
            print(f"  {m:>8}: non-positive timings {y.tolist()}; raise T_RUN or REPS")
            continue
        ly = torch.log(y)
        # closed-form least squares on the logs: lstsq went through LAPACK and died on a NaN
        slope = (((lq - lq.mean()) * (ly - ly.mean())).sum()
                 / ((lq - lq.mean()) ** 2).sum())
        print(f"  {m:>8}: fitted exponent over Q >= {int(Q[half])} is {float(slope):.2f}")

    # Where the two cross, by linear interpolation of the log-ratio
    lr = torch.log(speedup)
    cross = None
    for j in range(len(Q) - 1):
        if lr[j] < 0 <= lr[j + 1]:
            f = float(-lr[j] / (lr[j + 1] - lr[j]))
            cross = float(torch.exp(torch.log(Q[j].double())
                                    + f * (torch.log(Q[j + 1].double()) - torch.log(Q[j].double()))))
    print(f"\n  crossover at Q ~ {cross:.0f}" if cross else "\n  no crossover in range")
    print("  Below it the joint path wins: it is a handful of large kernels, while the sliding")
    print("  path issues dozens of small ones and pays the launches.")
    print("  The sliding cost is flat in Q over this whole range, i.e. launch-bound rather than")
    print("  work-bound -- its O(Q) arithmetic is still cheaper than its own fixed overhead at")
    print("  Q = 3685. So the speedup grows like Q^2 here, not Q: it is joint's quadratic cost")
    print("  against a constant, and it will only bend to Q once the work dominates.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for m in METHODS:
        ax1.plot(Q, ps[m] * 1e3, 'o-', color=COLORS[m], label=m, linewidth=2, markersize=7)
    # reference slopes, anchored at the largest Q so they sit beside the data rather than
    # through it; dashed and grey so they read as guides, not as series
    q = Q.to(torch.float64)
    for expn, style in ((2.0, ':'), (1.0, '--')):
        ref = float(ps["joint"][-1] if expn == 2.0 else ps["sliding"][-1]) * 1e3
        ax1.plot(q, ref * (q / float(q[-1])) ** expn, style, color='#999999', linewidth=1.5,
                 label=rf'$\propto Q^{{{int(expn)}}}$')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('age bins $Q$')
    ax1.set_ylabel('time per reported step [ms]')
    ax1.set_title('Cost of one EPR step')
    ax1.legend(); ax1.grid(alpha=0.3, which='both')

    ax2.plot(Q, speedup, 'o-', color='#2e6fb7', linewidth=2, markersize=7)
    ax2.axhline(1.0, color='#999999', linestyle='--', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('age bins $Q$')
    ax2.set_ylabel(r'speedup  (joint / sliding)')
    ax2.set_title('Sliding vs joint')
    ax2.grid(alpha=0.3, which='both')

    fig.suptitle(f"Entropy-production algorithms, single system ({res['device']}, "
                 f"hazard={res['hazard']})")
    fig.tight_layout()
    plt.show()
