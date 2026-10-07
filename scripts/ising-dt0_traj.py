""" Mean-field trajectories at several time quanta, one figure per hazard.

The companion to ising-mf_vs_sm-dt_dt0.py, which reduced each run to a stationary average and
so could say nothing about the shape of the orbit. Here the time series themselves are
overlaid: for each dt0 one colour, against a time axis in MILLISECONDS, so runs at different
quanta are on a common footing and the comparison is of the dynamics rather than of the step.

DT_FACTOR sets dt = dt0 * DT_FACTOR.

  At DT_FACTOR = 1 (the default, dt = dt0) each model sits at its own quantum, lam0*dt = 1,
  and all three hazards are the SAME kernel -- Phi = sigma(beta*h) for each of them. The three
  figures are then identical by construction, and what they show is the drift of the discrete
  model as its quantum moves, with no numerical refinement anywhere in it.

  At DT_FACTOR < 1 the hazards separate and the three figures become the thing to compare:
  each is the orbit of that hazard's continuous-time model, resolved below its own quantum.

Phases are not aligned across dt0. Every run starts from the same state and equilibrates for
the same physical time, but the quanta give different burst periods, so by the start of the
reported window the orbits have drifted apart in phase. Compare envelopes, amplitudes and
periods, not peak positions.

Runs are cached individually, keyed by the full physics + window signature.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from rdme.single import RDMIsingModel


# Cache
# ~~~~~

CACHE_DIR  = Path(__file__).parents[1] / "cache"
CACHE_FILE = CACHE_DIR / f"{Path(__file__).stem}.pt"

overwrite = True   # recompute every run and replace the cache file
run_plot  = True


# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

HAZARDS = ["synchronous", "async", "escape"]

# integration-based oscillatory regime -- the point of plotting trajectories at all. At a
# stable fixed point every panel here is a flat line.
PARAMS = dict(J=4.0,          # coupling strength, in physical units (per ms)
              E=1.0,          # external field
              beta=30,
              theta=1,
              tau_int=12,
              tau_ref=3,
              K_ref=0)

DT0_LIST = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
DT_FACTOR = 1.0   # dt = dt0 * DT_FACTOR; 1 is the diagonal, where the hazards coincide

# Windows in MILLISECONDS: every run must cover the same physical time, or the overlay is
# partly a comparison of how long each was watched.
WINDOWS = dict(equi_ms=800.0,   # discarded equilibration
               meas_ms=200.0)   # the reported window

EPS = 0.01    # mean-field age truncation

# Rates are stored per MILLISECOND, which is the unit the kernels are given in; HZ converts
# them to per second for display only, so the cache is unaffected and nothing is recomputed.
# The time axis stays in ms -- a burst is tens of ms wide, and seconds would compress it.
HZ = 1000.0


# Measurement
# ~~~~~~~~~~~

def cache_key(dt: float, dt0: float, hazard: str) -> str:
    """ Everything a run depends on, so a changed parameter misses rather than silently
    returning the old series. """
    sig = {**PARAMS, **WINDOWS, "eps": EPS, "dt": dt, "dt0": dt0, "hazard": hazard}
    return "|".join(f"{k}={sig[k]!r}" for k in sorted(sig))


def run_traj(dt: float, dt0: float, hazard: str) -> dict | None:
    """ One (dt, dt0, hazard) mean-field run, kept as time series.

    Everything is returned as a RATE (per ms) against a time axis in ms, which is the only
    form in which runs at different dt can be drawn on one pair of axes.
    """
    key = cache_key(dt, dt0, hazard)
    if not overwrite and key in cache:
        print(f"    dt0={dt0:<6g} dt={dt:<8g} [cached]")
        return cache[key]

    steps_equi = int(round(WINDOWS["equi_ms"] / dt))
    steps_meas = int(round(WINDOWS["meas_ms"] / dt))

    try:
        mf = RDMIsingModel(**PARAMS, dt=dt, dt0=dt0, hazard=hazard, eps=EPS, device=device)
    except ValueError as err:
        print(f"    dt0={dt0:<6g} dt={dt:<8g} skipped: {err}")
        store(key, None)
        return None

    t0 = time.time()
    mf.forward(steps_equi)
    out = mf.entropy_trajectory(steps_meas, pb=False)
    t1 = time.time()

    # Qm is sized from the KERNELS, not from the firing rate, so a slow regime -- which is what
    # a large dt0 produces -- can leave appreciable mass in the absorbing last bin and bias the
    # run. Worth seeing rather than silently plotting.
    tail_mass = mf.p[0][-1].item()
    if tail_mass > 1e-3:
        print(f"    dt0={dt0:<6g} [warn] {tail_mass:.1%} of the age mass sits in the "
              f"absorbing bin (Qm={mf.Qm[0]}); raise Qm or lower eps.")

    # every series is divided by dt: sigma and H are per STEP as reported, and a per-step
    # quantity plotted against a ms axis would make a fine-dt run look quiet purely because
    # its bins are narrow
    res = {"dt": dt, "dt0": dt0, "hazard": hazard, "Qm": mf.Qm[0],
           "t":     torch.arange(steps_meas, dtype=torch.get_default_dtype()) * dt,
           "r":     out["m_tot"] / dt,
           "sigma": out["sigma_tot"] / dt,
           "H_fwd": out["H_fwd_tot"] / dt,
           "H_rev": out["H_rev_tot"] / dt}

    print(f"    dt0={dt0:<6g} dt={dt:<8g} Qm={res['Qm']:<5d} "
          f"<r>={res['r'].mean() * HZ:.2f} Hz  "
          f"<sigma>={res['sigma'].mean() * HZ:.4g} nats/s  ({t1 - t0:.1f}s)")
    store(key, res)
    return res


def store(key: str, res: dict | None) -> None:
    """ Written out after every run, not at the end: the fine steps are the long ones, and
    an interrupted sweep should cost only the run it was on.

    Merged against whatever is on disk rather than dumping this process's dict over it. Two
    runs of this script (or of its sibling) share one file, and a plain overwrite silently
    deletes every point the other one computed while this one was working. """
    cache[key] = res
    CACHE_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        try:
            on_disk = torch.load(CACHE_FILE, weights_only=False)
        except Exception:
            on_disk = {}          # half-written file from a concurrent save; ours supersedes it
        on_disk.update(cache)
        cache.update(on_disk)
    torch.save(cache, CACHE_FILE)


cache: dict = {}
if CACHE_FILE.exists() and not overwrite:
    cache = torch.load(CACHE_FILE, weights_only=False)
    print(f"loaded {len(cache)} cached runs from {CACHE_FILE}")

if DT_FACTOR == 1.0:
    print("\nDT_FACTOR = 1, so dt = dt0 and lam0*dt = 1: the three hazards are the same\n"
          "kernel there and the three figures will be identical. Set DT_FACTOR < 1 to\n"
          "resolve each model below its own quantum and separate them.")

trajectories: dict[str, list] = {}
t_start = time.time()
for hazard in HAZARDS:
    print(f"\nhazard={hazard!r}  (mean field, dt = {DT_FACTOR:g} * dt0)")
    trajectories[hazard] = [run_traj(dt0 * DT_FACTOR, dt0, hazard) for dt0 in DT0_LIST]
print(f"\nall runs done in {time.time() - t_start:.0f}s")


# Plotting
# ~~~~~~~~

# Two forms of the same runs. Everything is stored as a RATE, so the per-step figure is the
# stored series multiplied back by that run's dt: what one update of the model actually emits,
# as against what it emits per unit time.
PANELS_RATE = (
    ("r",     "Rate  r  (Hz)",            "Firing rate"),
    ("sigma", r"$\sigma/dt$  (nats/s)",   "Entropy production rate"),
    ("H_fwd", r"$H_{fwd}/dt$  (nats/s)",  "Forward entropy rate"),
    ("H_rev", r"$H_{rev}/dt$  (nats/s)",  "Backward entropy rate"),
)

PANELS_STEP = (
    ("r",     r"Overlap  $m = r\,dt$",   "Overlap per step"),
    ("sigma", r"$\sigma$  (nats)",        "Entropy production per step"),
    ("H_fwd", r"$H_{fwd}$  (nats)",       "Forward entropy per step"),
    ("H_rev", r"$H_{rev}$  (nats)",       "Backward entropy per step"),
)

cmap = plt.get_cmap('viridis')
DT0_COLOR = {dt0: cmap(i / max(1, len(DT0_LIST) - 1) * 0.88)
             for i, dt0 in enumerate(DT0_LIST)}


def traj_figure(runs: list, panels: tuple, per_step: bool, hazard: str) -> None:
    """ One hazard's overlay, in one of its two forms.

    per_step multiplies by each run's OWN dt, which differs from curve to curve -- that factor
    is exactly what separates the two figures, and reading it off the run rather than from a
    global is what keeps them honest when DT_FACTOR is changed. """
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 11))
    fig.suptitle(f"Mean field,  hazard = {hazard!r},  dt = {DT_FACTOR:g} * dt0  "
                 f"--  {'per step (x dt)' if per_step else 'per second (Hz)'}", fontsize=13)

    for ax, (key, ylabel, name) in zip(axes, panels):
        ax.set_title(name, fontsize=10)
        for res in runs:
            y = res[key] * res["dt"] if per_step else res[key] * HZ
            ax.plot(res["t"], y, '-', color=DT0_COLOR[res["dt0"]], linewidth=1.6)
        ax.set_ylabel(ylabel); ax.grid()
    axes[-1].set_xlabel('Time (ms)')

    handles = [Line2D([], [], color=DT0_COLOR[r["dt0"]], linewidth=1.8,
                      label=f"dt0 = {r['dt0']:g}  (dt = {r['dt']:g})") for r in runs]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=min(len(handles), 6), fontsize=9, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))


if run_plot:
    for hazard in HAZARDS:
        runs = [r for r in trajectories[hazard] if r is not None]
        if not runs:
            continue
        # the two forms of one hazard back to back, so they can be flipped between rather
        # than hunted for among the other hazards' figures
        traj_figure(runs, PANELS_RATE, per_step=False, hazard=hazard)
        traj_figure(runs, PANELS_STEP, per_step=True,  hazard=hazard)

    plt.show()
