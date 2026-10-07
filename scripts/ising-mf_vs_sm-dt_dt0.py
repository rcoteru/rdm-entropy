""" Model drift along dt = dt0, and where refining dt takes each hazard from there.

The model carries two time scales that are easy to conflate (see rdme/__init__.py):

  dt0  the model's time quantum, which fixes the rate calibration lam0 = 1/dt0. Changing it
       walks across a FAMILY of different continuous-time models.
  dt   the resolution of the solution. Refining it refines the NUMERICS of one of them, and
       every observable with a continuum limit must settle as dt -> 0.

The figure puts both on one axis. Everything is plotted against dt0, and for each dt0:

  the DIAGONAL dt = dt0 (black) is the discrete model sitting at its own quantum. This is
  where lam0*dt = 1 and all three hazards collapse onto the sigmoid, so there is one black
  curve, not three -- it is the drift of the discrete model as its quantum is moved, with no
  numerical refinement anywhere in it.

  the BRANCHES drop dt below dt0 at that same fixed model, one line per hazard per refinement
  dt = dt0/2, /4, /8 (fainter/thinner = coarser). They answer what the diagonal cannot:
  starting from the same discrete model, do the three hazards converge, do they converge to the
  same place, and how far is the converged value from the diagonal point you started at.

What to expect: r and the EPR RATE sigma/dt have dt -> 0 limits for "async" and "escape", so
those branches should stack up and stop moving. "synchronous" has none -- it fixes the firing
probability per BIN, so the rate it implies is sigma(h)/dt and diverges as dt shrinks -- and
its branches should run away instead, which is the clearest picture in the script of why the
other two exist. H_fwd and H_rev have no limit for any hazard: they are entropies of a binned
process against a counting measure and each diverges like -r ln dt. They are plotted anyway
because the drift is worth seeing, but only their diagonal values are comparable.

The spin model (open markers) is overlaid on the mean field (lines) everywhere, so each panel
is also a finite-N check; panel 6 is that gap on its own, since at the scale of the others a
1% discrepancy is invisible.

Points are cached individually, keyed by the full physics + window signature, so an
interrupted run resumes and adding a dt0 or a refinement only costs the new points.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from rdme.single import RDMIsingModel
from rdme.spin_model import SpinIsingModel


# Cache
# ~~~~~

CACHE_DIR  = Path(__file__).parents[1] / "cache"
CACHE_FILE = CACHE_DIR / f"{Path(__file__).stem}.pt"

overwrite = False   # recompute every point and replace the cache file
run_plot  = True


# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

N = 20000     # neurons in the spin model; the mean field is the N -> inf limit

HAZARDS = ["synchronous", "async", "escape"]
HZ_COLOR = {"synchronous": "#6b6b6b", "async": "#2e6fb7", "escape": "#b4453c"}

# integration-based oscillatory regime -- a genuinely driven state, so sigma is well clear of
# zero and the sweep measures something. At a stable fixed point the EPR is ~1e-9 and the EPR
# panels would just be plotting their own sampling noise.
PARAMS = dict(J=4,          # coupling strength, in physical units (per ms)
              E=1,          # external field
              beta=30,
              theta=1,
              tau_int=12,
              tau_ref=3,
              K_ref=0)

# the models to visit (the diagonal), and the refinements to hang off each one. dt = dt0*f,
# so f = 1 IS the diagonal and every f <= 1 keeps lam0*dt <= 1, which is what "async" needs.
DT0_LIST = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
DT_FACTORS = [1.0, 1/2, 1/4, 1/8]

# Windows are in MILLISECONDS, not steps: every run has to cover the same physical time, or a
# sweep over dt is partly a sweep over how long the system was watched.
WINDOWS = dict(equi_ms=600.0,   # discarded equilibration
               meas_ms=600.0,   # reported window
               skip_ms=60.0,    # trimmed from each end of the window before averaging
               tail_ms=120.0)   # spin lookahead for the reverse recursion; > longest ISI

EPS = 0.01    # mean-field age truncation


# Measurement
# ~~~~~~~~~~~

def cache_key(dt: float, dt0: float, hazard: str) -> str:
    """ Everything a point depends on, so a changed parameter misses rather than silently
    returning the old number. The spin half is stochastic, so a cached point is one
    realization of it -- delete the file to resample. """
    sig = {**PARAMS, **WINDOWS, "N": N, "eps": EPS,
           "dt": dt, "dt0": dt0, "hazard": hazard}
    return "|".join(f"{k}={sig[k]!r}" for k in sorted(sig))


def run_point(dt: float, dt0: float, hazard: str) -> dict | None:
    """ One (dt, dt0, hazard) through both models, reduced to stationary averages.

    Entropy quantities come back as RATES (per ms) so they are comparable along a sweep; the
    per-step sigma is kept alongside as "sigma_bin" to make the point that it is not.
    """
    key = cache_key(dt, dt0, hazard)
    if not overwrite and key in cache:
        res = cache[key]
        if res is None:
            return None
        print(f"    dt={dt:<9g} Qm={res['Qm']:<5d} r={res['mf_r']:.5f}/{res['sm_r']:.5f}  "
              f"sigma={res['mf_sigma']:.5f}/{res['sm_sigma']:.5f}   [cached]")
        return res

    steps_equi = int(round(WINDOWS["equi_ms"] / dt))
    steps_meas = int(round(WINDOWS["meas_ms"] / dt))
    skip = max(1, int(round(WINDOWS["skip_ms"] / dt)))
    tail = max(4, int(round(WINDOWS["tail_ms"] / dt)))

    try:
        mf = RDMIsingModel(**PARAMS, dt=dt, dt0=dt0, hazard=hazard, eps=EPS, device=device)
        sm = SpinIsingModel(N, **PARAMS, dt=dt, dt0=dt0, hazard=hazard,
                            device=device, ic="silent")
    except ValueError as err:
        print(f"    dt={dt:<9g} skipped: {err}")
        store(key, None)
        return None

    t0 = time.time()
    mf.forward(steps_equi)
    mf_out = mf.entropy_trajectory(steps_meas, pb=False)
    t1 = time.time()
    sm.forward(steps_equi)
    sm_out = sm.entropy_trajectory_chunked(steps_meas, chunk=max(512, tail), overlap=tail,
                                           check_overlap=True)
    t2 = time.time()

    # Qm is sized from the KERNELS, not from the firing rate, so a slow regime -- which is what
    # a large dt0 produces -- can leave appreciable mass in the absorbing last bin and bias the
    # mean field. Worth seeing rather than silently averaging in.
    tail_mass = mf.p[0][-1].item()
    if tail_mass > 1e-3:
        print(f"    dt={dt:<9g} [warn] {tail_mass:.1%} of the age mass sits in the absorbing "
              f"bin (Qm={mf.Qm[0]}); raise Qm or lower eps.")

    def reduce(out, k, rate=True):
        v = out[k][skip:-skip].mean().item()
        return v / dt if rate else v

    res = {"dt": dt, "dt0": dt0, "hazard": hazard, "Qm": mf.Qm[0]}
    for tag, out in (("mf", mf_out), ("sm", sm_out)):
        res[f"{tag}_r"]         = reduce(out, "m_tot")
        res[f"{tag}_sigma"]     = reduce(out, "sigma_tot")
        res[f"{tag}_H_fwd"]     = reduce(out, "H_fwd_tot")
        res[f"{tag}_H_rev"]     = reduce(out, "H_rev_tot")
        res[f"{tag}_sigma_bin"] = reduce(out, "sigma_tot", rate=False)

    print(f"    dt={dt:<9g} Qm={res['Qm']:<5d} r={res['mf_r']:.5f}/{res['sm_r']:.5f}  "
          f"sigma={res['mf_sigma']:.5f}/{res['sm_sigma']:.5f}  "
          f"({t1 - t0:.1f}s + {t2 - t1:.1f}s)")
    store(key, res)
    return res


def store(key: str, res: dict | None) -> None:
    """ Written out after every point, not at the end: the fine steps are the long ones, and
    an interrupted sweep should cost only the point it was on.

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
    print(f"loaded {len(cache)} cached points from {CACHE_FILE}")

# points[hazard][factor] is a list over DT0_LIST. The f = 1 row lives in `diagonal` and is
# computed ONCE: at lam0*dt = 1 the three hazards are the same kernel, and running it three
# times would only invite the reader to wonder why three "different" curves coincide.
points: dict[str, dict[float, list]] = {h: {f: [] for f in DT_FACTORS if f != 1.0}
                                        for h in HAZARDS}
diagonal: list = []

t_start = time.time()
for dt0 in DT0_LIST:
    print(f"\ndt0 = {dt0}  (mean field / spin model, N={N})")
    print("  diagonal, dt = dt0  -- all hazards coincide here")
    diagonal.append(run_point(dt0, dt0, HAZARDS[0]))
    for hazard in HAZARDS:
        print(f"  hazard={hazard!r}, refining dt below dt0")
        for f in DT_FACTORS:
            if f == 1.0:
                continue
            points[hazard][f].append(run_point(dt0 * f, dt0, hazard))
print(f"\nall points done in {time.time() - t_start:.0f}s")


# Plotting
# ~~~~~~~~

def series(row: list, key: str, per_step: bool = False) -> tuple[list[float], list[float]]:
    """ (dt0, value) over a row of points, dropping the ones that were skipped.

    per_step multiplies by that point's OWN dt, which is what turns a rate back into the
    per-step quantity the model actually reports. dt varies along a branch (dt = dt0*f), so
    the factor has to come from the point and not from the row. """
    pts = [p for p in row if p is not None]
    return ([p["dt0"] for p in pts],
            [p[key] * p["dt"] if per_step else p[key] for p in pts])


# Per-ms panels: everything divided by dt, which is the form with a continuum limit.
PANELS_RATE = (
    ("r",         "Firing rate (1/ms)",              "Rate  r"),
    ("sigma",     "EPR rate (nats/ms)",              r"EPR rate  $\sigma/dt$"),
    ("sigma_bin", "EPR per step (nats)",             r"EPR per step  $\sigma$"),
    ("H_fwd",     "Forward entropy rate (nats/ms)",  r"$H_{fwd}/dt$"),
    ("H_rev",     "Backward entropy rate (nats/ms)", r"$H_{rev}/dt$"),
)

# Per-step panels: the same quantities multiplied back by dt, i.e. what one update of the
# model emits. sigma_bin is left out because it is ALREADY per step -- it appears in the
# per-ms figure precisely so the two forms can be seen side by side there, and multiplying
# it by dt again would be a quantity with no meaning.
PANELS_STEP = (
    ("r",     "Overlap per step",             r"Overlap  $m = r\,dt$"),
    ("sigma", "EPR per step (nats)",          r"EPR per step  $\sigma$"),
    ("H_fwd", "Forward entropy per step (nats)",  r"$H_{fwd}$"),
    ("H_rev", "Backward entropy per step (nats)", r"$H_{rev}$"),
)

# coarse -> fine reads as faint/thin -> solid/thick, so a branch that is still moving shows up
# as a spread and a converged one as a single heavy line
REFINEMENTS = [f for f in DT_FACTORS if f != 1.0]
ALPHA = {f: 0.35 + 0.65 * i / max(1, len(REFINEMENTS) - 1) for i, f in enumerate(REFINEMENTS)}
LW    = {f: 0.9 + 1.1 * i / max(1, len(REFINEMENTS) - 1) for i, f in enumerate(REFINEMENTS)}


def sweep_figure(panels: tuple, per_step: bool, subtitle: str) -> None:
    """ The sweep in one of its two forms: per ms (per_step=False) or per step (True).

    Both are drawn from the same stored rates, so the two figures cannot disagree about
    anything except the dt factor that distinguishes them. """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Discrete model along dt = dt0 (black), and where refining dt takes each "
                 f"hazard from there  --  {subtitle},  N={N}", fontsize=13)

    for ax, (key, ylabel, name) in zip(axes.flat, panels):
        ax.set_title(name)

        for hazard in HAZARDS:
            for f in REFINEMENTS:
                row = points[hazard][f]
                x, y = series(row, f"mf_{key}", per_step)
                ax.plot(x, y, '-', color=HZ_COLOR[hazard], alpha=ALPHA[f], linewidth=LW[f])
                xs, ys = series(row, f"sm_{key}", per_step)
                ax.plot(xs, ys, 'o', color=HZ_COLOR[hazard], markersize=3.5,
                        markerfacecolor='none', alpha=ALPHA[f])

        # the diagonal last, so it sits on top of the branches that leave it
        x, y = series(diagonal, f"mf_{key}", per_step)
        ax.plot(x, y, 'k-', linewidth=2.6, marker='o', markersize=6, zorder=5)
        xs, ys = series(diagonal, f"sm_{key}", per_step)
        ax.plot(xs, ys, 'ko', markersize=10, markerfacecolor='none', zorder=5)

        ax.set_xscale('log'); ax.set_xlabel('dt0 (ms)'); ax.set_ylabel(ylabel)
        ax.grid()

    # the finite-N check on its own: at the scale of the panels above, a 1% gap is invisible.
    # Normalised by the LARGEST mean-field value of each quantity over the whole sweep rather
    # than pointwise: sigma passes through zero at the top of the dt0 range, and a pointwise
    # relative error there is a finite-N discrepancy divided by nothing, which reads as 1e5
    # and buries the three decades that actually mean something.
    ax = axes.flat[len(panels)]
    ax.set_title('Spin model vs mean field')
    all_rows = [diagonal] + [points[h][f] for h in HAZARDS for f in REFINEMENTS]
    scale = {}
    for key in ("r", "sigma"):
        vals = [abs(v) for row in all_rows for v in series(row, f"mf_{key}", per_step)[1]]
        scale[key] = max(vals) if vals else 1.0
    for hazard in HAZARDS:
        for f in DT_FACTORS:
            row = diagonal if f == 1.0 else points[hazard][f]
            for key, mark in (("r", 'o'), ("sigma", 's')):
                x, mf_v = series(row, f"mf_{key}", per_step)
                _, sm_v = series(row, f"sm_{key}", per_step)
                rel = [abs(s - m) / scale[key] for s, m in zip(sm_v, mf_v)]
                ax.plot(x, rel, mark, color='k' if f == 1.0 else HZ_COLOR[hazard],
                        markersize=4, markerfacecolor='none', alpha=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('dt0 (ms)'); ax.set_ylabel('|spin - mf| / max|mf|')
    ax.grid()
    ax.legend(handles=[Line2D([], [], color='k', marker='o', linestyle='none',
                              markerfacecolor='none', label=r'$m$' if per_step else r'$r$'),
                       Line2D([], [], color='k', marker='s', linestyle='none',
                              markerfacecolor='none',
                              label=r'$\sigma$' if per_step else r'$\sigma/dt$')],
              fontsize=8, loc='best')

    for ax in axes.flat[len(panels) + 1:]:
        ax.axis('off')

    # one legend for the whole figure: colour is the hazard, weight is the refinement
    handles = [Line2D([], [], color='k', linewidth=2.6, marker='o',
                      label='dt = dt0 (all hazards)')]
    handles += [Line2D([], [], color=HZ_COLOR[h], linewidth=2.0, label=h) for h in HAZARDS]
    handles += [Line2D([], [], color='0.4', alpha=ALPHA[f], linewidth=LW[f],
                       label=f'dt = dt0/{round(1 / f)}') for f in REFINEMENTS]
    handles += [Line2D([], [], color='0.4', marker='o', linestyle='none',
                       markerfacecolor='none', label=f'spin model (N={N})')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955),
               ncol=len(handles), fontsize=9, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))


if run_plot:
    sweep_figure(PANELS_RATE, per_step=False, subtitle="per ms")
    sweep_figure(PANELS_STEP, per_step=True,  subtitle="per step (x dt)")
    plt.show()
