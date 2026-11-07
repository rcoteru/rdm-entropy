from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import math

import torch

""" Small utilities for the scripts -- the cache naming and the plotting-stage helpers --
rather than for the model. The file exists so that an array utility or a path rule does not
have to live in a physics module to be shared between scripts. """


def grab_closest_idxs(vector: torch.Tensor, values: Sequence[float]) -> list[int]:
    """ Indices of the entries of `vector` closest to each of `values`. """
    return [int(torch.argmin((vector - value).abs())) for value in values]


def cycle_window(x: torch.Tensor, rel_tol: float = 1e-9) -> tuple[int, int, int]:
    """ (start, stop, n_cycles) spanning a whole number of periods of `x`.

    For averaging a periodic signal: a window that is not a whole number of periods leaves a
    phase error of order amplitude/n_periods, which on a burst whose amplitude is several times
    its mean is larger than most things worth measuring. Delimited by upward crossings of the
    mean rather than by peaks -- a crossing of a smooth signal is located to within one step,
    while the peak of a flat-topped burst is not.

    A settled state has no crossings to find and returns the whole window with n_cycles = 0,
    which is the honest answer: there are no cycles to align to and none are needed. So is a
    NOISY signal's answer meaningful only when the oscillation is deterministic -- a finite-N
    trajectory crosses its mean many times per period, and this will happily report those as
    cycles. Use it on the mean field, and a plain long average on the samples.

    `rel_tol` is the peak-to-peak, relative to the mean, below which the signal counts as
    settled. """
    if float(x.max() - x.min()) < rel_tol * float(x.mean().abs()):
        return 0, len(x), 0
    y  = x - x.mean()
    up = ((y[:-1] <= 0) & (y[1:] > 0)).nonzero().flatten()
    if len(up) < 2:
        return 0, len(x), 0
    return int(up[0]), int(up[-1]), len(up) - 1


# Durations
# ~~~~~~~~~

def n_steps(ms: float, dt: float, *, name: str = "duration") -> int:
    """ Steps for a physical duration of `ms` milliseconds at solver step `dt`.

    Runs are specified in ms, not in steps. A step count is a statement about the solver and
    says nothing about the system: the same 20000 steps is 20000 ms at dt = 1 and 4000 ms at
    dt = 0.2, so a dt refinement done on a fixed step count silently shortens the window it
    measures, and two scripts at different dt cannot be compared by the numbers at the top of
    them. In ms, equilibration is "long enough for the slowest timescale in the model" and the
    recording is "this many periods", which are the statements actually meant.

    Rounds to the nearest step, so a duration that is not a multiple of dt costs at most half a
    step; the realized duration is `n_steps(ms, dt) * dt`, which is what a dump should report
    if the difference ever matters. Raises below one step, which is a duration that was meant
    in ms and given in steps, or a dt that is coarser than the window. """
    if not ms > 0 or not dt > 0:
        raise ValueError(f"{name} and dt must be positive, got {name}={ms}, dt={dt}")
    n = int(round(ms / dt))
    if n < 1:
        raise ValueError(f"{name}={ms} ms is shorter than one step of dt={dt} ms")
    return n


# Cache naming
# ~~~~~~~~~~~~

def _stable(v):
    """ A value's identity for hashing, normalized so that equal runs hash equal.

    Numbers collapse to float, because J=4 and J=4.0 are the same model and would otherwise
    get separate caches. Tensors hash by their contents rather than by their id, which is what
    makes this work for the swept axes of the grid scripts. Mappings sort by key, so a dict
    rebuilt in a different order is still the same run. """
    if torch.is_tensor(v):
        return ("tensor", tuple(v.flatten().tolist()), tuple(v.shape))
    if isinstance(v, Mapping):
        return tuple((k, _stable(v[k])) for k in sorted(v))
    if isinstance(v, (list, tuple)):
        return tuple(_stable(x) for x in v)
    if isinstance(v, bool):          # before int: True == 1, but it does not mean 1
        return ("bool", v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def cache_path(script: str, key: Mapping, *, tag: str = "",
               root: str | Path = "cache", suffix: str = ".pt") -> Path:
    """ Cache file for `script` run at parameters `key`, named so the parameters pick the file.

    Changing any parameter changes the hash, so a run never finds a cache another parameter
    set wrote and there is no `overwrite` flag to remember -- the old file simply stays where
    it is, which also means switching between parameter sets costs nothing after the first
    run of each. `tag` is a readable prefix (a regime name, say) so the directory can still be
    read by eye.

    What this does NOT catch is a change to the code: the same parameters through a fixed
    model still hash the same. That is the one case still worth an explicit re-run. """
    h = hashlib.sha1(repr(_stable(key)).encode()).hexdigest()[:8]
    name = "_".join(p for p in (script, tag, h) if p)
    return Path(root) / f"{name}{suffix}"


# Text width of the SciPost class (`article`, 11pt, a4paper, 30 mm side margins), in inches:
# 210mm - 2*30mm = 150mm. Figures drawn at this width go in with `\includegraphics` at scale
# 1, so nothing is resampled and the font sizes below survive into the PDF.
SCIPOST_TEXTWIDTH = 150 / 25.4

# Body size of the class; its `caption` setup inherits it. Figure text is set a bit smaller
# by default, which is conventional and still comfortably legible at this width.
SCIPOST_FONTSIZE = 11.0

_FONTS = {
    # SciPost sets its text in Bitstream Charter (the class pulls in `mathdesign`, which is
    # loaded with the charter option). matplotlib ships no Charter, and TeX's copy is Type 1,
    # which only the usetex path can use -- so without usetex we fall back through any system
    # Charter to STIX Two, a Times-like face that is far closer to Charter than CM is.
    "charter": (["XCharter", "Charter", "Bitstream Charter", "STIX Two Text"], "stix",
                r"\usepackage[charter]{mathdesign}"),
    "cm":      (["cmr10", "Computer Modern Roman", "DejaVu Serif"], "cm",
                r"\usepackage{lmodern}"),
    "stix":    (["STIX Two Text", "DejaVu Serif"], "stix",
                r"\usepackage{stix2}"),
}


def pub_style(font: str = "charter", base: float = SCIPOST_FONTSIZE - 2,
              usetex: bool = False) -> None:
    """ Publication rcParams, defaulting to the SciPost class's look. `font` picks a family
    from `_FONTS`; `base` is the figure's body size in points. `usetex=True` routes every
    label through a real LaTeX run, which is the only way to match the document exactly --
    it needs the font package installed, and dvipng for on-screen output, though saving to
    PDF needs neither. The mathtext path is the fast default and differs only in the math
    glyphs. """
    import warnings

    import matplotlib as mpl
    from matplotlib import font_manager

    if font not in _FONTS:
        raise ValueError(f"unknown font '{font}', expected one of {sorted(_FONTS)}")
    family, mathfont, preamble = _FONTS[font]

    # matplotlib falls back to DejaVu silently when a family is missing, which is easy not to
    # notice until the figure is next to the text it should match -- so say so.
    if not usetex:
        def installed(name: str) -> bool:
            try:  # findfont raises rather than returning None when the fallback is disabled
                font_manager.findfont(name, fallback_to_default=False)
                return True
            except ValueError:
                return False

        found = next((f for f in family if installed(f)), None)
        if found != family[0]:
            warnings.warn(f"'{family[0]}' not installed, falling back to "
                          f"'{found or 'matplotlib default'}'")

    mpl.rcParams.update({
        "text.usetex":          usetex,
        "text.latex.preamble":  preamble,
        "font.family":          "serif",
        "font.serif":           family,
        "mathtext.fontset":     mathfont,
        # cmr10 has no proper Unicode minus; let mathtext draw it instead of boxing it
        "axes.unicode_minus":   font != "cm",
        "axes.formatter.use_mathtext": True,

        "font.size":            base,
        "axes.titlesize":       base,
        "axes.labelsize":       base,
        "xtick.labelsize":      base - 1,
        "ytick.labelsize":      base - 1,
        "legend.fontsize":      base - 1,

        "axes.linewidth":       0.6,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.top":            True,
        "ytick.right":          True,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "lines.linewidth":      1.2,
        "legend.frameon":       False,

        "figure.dpi":           150,
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })


# Age-grid sizing
# ~~~~~~~~~~~~~~~

def fit_age_grid(build, *, tol: float = 1e-2, equi: int = 20000,
                 max_doublings: int = 6, verbose: bool = True) -> list[int]:
    """ The smallest age grid that keeps `truncation_error()` below `tol`.

    There is nothing fitted about the criterion -- `truncation_error` is a rigorous relative
    bound on the firing-rate error (see RDMNetworkBatch.truncation_error), so `tol = 1e-2`
    means "the rate is right to 1%". What this routine does is solve `err(Q) < tol` for Q,
    which has to be done numerically because `err` depends on the attractor and the attractor
    depends on Q.

    `build(qm)` takes a `Qm` argument -- `None` for the model's own `q_from_tau` default, or a
    list of ints -- and returns a model. The default is the right first guess and the wrong
    answer whenever the firing interval is comparable to `Q*dt`, where mass reaches the
    absorbing last bin and the run converges on a different attractor.

    The search doubles until the bound is met, then takes one log-linear interpolation step
    back: `err` falls geometrically in Q (mass beyond U_max and the kernel tails both decay
    exponentially), so two bracketing measurements predict the crossing well. The interpolated
    Q is verified before being returned and the bracketing Q is kept if it fails, so the
    result always satisfies the tolerance -- the interpolation only trims the overshoot that
    doubling alone leaves behind, which can be a factor of two in Q and four in memory.

    `equi` must take the pilot to the SAME state the caller will measure -- for a script that
    equilibrates and then records, that is `EQUI + STEPS`, not `EQUI`. The boundary mass
    settles on the slowest timescale in the problem and the attractor keeps taking it on after
    equilibration ends, so a pilot stopped earlier under-reports and the caller's own assertion
    then fires. That is not a false alarm; it is this argument being set too short.

    Sized on the ATTRACTOR, deliberately. A fixed point holds more mass at old ages than the
    orbit around it does, so measuring there would ask for a larger grid -- but that mass is
    not what the dynamics is made of, and the attractor is the object every reported quantity
    describes. Fixed-point work done at the same parameters inherits this grid.

    Qm is a solver setting, not a model parameter: the physics is fixed by the couplings and
    the timesteps. So a script should fit it here rather than declaring a number, and should
    put `tol` in its cache key rather than the resulting Q. """
    def measure(qm):
        mf = build(qm)
        qm = list(mf.Qm)
        mf.forward(equi, pb=False)
        err = mf.truncation_error()
        err = float(err.max()) if torch.is_tensor(err) else float(err)
        del mf
        if math.isnan(err):
            raise RuntimeError(
                f"truncation_error is nan at Qm={qm}: the pilot run itself diverged, so "
                "there is no grid size that will satisfy the tolerance. Check the swept "
                "axis for degenerate values (beta = 0 gives an infinite R_abs) before "
                "raising tol or max_doublings.")
        if verbose:
            print(f"  age grid: Qm={qm} -> truncation_error={err:.2e}"
                  f"{'  [ok]' if err < tol else '  [too coarse]'}")
        return qm, err

    qm, err = measure(None)
    if err < tol:
        return qm

    for _ in range(max_doublings):
        lo, lo_err = qm, err
        qm, err = measure([2 * q for q in qm])
        if err < tol:
            break
    else:
        raise RuntimeError(f"truncation_error still {err:.2e} at Qm={qm}; raise "
                           f"max_doublings, or check the model is not diverging")

    # Log-linear interpolation between the bracketing pair, then verify. Aimed at tol/3
    # rather than tol: the interpolation is only approximate, so targeting the threshold
    # itself lands on either side of it with even odds and wastes the probe half the time.
    if lo_err > 0 and err > 0:
        target = tol / 3
        f = (math.log(lo_err) - math.log(target)) / (math.log(lo_err) - math.log(err))
        mid = [max(a + 1, int(a + f * (b - a))) for a, b in zip(lo, qm)]
        if all(a < m < b for a, m, b in zip(lo, mid, qm)):
            mid_qm, mid_err = measure(mid)
            if mid_err < tol:
                return mid_qm
    return qm


# EPR memory
# ~~~~~~~~~~

# Measured on the sliding path, float64, by sweeping `chunk` at fixed (B, Q) and fitting the
# peak allocation (scripts in the repo history; the fit is linear to three digits):
#
#   peak = (1 + not inplace) * B * Q^2 * s   +   3 * chunk * Q^2 * s
#
# The first term is persistent for the whole trajectory: one ring buffer buff_S of (B, Q, Q),
# plus -- when `inplace` is off -- the (B, Q, Q) gather that init_epr_state's phi_end is a view
# into, which therefore stays pinned until the run ends. The second is transient, the chain of
# joint-sized temporaries inside init_epr_state, and is the one `chunk` caps.
#
# Q is SQUARED here and B is not: an age grid that comes back twice as long is four times the
# memory, which is why fit_age_grid's result has to be fed to this and not just to the model.
EPR_PERSISTENT = 1.0   # ring buffer, per batch element
EPR_PINNED     = 1.0   # the gather phi_end views, per batch element, out-of-place only
EPR_TRANSIENT  = 3.0   # init_epr_state's temporaries, per chunk element


def epr_peak_bytes(B: int, Q: int, chunk: int | None = None, *,
                   inplace: bool = True, itemsize: int = 8) -> int:
    """ Peak memory of `entropy_trajectory` for a batch of B on an age grid of Q bins.

    The estimate the scripts size their tiles with. Sizing on the ring buffer alone -- the
    obvious thing, since it is the only allocation named in the signature -- under-counts by
    between 2x and 5x, which is the usual way one of these scans dies. """
    cs = B if chunk is None else min(chunk, B)
    per_b = EPR_PERSISTENT + (0.0 if inplace else EPR_PINNED)
    return int((per_b * B + EPR_TRANSIENT * cs) * Q * Q * itemsize)


def device_budget(device: str | torch.device, fraction: float = 0.8) -> int:
    """ Bytes a run may use on `device`: a fraction of what is actually free right now.

    Read rather than declared, so the same script runs on a different card without a constant
    to edit, and so a budget set while something else holds half the card does not OOM. """
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(fraction * free)
    with open("/proc/meminfo") as f:                      # MemAvailable, in kB
        avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
    return int(fraction * avail * 1024)


def epr_plan(B: int, Q: int, *, device: str | torch.device, budget: int | None = None,
             inplace: bool = True, itemsize: int = 8, verbose: bool = True
             ) -> tuple[int, int]:
    """ The (tile size, chunk) that keep an EPR run of B systems inside the memory budget.

    Returns a batch size to split the scan into and a `chunk` to pass to entropy_trajectory;
    `budget` defaults to most of what is free on the device. The split gives three quarters of
    the budget to the tile and a quarter to the chunk: the tile is what costs wall time (each
    one re-pays the equilibration), while the chunk only paces init_epr_state, which runs once
    per tile. A tile of 1 is possible and means the grid is too fine for this card at this Q --
    the fix then is a shorter age grid (a looser `tol`) or a coarser scan, not a smaller tile.

    On the sliding path -- the default, and what every scan here uses -- `chunk` costs nothing
    but the init's speed, so there is no reason for a script to set it by hand. """
    budget = device_budget(device) if budget is None else budget
    unit   = Q * Q * itemsize
    # The split below saturates whatever it is given exactly, so it is given nine tenths: the
    # formula counts the (Q, Q)-shaped objects and nothing else, and the model's own O(B * Q)
    # state puts the true peak 1-3% above it. Measured, not guessed -- see the test.
    cap    = 0.9 * budget / unit               # batch-elements' worth of (Q, Q) planes
    per_b  = EPR_PERSISTENT + (0.0 if inplace else EPR_PINNED)

    B_tile = max(1, min(B, int(0.75 * cap / per_b)))
    chunk  = max(1, min(B_tile, int(0.25 * cap / EPR_TRANSIENT)))
    while B_tile > 1 and epr_peak_bytes(B_tile, Q, chunk, inplace=inplace,
                                        itemsize=itemsize) > budget:
        B_tile = max(1, B_tile // 2)           # only reachable through the rounding above
        chunk  = min(chunk, B_tile)

    if verbose:
        peak = epr_peak_bytes(B_tile, Q, chunk, inplace=inplace, itemsize=itemsize)
        n = -(-B // B_tile)
        print(f"  epr plan: Q={Q}, {n} tile(s) of <= {B_tile} of {B}, chunk={chunk} -> peak "
              f"{peak / 2**30:.2f} GiB against a {budget / 2**30:.2f} GiB budget")
    return B_tile, chunk


# The dt -> 0 limit
# ~~~~~~~~~~~~~~~~~

# CLAUDE.md's reporting rule: every number comes with a dt -> 0 check, and the default is
# Richardson extrapolation from a coarse pair (dt, dt/2) with `poisson_exact`. These three
# are that rule. They were written in and for ising_dt-conv.py (now S1-dt_conv.py) and moved
# here when the cycle-point, cost-table and API-check studies all needed the same refusal: a
# guard that three scripts carry their own copy of is three guards that can disagree.

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
    answer; a plausible-looking limit with a relative-error column beside it is not.

    A bare PAIR (len 2) has no order to read and no sign change to see, so it is extrapolated
    at first order unconditionally -- which is why the standing rule asks the pair's own order
    to be reported from somewhere else, usually a third, coarser step. """
    if len(vals) < 2:
        raise ValueError("richardson needs at least a (dt, dt/2) pair")
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


def converged_at(dts: Sequence[float], vals: torch.Tensor, limit: float,
                 thresh: float) -> float:
    """ Coarsest dt from which the relative error stays under `thresh` all the way down.
    'And stays': a curve crossing a threshold once on its way past is not converged there. """
    err = (vals - limit).abs() / max(abs(limit), 1e-300)
    ok = float('nan')
    for k in range(len(dts) - 1, -1, -1):
        if float(err[k]) < thresh: ok = dts[k]
        else: break
    return ok
