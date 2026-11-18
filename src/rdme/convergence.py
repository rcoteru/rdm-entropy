""" Is the solver step fine enough: dt, extrapolated to its dt -> 0 limit. The age grid needs
no check of this kind -- equations.age_window sizes it from a bound -- but every reported
number has to come with this one. """

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

import rdme.helpers as hlp

# The dt -> 0 limit
# ~~~~~~~~~~~~~~~~~

# The reporting rule: every number comes with a dt -> 0 check, and the default is
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
                          f"{[f'{v:.2f}' for v in o.tolist()]}", RuntimeWarning, stacklevel=2)
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


# Burst resolution
# ~~~~~~~~~~~~~~~~
#
# Whether the solver step resolves the population burst, which is what the dt -> 0 limit of an
# oscillating run hinges on. The measured quantity is the burst FWHM, and the question is not
# "is it locked to the grid" but "what is its continuum width w0, and how many bins of dt fit
# in it". Three steps (dt, dt/2, dt/4) give w0 by Richardson; two cannot separate a width that
# converges slowly from one that collapses.
#
# Measured on S1 (2026-11-16, both route-B kernels): w0 falls roughly exponentially with J
# (2.1 ms at J = 17, 0.5 at 20, 0.07 at 25, ~0.005 at 45), all at p ~ 0.9-1. There is no sharp
# boundary: resolution is a property of w0/dt, and the cutoff in J moves slowly with dt.

def watch_bursts(mf, t_ms: float) -> torch.Tensor:
    """ Step a batched model through `t_ms` ms and return its N-weighted activity r (B, T) in
    1/ms, on CPU: the series burst_fwhm reads. """
    n = hlp.n_steps(t_ms, mf.dt, name="t_ms")
    r = torch.empty(mf.batch_shape + (n,), device=mf.device, dtype=torch.float64)
    for t in range(n):
        r[..., t] = mf.activity()
        mf.update()
    return r.cpu()


def burst_fwhm(r: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ Relative amplitude (max - min)/mean and mean burst FWHM in ms, per row of r (B, T).

    Bursts are the excursions above the window's mid-level. Each one's half maximum sits
    halfway between its peak and the window minimum (= peak/2 when the trough is silent, and
    still defined near a Hopf, where peak/2 can fall below the trough). Crossings are linearly
    interpolated between bins, which matters when a burst is two bins wide; only bursts with
    both crossings inside the window count. NaN where there are none. """
    amp  = (r.amax(-1) - r.amin(-1)) / r.mean(-1)
    fwhm = torch.full(amp.shape, float("nan"), dtype=torch.float64)
    for b, x in enumerate(r.numpy()):
        lo  = x.min()
        mid = 0.5 * (lo + x.max())
        widths = []
        for s in np.flatnonzero(np.diff((x > mid).astype(int)) == 1) + 1:
            e = s
            while e < len(x) and x[e] > mid:
                e += 1
            if e == len(x):
                break
            k    = s + int(np.argmax(x[s:e]))
            half = 0.5 * (x[k] + lo)
            i, j = k, k
            while i > 0 and x[i - 1] >= half:
                i -= 1
            while j < len(x) - 1 and x[j + 1] >= half:
                j += 1
            if i == 0 or j == len(x) - 1:
                continue
            left  = (i - 1) + (half - x[i - 1]) / (x[i] - x[i - 1])
            right = j + (x[j] - half) / (x[j] - x[j + 1])
            widths.append((right - left) * dt)
        if widths:
            fwhm[b] = float(np.mean(widths))
    return amp, fwhm


def burst_limit(fwhm: torch.Tensor) -> torch.Tensor:
    """ Continuum burst width w0 per column of fwhm (3, B), whose rows are the widths at dt,
    dt/2, dt/4. NaN where richardson refuses (turning point or wild order). """
    w0 = torch.full(fwhm.shape[1:], float("nan"), dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(fwhm.shape[1]):
            if not fwhm[:, b].isnan().any():
                w0[b] = richardson(fwhm[:, b])[0]
    return w0


BURST_CLASSES = ("no_osc", "resolved", "unresolved", "unconverged")
OSC_MIN = 0.05    # (max - min)/mean of r below this: a fixed point, no burst


def classify_bursts(amp: torch.Tensor, w0: torch.Tensor, dt: float, *,
                    n_res: float) -> np.ndarray:
    """ Per row, one of BURST_CLASSES:

      no_osc       relative amplitude below OSC_MIN: a fixed point, no burst to resolve.
      unconverged  w0 not extrapolable: the widths are not in their asymptotic regime yet.
      resolved     w0/dt >= n_res: the continuum burst spans at least n_res bins of dt.
      unresolved   otherwise, including w0 <= 0 (a burst collapsing with the grid).

    Validity only: resolved means this dt can see the burst, not that sigma is accurate there.
    """
    cls = np.full(len(amp), "unresolved", dtype=object)
    cls[(w0 / dt >= n_res).numpy()] = "resolved"
    cls[w0.isnan().numpy()] = "unconverged"
    cls[(amp < OSC_MIN).numpy()] = "no_osc"
    return cls
