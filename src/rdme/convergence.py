from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch

""" Is the solver step fine enough: dt, extrapolated to its dt -> 0 limit. The age grid needs
no check of this kind -- equations.age_window sizes it from a bound -- but every reported
number has to come with this one. """


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
