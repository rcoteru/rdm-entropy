""" From a trajectory to a reported number: the window it is averaged over, the units it is
reported in, and whether the solver step resolves it -- dt, extrapolated to its dt -> 0 limit.
The age grid needs no check of this kind -- equations.age_window sizes it from a bound -- but
every reported number has to come with this one. """

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import torch

import rdme.helpers as hlp

# Cycle windows
# ~~~~~~~~~~~~~

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
    up = _up_crossings(x)
    if len(up) < 2:
        return 0, len(x), 0
    return int(up[0]), int(up[-1]), len(up) - 1


def _up_crossings(x: torch.Tensor) -> torch.Tensor:
    """ Indices of the upward crossings of the mean of `x`, the cycle boundaries. """
    y = x - x.mean()
    return ((y[:-1] <= 0) & (y[1:] > 0)).nonzero().flatten()


def cycle_amplitude_ratio(x: torch.Tensor) -> float:
    """ Peak-to-peak of the last whole cycle of `x` over that of the first, the cycles as
    cycle_window delimits them: 1 on a limit cycle, below 1 on a decaying oscillation (ringing
    onto a fixed point), above 1 on a growing one. With a single cycle the two are the same
    cycle and the ratio is 1; nan with none. """
    up = _up_crossings(x)
    if len(up) < 2:
        return float("nan")
    first, last = x[up[0]:up[1]], x[up[-2]:up[-1]]
    return float((last.max() - last.min()) / (first.max() - first.min()))


def phase_profile(x: torch.Tensor, r: torch.Tensor, n_bins: int) -> torch.Tensor:
    """ Mean of `x` against the phase of `r`, (n_bins,), over every whole cycle of r.

    Phase 0 is an upward crossing of r's mean, located between steps by linear interpolation,
    and each cycle is rescaled to [0, 1) by its own length -- so traces at different dt, whose
    periods differ slightly, land on one axis. Bin k holds phases [k, k+1)/n_bins; a bin no
    sample falls in is nan. For diagnosis: a per-step quantity's profile, not a reported
    number. """
    y  = (r - r.mean()).double()
    up = _up_crossings(r)
    if len(up) < 2:
        raise ValueError("phase_profile needs at least one whole cycle of r")
    tc = up.double() + (-y[up]) / (y[up + 1] - y[up])     # crossing times, in steps
    i  = torch.arange(int(up[0]) + 1, int(up[-1]) + 1)    # samples inside whole cycles
    k  = torch.searchsorted(tc, i.double(), right=True) - 1
    ph = (i.double() - tc[k]) / (tc[k + 1] - tc[k])
    b  = (ph * n_bins).long().clamp(max=n_bins - 1)
    s  = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b, x[i].double())
    c  = torch.zeros(n_bins, dtype=torch.float64).index_add_(0, b, torch.ones_like(ph))
    return s / c


def running_mean(x: torch.Tensor, n: int) -> torch.Tensor:
    """ Centred running mean of `x` over n samples along its first axis, the same length as
    x; at the ends the window is truncated and the mean taken over what it holds. """
    x = x.double()
    c = torch.cat([torch.zeros_like(x[:1]), x.cumsum(0)])
    i = torch.arange(len(x))
    lo, hi = (i - n // 2).clamp(min=0), (i - n // 2 + n).clamp(max=len(x))
    w = (hi - lo).double().view((-1,) + (1,) * (x.dim() - 1))
    return (c[hi] - c[lo]) / w


def noisy_cycles(r: torch.Tensor, dt: float, period_ms: float,
                 rel_tol: float = 0.05) -> tuple[torch.Tensor, float, int]:
    """ (phase reference, period in ms, n_cycles) of a noisy rate r whose period is about
    period_ms (the mean field's): r smoothed over an eighth of that period, and the cycles
    cycle_window finds in it. Unsmoothed, or smoothed over a few ms, a finite-N rate crosses
    its mean several times per cycle (on E1 a 4 ms window counts 30 cycles where there are
    28). The period is taken from the hint, not from r's Fourier peak: on a burst the second
    harmonic is the larger peak (EI-burst: 23.5 ms against a 47 ms cycle). """
    ref = running_mean(r, max(1, round(period_ms / 8 / dt)))
    a, b, n = cycle_window(ref, rel_tol)
    return ref, ((b - a) * dt / n if n else float("nan")), n


# Standard outputs
# ~~~~~~~~~~~~~~~~

def rate_and_epr(traj: dict, dt: float, t_abs: float, *, cycles: bool = False,
                 amp_tol: float = 0.05) -> dict:
    """ Rate and entropy production of one trajectory, in physical units.

    `traj` is an entropy_trajectory dict: m_tot is the fraction of neurons spiking per bin,
    sigma_tot the entropy production in nats per neuron per bin. `dt` and `t_abs` are in ms.
    The average runs over the whole trajectory, or with `cycles` over a whole number of
    periods of r (cycle_window): for the mean field only, since on a finite-N trace it counts
    noise crossings as cycles. With `cycles`, an oscillation counts as a cycle only if the
    peak-to-peak of its last and first recorded cycles agree within `amp_tol`
    (cycle_amplitude_ratio); otherwise it is still settling -- ringing onto a fixed point, or
    growing towards a cycle -- and is reported as such.

    Returns
      r, sigma           traces: Hz, nats per neuron per s
      r_mean, r_min, r_max   over the averaging window, Hz
      sigma_mean         nats per neuron per s; the EPR
      sigma_per_spike    sigma_mean / r_mean, nats per spike; nan at r_mean = 0
      dead_frac          r_mean * t_abs, dimensionless: the fraction of time a neuron spends
                         in the absolute window, against the dead-time ceiling 1/t_abs. The
                         mean, not the peak: a burst can briefly exceed 1/t_abs
      n_cycles           periods averaged over; None without `cycles`
      state              "settled" (no oscillation, n_cycles = 0), "cycle" or "settling";
                         None without `cycles`
      amp_ratio          cycle_amplitude_ratio of r; None unless oscillating """
    r     = traj["m_tot"] / dt * 1e3        # spikes per neuron per s
    sigma = traj["sigma_tot"] / dt * 1e3    # nats per neuron per s
    a, b, n = cycle_window(r) if cycles else (0, len(r), None)
    state, ratio = None, None
    if cycles:
        if n == 0:
            state = "settled"
        else:
            ratio = cycle_amplitude_ratio(r)
            state = "cycle" if abs(ratio - 1) <= amp_tol else "settling"
    r_mean, sigma_mean = r[a:b].mean().item(), sigma[a:b].mean().item()
    return dict(r=r, sigma=sigma, r_mean=r_mean, r_min=r[a:b].min().item(),
                r_max=r[a:b].max().item(), sigma_mean=sigma_mean,
                sigma_per_spike=sigma_mean / r_mean if r_mean > 0 else float("nan"),
                dead_frac=r_mean * t_abs * 1e-3, n_cycles=n, state=state, amp_ratio=ratio)


def isi_cv(fprobs: torch.Tensor, dt: float) -> float:
    """ CV of the interspike interval of a renewal neuron with the age-resolved hazard fprobs
    (Q,): the stationary ISI density is hazard x survival, Phi(n) prod_{k<n}(1 - Phi(k)), at
    ISI = (n+1)*dt (equations.n_abs_bins). The absorbing last bin contributes its exact
    geometric tail, so the mean ISI is 1/equations.stationary_rate. Meaningful only where the
    hazard is stationary -- an uncoupled population, or a settled fixed point. """
    phi = fprobs.double()
    Q = phi.shape[-1]
    surv = torch.cat([phi.new_ones(1), torch.cumprod(1 - phi, dim=-1)[:-1]])
    f, x = surv[:-1] * phi[:-1], torch.arange(1, Q, dtype=phi.dtype, device=phi.device) * dt
    s, h = surv[-1], phi[-1]                  # the tail: ISI = (Q + k)*dt, k ~ Geometric(h)
    m1 = (f * x).sum() + s * dt * (Q + (1 - h) / h)
    m2 = (f * x**2).sum() + s * dt**2 * (Q**2 + 2 * Q * (1 - h) / h + (1 - h) * (2 - h) / h**2)
    return float((m2 - m1**2).sqrt() / m1)


def isi_cv_spikes(s: torch.Tensor, dt: float) -> float:
    """ CV of the interspike intervals recorded in a spike raster s (T, N): every interval
    between consecutive spikes of one neuron, pooled over neurons. """
    neuron, t = s.bool().T.nonzero(as_tuple=True)          # sorted by neuron, then time
    isi = t.diff()[neuron[1:] == neuron[:-1]].double() * dt
    return float(isi.std() / isi.mean())


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


def observed_order(vals: Sequence[float] | torch.Tensor,
                   p_range: tuple[float, float] = (0.0, 3.0)) -> tuple[float, str]:
    """ (order the last triple shows, why it is not one): why is "" for an asymptotic
    sequence, "sign change" when the successive differences change sign anywhere (p is then
    nan), and "out of range" when any observed order falls outside p_range (p is still
    returned). Needs at least three values at halving dt. """
    vals = torch.as_tensor(vals, dtype=torch.float64)
    if len(vals) < 3:
        raise ValueError("observed_order needs at least three values")
    d = vals.diff()
    if bool(((d[:-1] * d[1:]) < 0).any()):
        return float("nan"), "sign change"
    o = orders(vals)
    ok = bool(((o >= p_range[0]) & (o <= p_range[1])).all())
    return float(o[-1]), ("" if ok else "out of range")


def richardson(vals: Sequence[float] | torch.Tensor, p: float | None = None,
               p_range: tuple[float, float] = (0.0, 3.0)) -> tuple[float, float]:
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
      an observed order outside p_range (default [0, 3]).  Same disease, caught one step
          earlier -- a first-order method cannot show p = 8, so a triple that says so is
          measuring something other than the discretization error.

    Both are observed_order's verdict. A given `p` extrapolates at that order instead of the
    observed one, but the refusal still applies.

    Returning nan rather than a cautious number is the point. The caller formats it as a
    refusal and the convergence verdicts fall through to "not resolved", which is the honest
    answer; a plausible-looking limit with a relative-error column beside it is not.

    A bare PAIR (len 2) has no order to read and no sign change to see, so it is extrapolated
    at first order unconditionally -- which is why the standing rule asks the pair's own order
    to be reported from somewhere else, usually a third, coarser step. """
    vals = torch.as_tensor(vals, dtype=torch.float64)
    if len(vals) < 2:
        raise ValueError("richardson needs at least a (dt, dt/2) pair")
    p_obs = 1.0
    if len(vals) > 2:
        p_obs, why = observed_order(vals, p_range)
        if why:
            warnings.warn(f"richardson refused: {why}; orders "
                          f"{[f'{v:.2f}' for v in orders(vals).tolist()]}", RuntimeWarning,
                          stacklevel=2)
            return float('nan'), float('nan')
    if p is None:
        p = p_obs if 0.2 < p_obs < 4.0 else 1.0
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
