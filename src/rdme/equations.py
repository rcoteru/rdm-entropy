""" The mean field's equations: one timestep, the kernels it is built from, and the observables
read off it, as pure functions.

The base layer of the package. Nothing here knows which per-bin kernel is in use -- the hazard
arrives as an argument (fprobs, or a Hazard callable) or not at all -- and nothing here imports
the rest of the package: stability, lyapunov and epr build on this module, never the other
way round.

The per-step updates act on the age axis, (..., Q), and treat everything before it as
batch. The equations below them take the population state as one tensor (..., M, Q): every
population shares one age grid Q, and every tensor carries arbitrary LEADING axes, so the same
call serves one system, (M, Q), a batch, (B, M, Q), or a grid. Per-population parameters are
(..., M) and are lifted as x[..., None] against the age axis. RDMNetwork holds state and calls
these; it writes no equation of its own.

The arithmetic is the batch class's (decided 2026-11-13): out-of-place updates, and the firing
rate as a (1, Q) @ (Q, 1) product, which reproduces the batch's einsum bit for bit on every
shape including 1-D (see compute_firing_rate).

hazard_clamp is a dtype-dependent numerical bound for the stationary survival (Stationary
states, below). The entropy production no longer uses it: it reads ln Phi and ln(1 - Phi)
off the kernels' exact branches, because a clamped log is what a saturated H_rev is made of. """

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, overload

import torch

if TYPE_CHECKING:          # annotation only: this module imports nothing from the package
    from rdme.hazards import Hazard

# Per-step updates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_total_input(
    r: torch.Tensor,  # (M,) or (B, M) population RATES, in 1/ms -- NOT the per-bin overlap
    w: torch.Tensor,  # (M, M) or (B, M, M) pop connection weights, PHYSICAL units
    E: torch.Tensor,  # (M,) or (B, M) pop external drives
) -> torch.Tensor:
    """Total input to each population: I^a = sum_b w[a, b] r^b + E^a, i.e. w @ r + E.
    Row = target, column = source, matching w^{ab} ("from population b onto a") in the theory.

    THE WEIGHT CONVENTION, stated once here because all three model classes share it and
    nothing else in the package should restate it.

    The first argument is the RATE r = m/dt in 1/ms, not the per-bin firing probability m.
    Weights are therefore in physical units and are NOT divided by dt: pass J, never J/dt.

    The point of building the input from the rate is that it makes w independent of the
    solver. m is O(dt) and so means something different at every step size, while r has a
    dt -> 0 limit; a coupling multiplied into r therefore keeps its meaning when dt is
    refined. Concretely, J = 4 at dt = 0.5 is the same model as J = 4 at dt = 0.1 or dt = 1,
    and only the discretization error moves. A coupling multiplied into m would have to be
    rescaled at every step size, which is the trap this convention exists to remove.

    Historical note, because old caches and old scripts still carry it: the pre-port
    implementation built the input from m and so stored w already divided by dt. A J quoted
    against that convention is J_physical/dt and will be wrong by that factor here. The batch
    loader converts such saves on read; a script that hard-codes a pre-divided J will not be
    caught by anything and has to be checked by hand."""
    return (w @ r.unsqueeze(-1)).squeeze(-1) + E


def update_synaptic_term(
    S: torch.Tensor,                # (..., Q) synaptic term
    alpha: float | torch.Tensor,    # synaptic decay factor
    I: float | torch.Tensor,        # total input to the population
    blocked: torch.Tensor,          # (..., Q) bool, the absolute window (n < n_abs)
    inplace: bool = False,
) -> torch.Tensor:
    """Synaptic-term update: S_new(n+1) = alpha*I + (1-alpha)*S(n), with S_new(0) = 0, and
    S_new(n+1) = 0 where bin n is blocked.

    The clamp. S resets to 0 at the spike and stays there through the absolute refractory
    window; it starts integrating the input only when the window ends, as the GIF model
    clamps its voltage at u_r for t_ref (Schwalger, Deger & Gerstner 2017, PLoS Comput Biol
    13(4): e1005507, Methods: "After each spike the voltage is reset to the potential u_r,
    where it is clamped for an absolute refractory period t_ref"; u_r = 0 here). On the grid
    that is S(n) = 0 for n <= n_abs: the window's bins and the first eligible one, which,
    like R's -K there, has had zero time since the window ended. The filter runs from n_abs
    on, so S(n) has integrated (n - n_abs) bins of input: for t_abs > 0 the same clock R
    decays on, (n+1)*dt - t_abs (with t_abs = 0, R keeps its one-bin lead, as before).

    t_abs = 0 (and t_abs = dt) blocks nothing, and this is the plain update bit for bit."""
    gate = blocked[..., :-1]
    if inplace:
        shifted = S[..., :-1].mul(1 - alpha)
        shifted.add_(alpha * I).masked_fill_(gate, 0.0)
        S[..., 1:] = shifted
        S[..., 0] = 0
        return S
    else:
        S_new = torch.zeros_like(S)
        S_new[..., 1:] = torch.where(gate, 0.0, alpha * I + (1 - alpha) * S[..., :-1])
        return S_new


def integration_age(blocked: torch.Tensor) -> torch.Tensor:
    """ (..., Q) long: the number of bins of input the synaptic term at age n has integrated,
    max(n - n_abs, 0) -- see update_synaptic_term. The clamp is a pure shift on this clock,
    in either direction of time: S at age n is the unclamped S at age integration_age(n). """
    n = torch.arange(blocked.shape[-1], device=blocked.device)
    return (n - blocked.sum(dim=-1, keepdim=True)).clamp_min(0)


def compute_firing_rate(P: torch.Tensor, fprobs: torch.Tensor) -> torch.Tensor:
    """Firing rate for one population: sum(P * fprobs) over the age (LAST) axis.

    One form for every shape: (..., 1, Q) @ (..., Q, 1). It is bit-identical to the batch's
    former einsum("...q,...q->...") on (B, Q) and on grids, and on a 1-D input it gives exactly
    what a batch row gives -- which neither einsum (1-D) nor torch.dot does. A batch of B = 1
    is the exception (its rows differ from B >= 2's at the last bit), a dispatch quirk of the
    backend, not of this formula."""
    return (P.unsqueeze(-2) @ fprobs.unsqueeze(-1))[..., 0, 0]


def update_age_distribution(
    P: torch.Tensor, fprobs: torch.Tensor, inplace: bool = False
) -> torch.Tensor:
    """Renewal update: births into bin 0, survival-decay into interior bins, remainder
    absorbed into the last (boundary) bin to keep P normalized."""
    if inplace:
        frate = compute_firing_rate(P, fprobs)
        interior = (
            fprobs[..., :-2].neg().add_(1)
        )  # 1 - fprobs[..., :-2], one fresh buffer
        interior.mul_(P[..., :-2])  # fuse in the multiply, no second buffer
        P[..., 0] = frate  # interior is already materialised, so
        P[..., 1:-1] = interior  # the overlapping read/write is safe
        P[..., -1] = 1 - frate - interior.sum(dim=-1)
        return P
    else:
        P_new = torch.zeros_like(P)
        P_new[..., 0] = (P * fprobs).sum(dim=-1)
        P_new[..., 1:-1] = P[..., :-2] * (1 - fprobs[..., :-2])
        P_new[..., -1] = 1 - P_new[..., :-1].sum(dim=-1)
        return P_new


def check_age_normalization(P: torch.Tensor, tol: float = 1e-6) -> None:
    """Check that the population distribution P is normalized over its age (LAST) axis,
    for one distribution (Q,) or a batch of them (..., Q)."""
    s = P.sum(dim=-1)
    if not torch.allclose(s, torch.ones_like(s), atol=tol):
        raise ValueError(
            f"Population distribution not normalized: max |sum - 1| = {(s - 1).abs().max().item()}"
        )


# Numerical guards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def hazard_clamp(dtype: torch.dtype) -> float:
    """Upper bound to clamp a hazard to before taking log1p(-phi).

    1 - 1e-12 is below float32's resolution -- it rounds to exactly 1, so a saturated hazard
    turns the survival into -inf and everything downstream into NaN. The bound therefore
    follows the dtype's own resolution, while float64 keeps its historical 1 - 1e-12."""
    return 1.0 - max(1e-12, torch.finfo(dtype).eps)


# Kernels and bin counts
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The exponential kernels the model is built from, and the bin counts they imply. Pure
# functions of a time constant and a step: they tie the model's parameters, quoted in ms, to an
# age grid quoted in bins -- which is why every one of them takes dt and why changing dt changes
# nothing physical as long as it goes through these.

def synaptic_kernel(Q, tau_m, delta):
    """ Synaptic kernel for the model.
    Normalized exponential kernel with time constant tau_m,
    evaluated at integer multiples of delta."""
    t = torch.arange(0, Q)
    return torch.exp(-t*delta/tau_m)/tau_m*delta


@overload
def n_abs_bins(t_abs: torch.Tensor, dt: float) -> torch.Tensor: ...
@overload
def n_abs_bins(t_abs: float, dt: float) -> int: ...
def n_abs_bins(t_abs: float | torch.Tensor, dt: float) -> int | torch.Tensor:
    """ Number of leading age bins an absolute refractory period of t_abs blocks. A tensor
    t_abs gives a long tensor of the same shape, each distinct value checked as below.
    "Blocks" is exact: the hazard on those bins is masked to Phi = 0, never floored (see
    refractory_kernel).

    round(t_abs/dt) - 1, NOT round(t_abs/dt), and the off-by-one is the whole content of this
    function. A neuron that fires enters bin 0 and is first eligible again one step LATER, so
    firing from bin n produces an interspike interval of (n+1)*dt. Blocking bins 0..n-1
    therefore buys a dead time of (n+1)*dt, and asking for t_abs means n = t_abs/dt - 1.

    Getting this wrong is not a rounding detail: blocking round(t_abs/dt) bins gives a realised
    dead time of t_abs + dt, which MOVES with the solver step -- measured 1.5, 1.2, 1.1 ms for a
    nominal t_abs = 1 ms at dt = 0.5, 0.2, 0.1. A dead time that depends on dt is not an
    absolute refractory period.

    Two guards, both for that same failure. dt > t_abs is refused: the expression would go
    negative, and there is no way to represent a dead time shorter than one bin -- the realised
    one would be dt, i.e. the solver step, which is exactly the dt-dependent window this is
    here to prevent. A t_abs that is not an integer number of bins is refused for the same
    reason: it rounds, and a window that rounds moves with dt. Refine dt onto a divisor of
    t_abs instead (with t_abs = 1 ms, dt = 1, 1/2, 1/4, ... all divide it, which is what the
    (dt, dt/2) Richardson pairs are built from).

    t_abs = 0 is the one exemption and returns 0: no window at all, the plain exponential
    kernel, and an unbounded realised rate under an exponential hazard. It has to be asked for
    explicitly. """
    if torch.is_tensor(t_abs):
        u, inv = t_abs.unique(return_inverse=True)
        return torch.tensor([n_abs_bins(v, dt) for v in u.tolist()], device=t_abs.device)[inv]
    if t_abs == 0.0:
        return 0
    if not t_abs > 0.0:
        raise ValueError(f"t_abs must be non-negative, got {t_abs}")
    if dt > t_abs * (1.0 + 1e-9):
        raise ValueError(
            f"dt={dt:g} ms exceeds t_abs={t_abs:g} ms: the dead time cannot be shorter than "
            "one solver step, and rounding it to one would make the window move with dt. "
            "Refine dt to a divisor of t_abs, or set t_abs=0 for no window.")
    ratio = t_abs / dt
    if abs(ratio - round(ratio)) > 1e-9 * max(1.0, abs(ratio)):
        raise ValueError(
            f"t_abs={t_abs:g} ms is not an integer number of dt={dt:g} ms bins "
            f"(t_abs/dt = {ratio:.12g}): the window would round, and a dead time that rounds "
            "moves with the solver step. Pick dt as a divisor of t_abs.")
    return max(round(ratio) - 1, 0)


def refractory_kernel(Q: int, K, tau_r, dt: float = 1.0,
                      t_abs: float | torch.Tensor = 0.0,
                      device: str | torch.device = 'cpu') -> torch.Tensor:
    """ The refractory kernel: Gerstner's SRM_0 kernel (Neuronal Dynamics, Eq. 9.24), a hard
    absolute window on the first n_abs_bins(t_abs, dt) age bins followed by an exponential
    relative term that starts decaying when the window ENDS, not at the spike:

        R(u) = -K exp(-(u - t_abs)/tau_r),   u >= t_abs
             = -infinity,                    u <  t_abs

    K is therefore the amplitude at the first instant the neuron can fire again.

    The -infinity is NOT stored. What this returns is the relative part only, and on the
    blocked bins it holds the clamped value -K, which nothing reads: the window is applied by
    the caller as a boolean mask on the hazard (Phi = 0, ln(1 - Phi) = 0, dPhi/dh = 0, ln Phi
    = -inf exactly), never through the field. A finite field value would leak e^{-margin}
    and an infinite one would turn 0 * (-inf) into nan; the mask does neither. Every weight
    a blocked bin carries is exactly 0 -- a path has non-zero probability iff every ISI is
    >= t_abs, and reversal keeps the ISIs -- so the -inf in ln Phi only ever meets a zero.

    Discretization: bin n is evaluated at u = (n+1)*dt, the interspike interval a spike fired
    from that bin produces (see n_abs_bins). The first eligible bin, n_abs = t_abs/dt - 1,
    then carries exactly -K at every dt, and R at a fixed u >= t_abs does not depend on dt.
    The spin model's R state follows the same convention.

    K = 0 or tau_r = 0 means no relative term (R = 0 past the window): pure dead time.

    t_abs = 0 is the plain exponential kernel, R = -K exp(-(n+1)*dt/tau_r) on every bin, and
    an explicit choice to run with no rate ceiling at all; the model classes default to 1 ms
    instead.

    Why the window exists at all, in two independent senses:

    1. MODELLING. The exponential term alone is a *relative* refractory mechanism: it depresses
       the field by -K e^{-u/tau_r}, so a drive strong enough makes S + R - theta positive and
       the neuron fires straight through its own refractoriness. Measured on a Wilson-Cowan
       sweep with K up to 30 and tau_r up to 22 ms, the strongly-driven corner fires at an
       effective dead time of ~0.3 ms against the 2.5-22 ms the kernel nominally imposes -- a
       median population rate of 1264 Hz. Raising K does not fix it robustly, because "large
       enough not to be overwhelmed" is a statement about the drive range. A hard window is
       drive-independent, which is the point.

    2. WELL-POSEDNESS UNDER AN EXPONENTIAL HAZARD, which is why t_abs is a parameter of the
       model in its own right (1 ms by default) rather than a shadow of the solver step. The
       route-B rate nu0 e^{beta*h} is unbounded, so nothing in the kernel caps how fast a
       cohort fires; the dead time does, at 1/t_abs, and it is the ONLY thing that does. Taken
       to zero, the realised rate grows without bound with the drive -- measured in the spin
       model with weak refractoriness and strong drive, 25%, 51% and 56% of all interspike
       intervals fall below 1 ms at dt = 0.5, 0.2 and 0.1, with the population rate inflating
       from 0.67 to 0.90 /ms as the step is refined. With t_abs fixed in ms the same run gives
       the same dead time at every dt, which is what makes the dt -> 0 limit a limit of one
       model rather than of a family of them. n_abs_bins refuses a dt that does not divide
       t_abs, for that reason.

    This is the standard SRM treatment, where eta(u) is taken to -infinity below the absolute
    refractory period, taken literally: the mask is exact, so no drive however strong reaches
    a blocked bin.

    K, tau_r and t_abs may be scalars, or tensors shaped (..., 1) that broadcast against the
    age axis: RDMNetwork passes (*b, M, 1) and gets every population's kernel, (*b, M, Q), in
    one call; its window mask is RDMNetwork.blocked.
    """
    K, tau_r = torch.as_tensor(K, device=device), torch.as_tensor(tau_r, device=device)
    # in K's dtype: an integer n would put n * dt in torch's global default instead
    n = torch.arange(Q, device=device, dtype=K.dtype)
    # time since the end of the window, at the ISI (n+1)*dt each bin stands for; clamped so
    # the blocked bins (never read: see above) stay finite
    x = ((n + 1) * dt - torch.as_tensor(t_abs, device=device, dtype=K.dtype)).clamp_min(0.0)
    on = (K != 0) & (tau_r > 0)
    return torch.where(on, -K * torch.exp(-x / torch.where(on, tau_r, 1.0)), 0.0).to(K.dtype)


# The initial conditions the mean field and the spins share. Both have S = 0 (no input history)
# and R read off the refractory kernel at each neuron's age, so the mean field's starting state
# is exactly the expectation of the spins' one, and fdist(Q) matches p at t = 0.
#   "silent": every neuron older than any age grid, none firing: all mass in the last bin.
#   "random": ages uniform on bins 0..K-1, K = random_age_bins(age_max, dt); the ones at age
#             0 are the ones firing now, so the starting activity is 1/K.
INITIAL_CONDITIONS = ("silent", "random")


def random_age_bins(age_max: float, dt: float) -> int:
    """ K, the number of age bins the "random" initial condition spreads ages over: age_max is
    in ms, so the window is the same at every dt. """
    if not age_max > 0 or round(age_max / dt) < 1:
        raise ValueError(f"age_max={age_max} ms must cover at least one step of dt={dt} ms")
    return round(age_max / dt)


def tau2alpha(tau: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Convert time constant tau to integration coefficient alpha. """
    return 1.0 - torch.exp(-dt / tau)


# The age window
# ~~~~~~~~~~~~~~
#
# The one rule that sizes the age grid, everywhere. The last bin is absorbing: every cohort
# older than U = (Q-1)*dt fires at the field of age U, h(U), instead of its own h(u). For
# u >= U >= t_abs that field error is bounded, term by term:
#
#   refractory   |R(u) - R(U)| <= K_ref e^{-(U-t_abs)/tau_ref}         (R rises monotonically to 0)
#   synaptic     |S(u) - S(U)| <= A e^{-(U-t_abs)/tau_int}              (drive_bound below)
#
# so |h(u) - h(U)| <= Delta(U) = K_ref e^{-(U-t_abs)/tau_ref} + A e^{-(U-t_abs)/tau_int}. The
# synaptic term is clamped at 0 through the window (update_synaptic_term), so it has
# integrated only the time since the window ended -- (U - t_abs)/dt + 1 bins on the grid, at
# least the continuum's U - t_abs, which is what keeps the bound free of dt. Every hazard obeys
# Phi(h + d)/Phi(h) in [e^{-beta|d|}, e^{beta|d|}] -- exactly for "poisson", by concavity of
# 1 - e^{-x} for "poisson_exact", and by sigma(x + d)/sigma(x) <= e^{|d|} for "bernoulli" -- and
# the boundary bin's share of the firing is at most 1. Hence, at every step and for every
# population,
#
#   |firing misassigned by the truncation| / (total firing) <= expm1(beta * Delta(U)),
#
# and age_window returns the smallest U that holds this below `tol`. It depends on the model's
# parameters and on tol only -- not on dt, not on a run, not on the attractor -- so Q follows
# as ceil(U/dt) + 1 with nothing to fit. It bounds the error committed per step; what a
# perturbation of that size does to a long-time statistic is the dynamics' business, and on
# every S1 cell measured (fixed points, cycles, bursting at CV ~ 10) the realised error in the
# mean rate sat at or below the bound.


def drive_bound(w: torch.Tensor, E: torch.Tensor, t_abs: torch.Tensor,
                tau_int: torch.Tensor) -> torch.Tensor:
    """ A^a, (..., M): |S(u) - S(U)| <= A^a e^{-(U - t_abs^a)/tau_int^a} for every pair of
    cohorts u >= U >= t_abs^a.

    S is clamped at 0 through the target's own window and integrates only from its end (see
    update_synaptic_term), so a cohort of age U has integrated V = U - t_abs^a of input,
    V/dt + 1 bins on the grid. Two cohorts u >= U share those taps, and the gap is the
    filter's tail past them, sum_{k > V/dt} alpha (1-alpha)^{k-1} I(t-k): V plays the part U
    played before the clamp, and t_abs^a = 0 is exactly that case. The drive splits as
    I = E + sum_b w^{ab} r^b. E contributes |E| e^{-V/tau}. A rate has no instantaneous bound
    (a synchronous burst puts the whole population into one step), but the dead time caps it
    on average: no neuron of b fires twice within t_abs^b (the SOURCE's window), so any window
    of t_abs^b holds at most one spike per neuron. Taking the tail window by window at its
    largest weight,

        sum_{k > V/dt} alpha (1-alpha)^{k-1} r_{t-k} <= e^{-V/tau} (alpha/dt) / (1 - e^{-t_abs^b/tau})
                                                     <= e^{-V/tau} / (tau (1 - e^{-t_abs^b/tau})),

    using alpha/dt <= 1/tau. No dt survives, so the bound holds at every step size. A source
    with no dead time has no ceiling, and a population it drives, either sign, gets A = inf. """
    t_abs = torch.as_tensor(t_abs, dtype=w.dtype, device=w.device).expand_as(E)
    tau = tau_int[..., :, None]                                   # target's synapse, (..., M, 1)
    per_spike = 1.0 / (tau * -torch.expm1(-t_abs[..., None, :] / tau))   # (..., M, M_source)
    return E.abs() + torch.where(w != 0, w.abs() * per_spike, 0.0).sum(dim=-1)


def field_tail(U, K_ref: torch.Tensor, tau_ref: torch.Tensor, t_abs: torch.Tensor,
               tau_int: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """ Delta(U) = K_ref e^{-(U-t_abs)/tau_ref} + A e^{-(U-t_abs)/tau_int}, (..., M): the
    largest field error the absorbing bin can commit on a cohort older than U (U >= t_abs, in
    ms). Both terms run on the time since the window ended, R because it decays from there
    and S because it is clamped until then. The refractory term is 0 when K_ref = 0 or
    tau_ref = 0 (no relative term). """
    on = (K_ref != 0) & (tau_ref > 0)
    x = torch.as_tensor(U - t_abs).clamp_min(0.0)
    ref = torch.where(on, K_ref * torch.exp(-x / torch.where(on, tau_ref, 1.0)), 0.0)
    return ref + drive * torch.exp(-x / tau_int)


def age_window(beta: torch.Tensor, K_ref: torch.Tensor, tau_ref: torch.Tensor,
               t_abs: torch.Tensor, tau_int: torch.Tensor, drive: torch.Tensor,
               tol: float) -> torch.Tensor:
    """ The smallest U (ms), per population, with expm1(beta * Delta(U)) <= tol. See the
    section comment for the bound. Delta is decreasing in U, so the crossing is found by
    bisection from an upper bracket where each of its two terms is at most half the target. """
    if not (0.0 < tol < 1.0):
        raise ValueError(f"tol must be in (0, 1), got {tol}")
    if not bool(torch.isfinite(drive).all()):
        raise ValueError("A population is driven by one with no dead time (t_abs = 0): that "
                         "rate has no ceiling, so no age window is bounded. Pass Q explicitly; "
                         "truncation_error() is then inf, honestly.")
    target = math.log1p(tol) / beta
    on = (K_ref != 0) & (tau_ref > 0)
    hi_ref = torch.where(on, t_abs + tau_ref * torch.log(2 * K_ref / target), 0.0)
    hi = torch.maximum(hi_ref, t_abs + tau_int * torch.log(2 * drive / target)).clamp_min(0.0)
    lo = torch.zeros_like(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = field_tail(mid, K_ref, tau_ref, t_abs, tau_int, drive) <= target
        hi, lo = torch.where(ok, mid, hi), torch.where(ok, lo, mid)
    return hi


# Observables
# ~~~~~~~~~~~

def rates(m: torch.Tensor, dt: float) -> torch.Tensor:
    """ r = m/dt, in 1/ms, from the per-bin overlaps m (..., M). """
    return m / dt


def activity(m: torch.Tensor, N_ratios: torch.Tensor, dt: float) -> torch.Tensor:
    """ N-weighted mean network rate, (...), in 1/ms. """
    return (N_ratios * rates(m, dt)).sum(dim=-1)


def age_density(p: torch.Tensor, dt: float) -> torch.Tensor:
    """ q = p/dt, the age density of the transport equations, (..., M, Q). """
    return p / dt


def total_input(m: torch.Tensor, w: torch.Tensor, E: torch.Tensor, dt: float) -> torch.Tensor:
    """ I^a = sum_b w^{ab} r^b + E^a from the overlaps m (..., M); see compute_total_input
    for the weight convention. """
    return compute_total_input(rates(m, dt), w, E)


def field(S: torch.Tensor, R: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """ h^a(u) = S^a(u) + R^a(u) - theta^a, (..., M, Q). """
    return S + R - theta[..., None]


def per_population(fn: Callable[..., torch.Tensor], h: torch.Tensor,
                   beta: torch.Tensor, nu0: float, dt: float) -> torch.Tensor:
    """ A hazard function -- Hazard.phi (per-bin firing probability), .dphi (its h-derivative)
    or .rate (the rate it represents, 1/ms) -- applied to every population's field (..., M, Q)
    with that population's gain. """
    return fn(h, beta[..., None], nu0, dt)


def max_step_prob(fprobs: torch.Tensor) -> torch.Tensor:
    """ max over bins and populations of Phi, (...). """
    return fprobs.amax(dim=(-2, -1))


def truncation_error(p: torch.Tensor, fprobs: torch.Tensor, beta: torch.Tensor,
                     delta: torch.Tensor) -> torch.Tensor:
    """ (p[-1] Phi[-1] / sum p Phi) * expm1(beta * Delta), max over populations, (...): the
    age-window bound evaluated at the current state, with the boundary bin's actual share of
    the firing in place of its worst case 1. Never above the tol the window was sized for. """
    share = p[..., -1] * fprobs[..., -1] / (p * fprobs).sum(dim=-1).clamp_min(1e-300)
    return (share * torch.expm1(beta * delta)).amax(dim=-1)


def stationary_rate(fprobs: torch.Tensor, dt: float) -> torch.Tensor:
    """ r* = [int_0^inf Survival(u) du]^{-1} at the given hazards (..., M, Q), returned as
    (..., M); the absorbing last bin contributes its exact geometric tail. See
    RDMNetwork.stationary_rate. """
    # survival to the START of each bin: surv[0] = 1, surv[n] = prod_{k<n}(1 - Phi_k)
    keep = torch.cumprod(1 - fprobs, dim=-1)
    surv = torch.cat([torch.ones_like(keep[..., :1]), keep[..., :-1]], dim=-1)
    tail = surv[..., -1] / fprobs[..., -1].clamp_min(1e-300)     # absorbing bin, geometric
    return 1.0 / ((surv[..., :-1].sum(dim=-1) + tail) * dt)


# Dynamics
# ~~~~~~~~

def advance(p: torch.Tensor, S: torch.Tensor, I: torch.Tensor, fprobs: torch.Tensor,
            alpha: torch.Tensor, blocked: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ One step along the characteristics, given the input I (..., M) and hazards fprobs
    (..., M, Q) of the CURRENT state, and the absolute window blocked (..., M, Q). Returns
    (m, p, S) of the next one; nothing is modified in place. """
    m = compute_firing_rate(p, fprobs)
    p_new = update_age_distribution(p, fprobs)
    S_new = update_synaptic_term(S, alpha[..., None], I[..., None], blocked)
    return m, p_new, S_new


# Stationary states
# ~~~~~~~~~~~~~~~~~
#
# At a fixed point the input is constant, so the whole age-resolved state follows from it: the
# synaptic term unrolls to a geometric sum, the hazard is then a pure function of age, and the
# stationary age distribution is its normalized survival.

def compute_stationary_forward_field(
        I: torch.Tensor,               # (...) constant total input to the population
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        blocked: torch.Tensor,         # (..., Q) bool, the absolute window; Q from its length
        ) -> torch.Tensor:             # (..., Q) S_n at constant input
    """ The synaptic term at a fixed point, in closed form.

    With I frozen, S_n = alpha*I + (1-alpha)*S_{n-1} from S = 0 unrolls to the geometric
    sum S_n = I*[1 - (1-alpha)^e_n], e_n = max(n - n_abs, 0) the bins of input integrated
    since the window ended (equations.integration_age: S is clamped at 0 through the
    absolute window, see update_synaptic_term). The same saturating profile an isolated
    neuron has under a constant external drive, with the drive replaced by the
    self-consistent input; with t_abs = 0, e_n = n.

    This is the exact fixed point of update_synaptic_term, boundary bin included -- that
    update shifts rather than lumps, so its last bin is just the recursion at n = Q-1. """
    I = torch.as_tensor(I)
    decay = torch.as_tensor(1.0 - alpha, device=I.device, dtype=I.dtype)
    e = integration_age(blocked).to(I.dtype)
    return I.unsqueeze(-1) * (1.0 - decay.unsqueeze(-1) ** e)


def compute_stationary_survival(fprobs: torch.Tensor,   # (..., Q) age-resolved hazard
                     log: bool = False,
                     ) -> torch.Tensor:      # (..., Q) survival from age 0
    """ S_0(n) = prod_{j<n} (1 - Phi_j), with the empty product S_0(0) = 1.

    Accumulated in log space for the same reason the EPR path is: at large beta the hazard
    saturates and a long product of (1-Phi) underflows to zero, which would send the
    normalization of the stationary distribution to a ratio of zeros. The hazard is clamped
    below one first, with a dtype-dependent bound (see hazard_clamp). """
    log1m = torch.log1p(-fprobs.clamp(max=hazard_clamp(fprobs.dtype)))
    cum = torch.cumsum(log1m[..., :-1], dim=-1)
    log_surv = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)
    return log_surv if log else torch.exp(log_surv)


def compute_stationary_age_distribution(
        fprobs: torch.Tensor,                      # (..., Q) age-resolved hazard
        ) -> tuple[torch.Tensor, torch.Tensor]:    # (..., Q) p*, (...) m*
    """ Stationary age distribution and firing rate at a frozen, age-resolved hazard.

    Iterating the aging branch of update_age_distribution gives p*(n) = p*(0) * S_0(n) on
    the interior bins, while the lumped bin balances its own outflow against the inflow
    from n = Q-2, giving p*(Q-1) = p*(0) * S_0(Q-1) / Phi_{Q-1}: the exact resummation of
    the geometric tail the truncation cuts off, not an approximation of it. Normalization
    then fixes p*(0), which is the firing rate m* itself, since births equal spikes.

    The reset branch is not an extra condition -- sum_n p*(n) Phi_n = m* holds identically
    by telescoping -- so this single pass is the whole stationary solution. In the
    untruncated limit the boundary term drops and m* becomes the renewal firing rate
    1/sum_n S_0(n) of an isolated neuron. """
    surv = compute_stationary_survival(fprobs)
    tail = surv[..., -1] / fprobs[..., -1]                       # lumped bin, geometric tail
    m = 1.0 / (surv[..., :-1].sum(dim=-1) + tail)
    p = torch.cat([surv[..., :-1], tail.unsqueeze(-1)], dim=-1) * m.unsqueeze(-1)
    return p, m


def renewal_rate(x: float, nu0: float, t_abs: float) -> float:
    """ Stationary rate of a renewal process with a hard dead time t_abs followed by the
    constant route-B hazard lambda = nu0 e^x: the mean interval is t_abs + 1/lambda, so
    r = lambda / (1 + lambda t_abs) = (1/t_abs) sigma(x + ln(nu0 t_abs)). The continuum
    answer for an uncoupled population with K_ref = 0 and an instantaneous synapse. """
    lam = nu0 * math.exp(x)
    return lam / (1.0 + lam * t_abs)


# The overlap map
# ~~~~~~~~~~~~~~~
#
# Composing the two closed forms above gives the overlap map F, whose roots m = F(m) are the
# fixed points -- one O(M*Q) pass per evaluation, no time integration.

def fp_overlap_map(m: torch.Tensor,        # (M,) trial overlaps
                   w: torch.Tensor,        # (M, M)
                   E: torch.Tensor,        # (M,)
                   alpha: torch.Tensor,    # (M,)
                   beta: torch.Tensor,     # (M,)
                   theta: torch.Tensor,    # (M,)
                   R: torch.Tensor,        # (M, Q)
                   blocked: torch.Tensor,  # (M, Q) bool, the absolute window
                   hazard: Hazard,
                   nu0: float,
                   dt: float,
                   ) -> torch.Tensor:      # (M,) F(m)
    """ One evaluation of the overlap map F: trial overlaps -> stationary overlaps.

    The hazard-parametrised counterpart of mean_field.fp_overlap_map. Only the hazard step
    differs: compute_stationary_forward_field and compute_stationary_age_distribution are
    reused from there unchanged, because both already take the hazard as data -- the first
    never sees it, and the second takes fprobs as an argument.

    Note this works in overlaps m, not rates, so that the residual lives on the same scale as
    the generic Newton and bracketing routines in rdme.stability (which clamp to
    [0, 1]) applies unchanged. Rates are m/dt throughout the class API. """
    return _stationary(m, w, E, alpha, beta, theta, R, blocked, hazard, nu0, dt)[2]


def fp_state(m: torch.Tensor,
             w: torch.Tensor,
             E: torch.Tensor,
             alpha: torch.Tensor,
             beta: torch.Tensor,
             theta: torch.Tensor,
             R: torch.Tensor,
             blocked: torch.Tensor,
             hazard: Hazard,
             nu0: float,
             dt: float,
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """ The full stationary state (p, S), each (..., M, Q), implied by a set of overlaps. """
    return _stationary(m, w, E, alpha, beta, theta, R, blocked, hazard, nu0, dt)[:2]


def _stationary(m, w, E, alpha, beta, theta, R, blocked, hazard: Hazard, nu0: float,
                dt: float):
    """ (p, S, F(m)) at the overlaps m: the state and the overlaps it fires at. Phi is exactly
    0 on the blocked bins, where the survival then carries ln(1 - 0) = 0. """
    I = compute_total_input(m / dt, w, E)       # w is unscaled here, so feed it the rate
    S = compute_stationary_forward_field(I, alpha, blocked)
    phi = hazard.phi(S + R - theta[..., None], beta[..., None], nu0, dt).masked_fill(blocked, 0.0)
    p, m_new = compute_stationary_age_distribution(phi)
    return p, S, m_new


def fp_residual(m: torch.Tensor, w: torch.Tensor, E: torch.Tensor, alpha: torch.Tensor,
                beta: torch.Tensor, theta: torch.Tensor, R: torch.Tensor,
                blocked: torch.Tensor, hazard: Hazard, nu0: float, dt: float) -> torch.Tensor:
    """ G(m) = m - F(m), in overlaps; F is fp_overlap_map. """
    return m - fp_overlap_map(m, w, E, alpha, beta, theta, R, blocked, hazard, nu0, dt)
