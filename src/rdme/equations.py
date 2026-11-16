from __future__ import annotations

import math
from typing import Callable

import torch

""" The mean field's equations: one timestep, the kernels it is built from, and the observables
read off it, as pure functions.

The base layer of the package. Nothing here knows which per-bin kernel is in use -- the hazard
arrives as an argument (fprobs, or a Hazard callable) or not at all -- and nothing here imports
the rest of the package: fixed_points, lyapunov and epr build on this module, never the other
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

hazard_clamp lives here rather than with the entropy production that motivated it, because the
stationary survival in rdme.fixed_points needs it too, and it is really a dtype-dependent
numerical bound. """


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
    S: torch.Tensor,                # (Q,) or (B, Q) synaptic term
    alpha: float | torch.Tensor,    # synaptic decay factor
    I: float | torch.Tensor,        # total input to the population
    inplace: bool = False,
) -> torch.Tensor:
    """Synaptic-term update: S_new(n+1) = alpha*I + (1-alpha)*S(n), with S_new(0) = 0."""
    if inplace:
        shifted = S[..., :-1].mul(1 - alpha)
        shifted.add_(alpha * I)
        S[..., 1:] = shifted
        S[..., 0] = 0
        return S
    else:
        S_new = torch.zeros_like(S)
        S_new[..., 1:] = alpha * I + (1 - alpha) * S[..., :-1]
        return S_new


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
    turns the survival into -inf and every EPR downstream into NaN. The bound therefore
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


def n_abs_bins(t_abs: float | torch.Tensor, dt: float) -> int | torch.Tensor:
    """ Number of leading age bins an absolute refractory period of t_abs blocks. A tensor
    t_abs gives a long tensor of the same shape, each distinct value checked as below.

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
    return max(int(round(ratio)) - 1, 0)


def abs_refractory_floor(w: torch.Tensor,        # (M, M) weights, row = target
                         E: torch.Tensor,        # (M,) external drive
                         theta: torch.Tensor,    # (M,) threshold
                         beta: torch.Tensor,     # (M,) gain
                         r_max,                  # (1/ms) float or (M,): per SOURCE ceiling
                         margin: float = 40.0,
                         ) -> torch.Tensor:      # (M,) R_abs per population
    """ How deep the absolute-refractory floor has to be, per population.

    The window blocks firing by pushing the field to S - R_abs, and R_abs has to be sized
    against the DRIVE, not against K_ref: what must be suppressed is Phi(beta*(S - R_abs -
    theta)) at the largest S the network reaches. Taking

        R_abs = S_max - theta + margin/beta,     margin ~ 40

    puts the hazard at sigma(-40) ~ 4e-18, below any rate that can matter and still comfortably
    inside float64. R_abs is finite rather than -inf on purpose: -inf gives 0 * -inf = nan in
    the log-probability branches of the entropy production and in dPhi/dh.

    S_max is bounded self-consistently. The window itself caps each population's rate at
    r_max^b = 1/t_abs^b, so S_max^a <= sum_b max(w_ab, 0) * r_max^b + max(E_a, 0) -- the
    argument closes on itself, which is why this works for the unbounded route-B kernels as
    well as for "bernoulli". A source with no window (r_max^b = inf) leaves S_max^a infinite
    wherever it excites a; a source it does not excite contributes nothing, not inf * 0.

    That the rule covers "poisson" and "poisson_exact", whose rate is exponential rather than
    linear in the field, is worth checking rather than assuming: an exponential outruns a
    softplus, so a rule sized on the latter need not hold for the former. It does here because
    the margin is a bound on the FIELD and not on the rate -- the floor puts beta*h at
    -margin, where the growth of the rate function is irrelevant and every kernel sits on the
    same Arrhenius tail. At margin = 40 the residual per-bin probability is of order
    e^{-40} * nu0*dt, i.e. below 4e-18 for every dt <= 1 ms at nu0 = 1/ms, and it falls
    linearly with dt as the step refines. The reference log-odds of "poisson" only ever shift
    the field DOWN (ell < 0 for c < 1/2), so that kernel is the mildest of the three here
    rather than the worst.

    Requires beta > 0: at zero gain the hazard does not depend on the field at all, so no
    finite floor suppresses firing and margin/beta is +inf -- which then reaches the hazard
    as beta*h = 0 * -inf = nan. Raised here rather than returned as inf, because a nan born
    in the first step only shows up much later, as a nan truncation_error. """
    if not bool((beta > 0).all()):
        raise ValueError("abs_refractory_floor needs beta > 0 everywhere; the absolute "
                         "refractory window is undefined at zero gain (got min beta = "
                         f"{float(beta.min()):g}).")
    w_pos = w.clamp_min(0.0)
    r_max = torch.as_tensor(r_max, dtype=w.dtype, device=w.device)
    r_max = r_max.unsqueeze(-2) if r_max.ndim else r_max       # (..., 1, M): along the sources
    S_max = torch.where(w_pos > 0, w_pos * r_max, 0.0).sum(dim=-1) + E.clamp_min(0.0)
    return (S_max - theta + margin / beta).clamp_min(margin / beta)


def refractory_kernel(Q: int, K, tau_r, dt: float = 1.0,
                      t_abs=0.0, R_abs=None, device='cpu') -> torch.Tensor:
    """ The refractory kernel: an exponential relative term, optionally floored by a hard
    absolute window on the first n_abs_bins(t_abs, dt) age bins.

        R(u) = -K exp(-u/tau_r),          u >= t_abs
             = -R_abs,                    u <  t_abs

    t_abs = 0 is the plain exponential kernel, the pre-window behaviour, and an explicit
    choice to run with no rate ceiling at all; the model classes default to 1 ms instead.

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

    This is also the standard SRM treatment, where eta(u) is taken to -infinity below the
    absolute refractory period. Since the kernel enters the model only through S + R - theta,
    and nothing assumes its functional form, the window propagates to the dynamics, the fixed
    points, the Jacobian and the entropy production with no other edit.

    R_abs must be sized against the drive; abs_refractory_floor computes it. Passing a value
    too small is not an error but silently makes the window soft again, which is the failure
    this is here to prevent.

    K, tau_r, t_abs and R_abs may be scalars, or tensors shaped (..., 1) that broadcast against
    the age axis: RDMNetwork passes (*b, M, 1) and gets every population's kernel, (*b, M, Q),
    in one call, each with its own window.
    """
    n = torch.arange(Q, device=device)
    K, tau_r = torch.as_tensor(K, device=device), torch.as_tensor(tau_r, device=device)
    R = -K * torch.exp(-n * dt / tau_r)
    n_abs = torch.as_tensor(n_abs_bins(t_abs, dt), device=device)
    if not bool((n_abs > 0).any()):
        return R
    if R_abs is None:
        raise ValueError(f"t_abs={t_abs} blocks {n_abs} bins but R_abs is None. Size it with "
                         "abs_refractory_floor(w, E, theta, beta, r_max=1/t_abs).")
    R_abs = torch.as_tensor(R_abs, device=device)
    return torch.where(n < n_abs, -R_abs.to(R.dtype), R)


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
#   refractory   |R(u) - R(U)| <= K_ref e^{-U/tau_ref}                 (R rises monotonically to 0)
#   synaptic     |S(u) - S(U)| <= A e^{-U/tau_int}                      (drive_bound below)
#
# so |h(u) - h(U)| <= Delta(U) = K_ref e^{-U/tau_ref} + A e^{-U/tau_int}. Every hazard obeys
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
    """ A^a, (..., M): |S(u) - S(U)| <= A^a e^{-U/tau_int^a} for every pair of cohorts u >= U.

    The two synaptic terms share their first U/dt filter taps, so the gap is the filter's tail,
    sum_{k >= Q} alpha (1-alpha)^{k-1} I(t-k). The drive splits as I = E + sum_b w^{ab} r^b. E
    contributes |E| e^{-U/tau}. A rate has no instantaneous bound (a synchronous burst puts the
    whole population into one step), but the dead time caps it on average: no neuron of b fires
    twice within t_abs^b, so any window of t_abs^b holds at most one spike per neuron. Taking
    the tail window by window at its largest weight,

        sum_{k >= Q} alpha (1-alpha)^{k-1} r_{t-k} <= e^{-U/tau} (alpha/dt) / (1 - e^{-t_abs/tau})
                                                   <= e^{-U/tau} / (tau (1 - e^{-t_abs/tau})),

    using alpha/dt <= 1/tau. No dt survives, so the bound holds at every step size. A source
    with no dead time has no ceiling, and a population it drives, either sign, gets A = inf. """
    t_abs = torch.as_tensor(t_abs, dtype=w.dtype, device=w.device).expand_as(E)
    tau = tau_int[..., :, None]                                   # target's synapse, (..., M, 1)
    per_spike = 1.0 / (tau * -torch.expm1(-t_abs[..., None, :] / tau))   # (..., M, M_source)
    return E.abs() + torch.where(w != 0, w.abs() * per_spike, 0.0).sum(dim=-1)


def field_tail(U, K_ref: torch.Tensor, tau_ref: torch.Tensor, tau_int: torch.Tensor,
               drive: torch.Tensor) -> torch.Tensor:
    """ Delta(U) = K_ref e^{-U/tau_ref} + A e^{-U/tau_int}, (..., M): the largest field error
    the absorbing bin can commit on a cohort older than U (U >= t_abs, in ms). """
    return K_ref * torch.exp(-U / tau_ref) + drive * torch.exp(-U / tau_int)


def age_window(beta: torch.Tensor, K_ref: torch.Tensor, tau_ref: torch.Tensor,
               tau_int: torch.Tensor, drive: torch.Tensor, tol: float) -> torch.Tensor:
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
    hi = torch.maximum(tau_ref * torch.log(2 * K_ref / target),
                       tau_int * torch.log(2 * drive / target)).clamp_min(0.0)
    lo = torch.zeros_like(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        ok = field_tail(mid, K_ref, tau_ref, tau_int, drive) <= target
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
            alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ One step along the characteristics, given the input I (..., M) and hazards fprobs
    (..., M, Q) of the CURRENT state. Returns (m, p, S) of the next one; nothing is modified
    in place. """
    m = compute_firing_rate(p, fprobs)
    p_new = update_age_distribution(p, fprobs)
    S_new = update_synaptic_term(S, alpha[..., None], I[..., None])
    return m, p_new, S_new
