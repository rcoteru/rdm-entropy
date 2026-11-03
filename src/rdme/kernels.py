from __future__ import annotations

import math

import torch

""" The exponential kernels the model is built from, and the bin counts they imply.

Pure functions of a time constant and a step: nothing here knows about populations, hazards or
networks. They are what ties the model's parameters, which are quoted in ms, to an age grid
quoted in bins -- which is why every one of them takes dt and why changing dt changes nothing
physical as long as it goes through these. """


def synaptic_kernel(Q, tau_m, delta):
    """ Synaptic kernel for the model.
    Normalized exponential kernel with time constant tau_m,
    evaluated at integer multiples of delta."""
    t = torch.arange(0, Q)
    return torch.exp(-t*delta/tau_m)/tau_m*delta


def n_abs_bins(t_abs: float, dt: float) -> int:
    """ Number of leading age bins an absolute refractory period of t_abs blocks.

    round(t_abs/dt) - 1, NOT round(t_abs/dt), and the off-by-one is the whole content of this
    function. A neuron that fires enters bin 0 and is first eligible again one step LATER, so
    firing from bin n produces an interspike interval of (n+1)*dt. Blocking bins 0..n-1
    therefore buys a dead time of (n+1)*dt, and asking for t_abs means n = t_abs/dt - 1.

    Getting this wrong is not a rounding detail: blocking round(t_abs/dt) bins gives a realised
    dead time of t_abs + dt, which MOVES with the solver step -- measured 1.5, 1.2, 1.1 ms for a
    nominal t_abs = 1 ms at dt = 0.5, 0.2, 0.1. A dead time that depends on dt is not an
    absolute refractory period.

    At t_abs = dt this returns 0, i.e. no blocking, which is the identity the whole mechanism
    rests on: with t_abs = dt0 the window is INVISIBLE at the baseline dt = dt0 and switches on
    only as dt is refined below it. It therefore does not change the discrete model; it is what
    makes refining dt refine that model rather than a different one. """
    return max(int(round(t_abs / dt)) - 1, 0)


def abs_refractory_floor(w: torch.Tensor,        # (M, M) weights, row = target
                         E: torch.Tensor,        # (M,) external drive
                         theta: torch.Tensor,    # (M,) threshold
                         beta: torch.Tensor,     # (M,) gain
                         r_max: float,           # (1/ms) largest rate the window itself allows
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

    S_max is bounded self-consistently. The window itself caps the rate at r_max = 1/t_abs, so
    S_max <= sum_b max(w_ab, 0) * r_max + max(E_a, 0) -- the argument closes on itself, which
    is why this works for an unbounded hazard like "escape" as well as for "async".

    It also covers "poisson", whose rate is exponential rather than linear in the field, and
    that is worth checking rather than assuming: an exponential outruns a softplus, so a rule
    sized on the latter need not hold for the former. It does here because the margin is a
    bound on the FIELD and not on the rate -- the floor puts beta*h at -margin, where the
    growth of the rate function is irrelevant and all three kernels sit on the same Arrhenius
    tail. Measured at margin = 40, beta = 6: Phi = 4.2e-18 for async and escape and
    2.1e-18 for poisson at dt = dt0 (the reference log-odds only ever shift the field DOWN,
    ell <= 0 for dt <= dt0, so poisson is the mildest of the three here, not the worst), and
    Phi falls by a further decade or two as dt refines. """
    S_max = w.clamp_min(0.0).sum(dim=-1) * r_max + E.clamp_min(0.0)
    return (S_max - theta + margin / beta).clamp_min(margin / beta)


def refractory_kernel(Q: int, K, tau_r, dt: float = 1.0,
                      t_abs: float = 0.0, R_abs=None, device='cpu') -> torch.Tensor:
    """ The refractory kernel: an exponential relative term, optionally floored by a hard
    absolute window on the first n_abs_bins(t_abs, dt) age bins.

        R(u) = -K exp(-u/tau_r),          u >= t_abs
             = -R_abs,                    u <  t_abs

    t_abs = 0 (the default) is the plain exponential kernel and the historical behaviour.

    Why the window exists at all, in two independent senses:

    1. MODELLING. The exponential term alone is a *relative* refractory mechanism: it depresses
       the field by -K e^{-u/tau_r}, so a drive strong enough makes S + R - theta positive and
       the neuron fires straight through its own refractoriness. Measured on a Wilson-Cowan
       sweep with K up to 30 and tau_r up to 22 ms, the strongly-driven corner fires at an
       effective dead time of ~0.3 ms against the 2.5-22 ms the kernel nominally imposes -- a
       median population rate of 1264 Hz. Raising K does not fix it robustly, because "large
       enough not to be overwhelmed" is a statement about the drive range. A hard window is
       drive-independent, which is the point.

    2. CONSISTENCY UNDER REFINEMENT, which is why t_abs defaults to dt0 in the model classes.
       The model is defined on binary spike trains binned at dt0, and such a train cannot carry
       two spikes closer than dt0. Nothing in the hazard enforces that: "async" caps the RATE
       at lam0 = 1/dt0 but not the interval, so refining dt below dt0 admits ISIs of dt. The
       cost is not marginal -- measured in the spin model at dt0 = 1 ms with weak
       refractoriness and strong drive, 25%, 51% and 56% of all interspike intervals fall below
       dt0 at dt = 0.5, 0.2 and 0.1, and the population rate inflates from 0.67 to 0.90 /ms.
       Setting t_abs = dt0 restores the floor exactly and costs nothing at dt = dt0, where the
       window is empty (see n_abs_bins).

    This is also the standard SRM treatment, where eta(u) is taken to -infinity below the
    absolute refractory period. Since the kernel enters the model only through S + R - theta,
    and nothing assumes its functional form, the window propagates to the dynamics, the fixed
    points, the Jacobian and the entropy production with no other edit.

    R_abs must be sized against the drive; abs_refractory_floor computes it. Passing a value
    too small is not an error but silently makes the window soft again, which is the failure
    this is here to prevent.

    K, tau_r and R_abs may be scalars or 0-dim tensors; the arithmetic is elementwise and
    torch.vmap-safe, so this drops into the per-element kernel construction of RDMNetworkBatch.
    """
    n = torch.arange(Q, device=device)
    K, tau_r = torch.as_tensor(K, device=device), torch.as_tensor(tau_r, device=device)
    R = -K * torch.exp(-n * dt / tau_r)
    n_abs = n_abs_bins(t_abs, dt)
    if n_abs == 0:
        return R
    if R_abs is None:
        raise ValueError(f"t_abs={t_abs} blocks {n_abs} bins but R_abs is None. Size it with "
                         "abs_refractory_floor(w, E, theta, beta, r_max=1/t_abs).")
    R_abs = torch.as_tensor(R_abs, device=device)
    return torch.where(n < n_abs, -R_abs.to(R.dtype), R)


def tau2alpha(tau: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Convert time constant tau to integration coefficient alpha. """
    return 1.0 - torch.exp(-dt / tau)


def q_from_tau(tau_m: float, dt: float, eps: float = 0.01) -> int:
    """
    Minimum number of bins Q such that the omitted tail mass of the
    normalized exponential kernel kappa(tau) = alpha * r**(tau-1),
    r = exp(-dt/tau_m), alpha = 1-r, falls below eps.

    Tail mass after Q terms is exactly r**Q = exp(-Q*dt/tau_m), so:
        r**Q < eps  =>  Q > -(tau_m/dt) * ln(eps)
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1)")
    tau_ratio = tau_m / dt          # tau_m expressed in units of dt (bins)
    Q = math.ceil(-tau_ratio * math.log(eps))
    return max(Q, 1)


def q_from_rate(rate: float, dt: float, eps: float = 0.01,
                dead_time: float = 0.0, lam_inf: float | None = None) -> int:
    """
    Minimum number of bins Q such that the stationary fraction of neurons OLDER than
    U_max = Q*dt falls below eps.

    The companion to q_from_tau, and the rule that actually binds in a low-rate regime.
    q_from_tau sizes the grid so the kernels have decayed; it says nothing about how much age
    mass is still out there when they have. Those are different questions whenever the mean
    interval is long compared with the kernels, and the gap is not a detail: at a measured
    cell with tau_ref = 2.8 ms and r = 7.7e-2/ms, q_from_tau at eps = 0.01 gives
    U_max = 12 ms, where 39% of the population is older than the truncation. The lumped
    boundary bin then holds that 39% at a single frozen field and the dynamics changes
    qualitatively -- a limit cycle collapses to a fixed point. At U_max = 24 ms the residual
    mass is 4.4% and the solution is converged to five digits against 48, 96 and 192 ms.

    The bound. For a renewal process the stationary age distribution is q(u) = r*Survival(u),
    so P[age > U] = r * int_U^inf Survival. With a flat hazard this is exactly e^{-r*U}, giving

        Q > -ln(eps) / (r*dt).

    That is the default and it is CONSERVATIVE for this model: refractoriness makes the
    survival flat over the dead time and then decay at the recovered hazard lam_inf >= r, and
    (r/lam_inf) * e^{-lam_inf*(U - T_d)} < e^{-r*U} for every positive dead time. Measured
    against the cell above, the Poisson form predicts 40% beyond 12 ms against 39% observed
    and 16% beyond 24 ms against 4.4%, i.e. it is tight where the mass is and loosens where it
    is not, which is the direction an upper bound should err in.

    Passing lam_inf (with dead_time) uses that sharper two-piece form instead,

        Q > (T_d - ln(eps * lam_inf / r) / lam_inf) / dt,

    which is worth it when the dead time dominates the interval -- a strongly refractory
    population at a modest rate, where the Poisson bound overshoots by a factor of a few. The
    recovered hazard is lam_inf = lam0 * sigma(beta*(S - theta)) for hazard='async' at the
    operating point, i.e. the hazard of the oldest bin, which `RDMNetwork.firing_prob()[a][-1]`
    reports directly as a per-bin probability.

    Getting `rate` at all. It is an output of the run, not an input, so in practice: size the
    grid generously once, measure, and size the production run from that -- which is cheap
    because the first run only has to settle, not to be accurate. `RDMNetwork.stationary_rate`
    is the per-population alternative at a settled state, and it corrects for exactly this
    truncation when it resums the absorbing bin.

    A population that can fall silent has no finite Q by this rule, and that is a true
    statement about the model rather than a failure of the bound: the age distribution really
    does run away. What saves that case is the absorbing bin, whose geometric resummation is
    exact PROVIDED the field has saturated there -- which is q_from_tau's job. The two rules
    are therefore not alternatives, and q_from_tau_and_rate below is the one to call.
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1)")
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}. A silent population has no "
                         "finite truncation age by this criterion; see the docstring.")
    if lam_inf is None:
        Q = -math.log(eps) / (rate * dt)
    else:
        if lam_inf < rate:
            raise ValueError(f"lam_inf={lam_inf} is below rate={rate}: the recovered hazard "
                             "cannot be slower than the mean rate it produces.")
        # P[age > U] = (r/lam_inf) * exp(-lam_inf*(U - T_d)) for U > T_d; invert for U, and
        # fall back to the dead time itself when eps is so loose that the tail is already under
        # it at U = T_d
        U = dead_time - math.log(max(eps * lam_inf / rate, 1e-300)) / lam_inf
        Q = max(U, dead_time) / dt
    return max(math.ceil(Q), 1)


def q_from_tau_and_rate(tau: float, rate: float, dt: float, eps: float = 0.01,
                        gain: float = 1.0, amplitude: float = 1.0, **kwargs) -> int:
    """
    Minimum Q from the TRUNCATION ERROR itself: the age mass beyond U_max times the field
    error it is held at, below eps.

    Neither q_from_tau nor q_from_rate is the criterion on its own, because the error the
    truncation actually commits is a product of the two. The boundary bin lumps every cohort
    older than U_max at one field value, and the appendix's geometric resummation of that tail
    is exact once the kernels have saturated there -- so a large residual mass is harmless if
    the field is right, and a wrong field is harmless if nothing is left to sit in it. What
    hurts is both at once.

    Measuring the two factors separately on the cell described in q_from_rate:

        U_max      mass beyond     gain*|R(U)|     product
         12 ms       3.9e-1          2.7e-1         1.0e-1
         18 ms       1.9e-1          3.2e-2         6.2e-3
         24 ms       4.4e-2          3.8e-3         1.7e-4

    and the solution is wrong at 12 ms, essentially right at 18 and converged at 24 -- which is
    the product column crossing 1e-2, not either factor crossing it alone.

    Measured against a Q = 400 reference on that cell, with eps = 0.01 throughout:

        rule                     Q      U_max     mean rate      CV     rel. error
        q_from_tau              26      13 ms     2.04e-2      0.000      7.3e-1
        (none)                  30      15 ms     2.91e-2      0.000      6.2e-1
        q_from_tau_and_rate     36      18 ms     7.35e-2      1.484      2.9e-2
        (none)                  48      24 ms     7.57e-2      1.499      3.9e-4
        q_from_rate            121      60 ms     7.57e-2      1.4997     1.8e-16

    The kernel rule alone does not merely lose accuracy, it returns the wrong attractor: a
    fixed point where the converged answer is a limit cycle of CV 1.5. The product rule
    recovers the attractor at a third of the grid the mass bound asks for, at the price of a
    few percent on the rate; tighten eps when that matters, since the realised error here runs
    well below eps rather than at it.

    The field error is measured against the GAIN, not against the kernel's own amplitude: what
    the hazard sees is beta*h, so an absolute field error delta matters in proportion to
    beta*delta. This is the same reason the kernel rule needs `gain` here while q_from_tau,
    which asks only about the kernel's own tail mass, does not. With gain = amplitude = 1 this
    reduces to q_from_tau's tolerance on a normalized kernel, which is why those are the
    defaults rather than required arguments.

    Combining P[age > U] <= e^{-rU} with |field error| ~ amplitude * e^{-U/tau},

        U * (rate + 1/tau) > ln(gain * amplitude / eps),

    which is the returned bound. For the refractory kernel take amplitude = K_ref and
    tau = tau_ref; for the synaptic term, amplitude = |I| and tau = tau_int, and the caller
    wants the larger of the two.

    The result is floored by q_from_tau, NOT by q_from_rate. Flooring by the mass bound would
    throw away everything this function is for -- on the cell above it would return 121 bins
    where 35 are demonstrably enough. The mass is allowed to be large precisely because the
    resummation absorbs it; what the resummation cannot absorb is cohorts that are NOT
    interchangeable, and in a time-varying input two cohorts agree only once their synaptic
    terms have both converged, S^a(u,t) = int_0^u kappa(s) I(t-s) ds being u-dependent for
    u <~ tau_int. That is a condition on the kernels alone, which is q_from_tau's question,
    and it is why the floor belongs there.
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1)")
    if tau <= 0.0 or gain <= 0.0 or amplitude <= 0.0:
        raise ValueError("tau, gain and amplitude must all be positive")
    if kwargs:
        # the sharper two-piece mass factor, as a correction to the Poisson exponent: the
        # product rule needs P[age > U], and q_from_rate already knows how to bound it
        mass_Q = q_from_rate(rate, dt, eps=eps, **kwargs)
        poisson_Q = q_from_rate(rate, dt, eps=eps)
        shift = mass_Q - poisson_Q
    else:
        shift = 0
    decay = rate + 1.0 / tau
    Q = math.log(gain * amplitude / eps) / (decay * dt) + shift
    return max(math.ceil(Q), q_from_tau(tau, dt, eps), 1)
