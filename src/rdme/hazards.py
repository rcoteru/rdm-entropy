from __future__ import annotations

from typing import Callable, NamedTuple
import math

import torch

""" Per-bin firing kernels and the rates they represent.

Model-independent: everything here is a pure function of the field h, the gain beta, the rate
unit nu0 and the step dt. Nothing knows about populations, age distributions or networks, the
same way the root finders in rdme.fixed_points know nothing about the model whose roots they
locate.

The kernels and their relationships are documented below; rdme is what applies them.
"""


# Hazards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# THE MODEL IS THE POISSON REFERENCE (route B). Writing x = beta*h, sigma for the logistic
# function, nu0 for the rate unit and c = nu0*dt for the reference's per-bin firing
# probability:
#
#   poisson        Phi = sigma(x + ell),  ell = logit(c)      rate nu0 e^x
#   poisson_exact  Phi = 1 - exp(-nu0 e^x dt)                 rate nu0 e^x
#   bernoulli      Phi = nu0*dt * sigma(x)                    rate nu0*sigma(x)
#
# ONE model, TWO kernels for it. poisson and poisson_exact share a rate function exactly and
# so are the same continuous-time model, reached by two different discretization maps:
# maximum caliber at each bin, and exact survival. The pair exists because the two maps have
# different error structure, which is a question about the solver rather than about the model
# -- see poisson_exact_hazard.
#
# bernoulli is NOT the model. It is route A, kept for comparison and for equivalence claims
# about the time-discrete construction the project started from: a neuron is offered an update
# with probability nu0*dt per bin and, if updated, fires with the synchronous probability
# sigma(x). At nu0*dt = 1 it IS the sigmoid, bit for bit on all four branches -- see
# bernoulli_log1m_phi, where that edge is taken deliberately -- which is the discrete model
# every pre-route-B result was computed in. Nothing in this repo should pick it as the hazard
# of a measurement.
#
# Why the Poisson reference is the model. Maximum caliber against a reference measure that
# fires with probability c per bin regardless of the field, under a mean-activity constraint,
# gives a logistic kernel with the reference's log-odds added to the field, at EVERY step.
# That is `poisson`, and it is MaxCal at every dt rather than at one privileged step, so
# refining dt refines one variational problem instead of walking across a family of them. The
# reference is a Poisson process of rate nu0, which is also the model's own reference measure
# in the entropy production: H and H_rev are reported against it (finite as dt -> 0) rather
# than against a counting measure (divergent as rate*ln(1/dt)).
#
# nu0 is a UNIT, not a calibration. It is the rate at which the reference process fires, fixed
# at 1 ms^-1, and the exponential tilt e^{beta*h} is measured against it; (nu0, theta) ->
# (nu0 e^{beta delta}, theta + delta) leaves Phi unchanged for poisson_exact exactly, which is
# the reparametrization the tests pin down. There is no time quantum: dt is the solver step
# and nothing else, and t_abs -- the physical dead time -- is an independent parameter of the
# model rather than a shadow of the step.
#
# The ceiling. poisson and poisson_exact have an UNBOUNDED, exponential rate, so nothing in
# the kernel caps the firing rate; the cap is the absolute refractory window t_abs, which
# bounds the realised rate by 1/t_abs. That is the SRM convention and is why a run with
# t_abs = 0 can reach arbitrary rates. bernoulli's rate saturates at nu0 instead, one spike
# per update interval, a property of the discrete state space it was built on.
#
# Every kernel takes `inplace`, with the same meaning as in rdme.rdm: the out-of-place
# branch is the definition and the oracle, the in-place branch is the same arithmetic in the
# same ORDER with the temporaries folded away, so the two agree bit for bit and the tests can
# assert torch.equal rather than allclose. inplace=True is a licence to destroy h -- the
# result may alias it, and the caller must own h and not read it afterwards -- which is why it
# is never the default: every caller here builds h as a throwaway `S + R - theta`, but a
# caller holding a field it still needs would be silently corrupted.
#
# The saving is not marginal where it matters. The entropy production evaluates a hazard on a
# (B, K, Q) buffer, so each avoided temporary is the size of that buffer; bernoulli_hazard
# alone goes from three of them to none. Only softplus is irreducible, torch exposing no
# in-place form and no `out=`, so the kernels built on it keep exactly one.


# Route A: the Bernoulli (linear-thinning) kernel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bernoulli_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ Phi = nu0*dt * sigma(beta*h): asynchronous updates, i.e. linear thinning.

    Each neuron is updated with probability nu0*dt per step and, if updated, fires with the
    synchronous probability; a neuron that is not updated stays silent. Marginalizing the
    update schedule gives this per-bin kernel, with rate nu0*sigma(beta*h).

    Route A, kept for comparison only. Valid for nu0*dt <= 1 -- beyond that this is not a
    probability, since sigma approaches 1 and Phi would exceed it; `check_step` enforces it. """
    if inplace:
        return h.mul_(beta).sigmoid_().mul_(nu0 * dt)
    return (nu0 * dt) * torch.sigmoid(beta * h)


def d_bernoulli_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ dPhi/dh = nu0*dt * beta*sigma*(1 - sigma): the synchronous derivative, thinned. """
    if inplace:
        sig = h.mul_(beta).sigmoid_()
        comp = 1.0 - sig
        return sig.mul_((nu0 * dt) * beta).mul_(comp)
    sig = torch.sigmoid(beta * h)
    return (nu0 * dt) * beta * sig * (1.0 - sig)


def bernoulli_rate(h, beta, nu0, dt, inplace: bool = False):
    """ nu0*sigma(beta*h), bounded by nu0: at most one spike per update interval on average,
    which is exactly the assumption the synchronous model rests on. """
    if inplace:
        return h.mul_(beta).sigmoid_().mul_(nu0)
    return nu0 * torch.sigmoid(beta * h)


def bernoulli_logphi(h, beta, nu0, dt, inplace: bool = False):
    """ ln Phi = ln(nu0*dt) - softplus(-beta*h): the synchronous log-probability, shifted by
    the thinning factor. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(-beta)).neg_().add_(math.log(nu0 * dt))
    return math.log(nu0 * dt) - torch.nn.functional.softplus(-beta * h)


def bernoulli_log1m_phi(h, beta, nu0, dt, inplace: bool = False):
    """ ln(1 - nu0*dt*sigma(beta*h)), written as ln(1 + (1-c)e^x) - softplus(x) with
    c = nu0*dt.

    The algebraic rearrangement, rather than log1p of the hazard: at c < 1 this is bounded
    below by ln(1 - c) -- a thinned neuron cannot fire with probability one -- but at c = 1 the
    silent branch does close, and the naive form reaches it as log1p(-1) = -inf several digits
    before the exact -softplus(x) does. Taking the limit here instead keeps the nu0*dt = 1
    edge, where bernoulli coincides with the synchronous sigmoid kernel, agreeing with it to
    the last bit rather than collapsing. That edge is the legacy discrete model, and is what
    test_hazards pins against the closed form. """
    c = nu0 * dt
    log1mc = math.log1p(-c) if c < 1.0 else -math.inf
    if inplace:
        x = h.mul_(beta)
        sp = torch.nn.functional.softplus(x)
        # the zero of the logaddexp is a 0-dim tensor rather than zeros_like(x): it broadcasts
        # to the same values and is the one temporary here that would be buffer-sized
        return torch.logaddexp(x.new_zeros(()), x.add_(log1mc), out=x).sub_(sp)
    x = beta * h
    return torch.logaddexp(torch.zeros_like(x), x + log1mc) - torch.nn.functional.softplus(x)


# Route B: the Poisson reference, by maximum caliber
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _poisson_logit(nu0: float, dt: float) -> float:
    """ ell = logit(c) with c = nu0*dt, the reference log-odds, in float64 Python.

    Once per call, not per element: it is a property of the (nu0, dt) pair and folding it into
    a tensor op would cost a buffer. c -> 1 is where the reference fires every bin and ell
    diverges; `check_step` refuses dt there, so this is only ever called with c < 1. """
    c = nu0 * dt
    if not 0.0 < c < 1.0:
        raise ValueError(f"the poisson kernel needs 0 < c = nu0*dt < 1, got c = {c:.6g} "
                         f"(nu0={nu0}, dt={dt}).")
    return math.log(c) - math.log1p(-c)


def poisson_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ Phi = sigma(beta*h + ell): maximum caliber against a Poisson reference measure.

    The reference fires with probability c = nu0*dt per bin, independently of the field, and
    MaxCal with a mean-activity constraint tilts it by e^{beta*h}, giving a logistic kernel
    whose field is shifted by the reference log-odds ell = logit(c). That is the content of
    the kernel, and the reason it is the model: bernoulli is MaxCal only at nu0*dt = 1, where
    the reference is unbiased, while this one is MaxCal at EVERY dt, so refining dt here is a
    refinement of one variational problem rather than a walk across a family of them.

    Its continuum rate is nu0 e^{beta*h}: Phi/dt -> c e^x / dt = nu0 e^x as c -> 0. There is
    no privileged dt at which it reduces to the sigmoid, and that is the point -- under route
    B nothing is calibrated to a time quantum. """
    if inplace:
        return h.mul_(beta).add_(_poisson_logit(nu0, dt)).sigmoid_()
    return torch.sigmoid(beta * h + _poisson_logit(nu0, dt))


def d_poisson_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ dPhi/dh = beta*Phi*(1 - Phi): the logistic derivative at the shifted field.

    Written in d_bernoulli_hazard's order -- the gain folded in first, then the complement --
    so that the two forms are the same arithmetic in the same order wherever they are
    compared. """
    if inplace:
        sig = h.mul_(beta).add_(_poisson_logit(nu0, dt)).sigmoid_()
        comp = 1.0 - sig
        return sig.mul_(beta).mul_(comp)
    sig = torch.sigmoid(beta * h + _poisson_logit(nu0, dt))
    return beta * sig * (1.0 - sig)


def poisson_rate(h, beta, nu0, dt, inplace: bool = False):
    """ nu0 * e^{beta*h}, the dt -> 0 limit of Phi/dt: an exponential escape rate, the exp-link
    GLM, and the rate function of the model under route B.

    The prefactor is the rate UNIT nu0 and carries no factor of one half: there is no quantum
    to pin the reference at, so c = nu0*dt throughout and the tilt is measured against a
    Poisson process of rate nu0. Equivalently (nu0, theta) -> (nu0 e^{beta delta}, theta +
    delta) is an exact reparametrization, so only nu0 e^{-beta*theta} is identifiable and nu0
    may as well be 1 ms^-1.

    Unbounded, unlike bernoulli's: nothing in the kernel caps it, and the cap on the realised
    rate is the dead time t_abs.

    Overflow. The exponential has no ceiling, so at the drives a strongly-coupled run reaches
    beta*h of a few hundred sends e^x past float32's 3.4e38. The RESULT is clamped at
    finfo.max, not the exponent: a rate that large is already far outside the regime any
    observable is read in, and clamping the result keeps the clamp visible as a finite number
    rather than silently flattening the field dependence below it. """
    if inplace:
        y = h.mul_(beta).exp_().mul_(nu0)
        return y.clamp_max_(torch.finfo(y.dtype).max)
    y = torch.exp(beta * h) * nu0
    return y.clamp_max(torch.finfo(y.dtype).max)


def poisson_logphi(h, beta, nu0, dt, inplace: bool = False):
    """ ln Phi = -softplus(-(beta*h + ell)), exact at both tails: at strong drive it goes to
    0 from below rather than losing the digits of a Phi that has rounded to one, and at weak
    drive it goes to beta*h + ell linearly rather than to ln(0). """
    if inplace:
        x = h.mul_(beta).add_(_poisson_logit(nu0, dt)).neg_()
        return torch.nn.functional.softplus(x).neg_()
    return -torch.nn.functional.softplus(-(beta * h + _poisson_logit(nu0, dt)))


def poisson_log1m_phi(h, beta, nu0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -softplus(beta*h + ell), the mirror image of logphi and exact for the
    same reason. This is the branch bernoulli has to rearrange to keep; here it is the closed
    form already, the logistic kernel being symmetric under x -> -x. """
    if inplace:
        x = h.mul_(beta).add_(_poisson_logit(nu0, dt))
        return torch.nn.functional.softplus(x).neg_()
    return -torch.nn.functional.softplus(beta * h + _poisson_logit(nu0, dt))


# Route B: the same model, by exact survival
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# The continuum rate of this kernel IS poisson's, so the function is shared rather than
# duplicated. Comparing the two rates has to compare the same object: two copies that happened
# to agree today could drift tomorrow, and the whole point of the pair is that they are one
# model under two discretization maps.
poisson_exact_rate = poisson_rate


def _poisson_exact_y(h, beta, nu0, dt, inplace: bool = False):
    """ y = lambda*dt = nu0 e^{beta*h} dt, the exponent the survival is built from.

    Factored out because all four branches below need exactly it, in exactly this order, and
    writing it four times is how the in-place and out-of-place forms drift apart. Clamped at
    finfo.max a second time after the multiplication by dt: poisson_rate already caps the rate
    itself, but rate*dt can still overflow at dt > 1, and an infinite y gives 1 - Phi = 0 and
    then 0 * inf = nan in the derivative. Clamped, that corner reads as a saturated hazard
    with a derivative of exactly zero, which is what it is. """
    if inplace:
        y = poisson_rate(h, beta, nu0, dt, inplace=True).mul_(dt)
        return y.clamp_max_(torch.finfo(y.dtype).max)
    y = poisson_rate(h, beta, nu0, dt) * dt
    return y.clamp_max(torch.finfo(y.dtype).max)


def poisson_exact_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ Phi = 1 - exp(-lambda*dt) with lambda = nu0 e^{beta*h}: the exact survival of a Poisson
    process whose EXPONENTIAL rate is frozen across the bin.

    The same continuous-time model as "poisson" -- identical rate function, identical dt -> 0
    limit -- reached by a different discretization map, and the SOLVER the dt -> 0 reporting
    rule is run with. poisson gets to the limit by maximum caliber at each bin, which makes it
    the right model for binned data but gives it a per-bin error that CHANGES SIGN at
    threshold: writing x = beta*h and c = nu0*dt,

        poisson:  Phi/dt = nu0 e^x / (1 + c(e^x - 1))

    sits above the rate for x < 0 and below it for x > 0. On an attractor that crosses
    threshold the quiet and bursting phases then contribute O(dt) errors of opposite sign, and
    their balance moves with dt -- a non-monotone approach to the limit, not a wrong limit, and
    one that Richardson extrapolation cannot be run on. Exact survival is single-signed (it
    always underestimates, 1 - e^{-z} <= z), so it approaches the same limit monotonically.

    Use "poisson" to MODEL binned data, "poisson_exact" to SOLVE the model -- and to extrapolate
    it, which is why it is the default hazard of a measurement. Its local relative error is
    bounded by lambda*dt/2; `step_error` reports it and `check_accuracy` tests it. """
    if inplace:
        return _poisson_exact_y(h, beta, nu0, dt, inplace=True).neg_().expm1_().neg_()
    return -torch.expm1(-_poisson_exact_y(h, beta, nu0, dt))


def d_poisson_exact_hazard(h, beta, nu0, dt, inplace: bool = False):
    """ dPhi/dh = (1 - Phi) * beta * y, with y = lambda*dt.

    Through y rather than re-forming nu0*dt*beta*e^x: it is the same product with one rounding
    fewer, and (1 - Phi) is built from the expm1 that produced Phi rather than from Phi itself,
    so a saturated hazard gives exactly zero. """
    if inplace:
        y = _poisson_exact_y(h, beta, nu0, dt, inplace=True)
        comp = y.neg().expm1_().add_(1.0)
        return comp.mul_(beta).mul_(y)
    y = _poisson_exact_y(h, beta, nu0, dt)
    comp = 1.0 + torch.expm1(-y)
    return comp * beta * y


def poisson_exact_logphi(h, beta, nu0, dt, inplace: bool = False):
    """ ln Phi = ln(-expm1(-y)). Through expm1 so a rarely-firing neuron, where Phi ~ lambda*dt,
    keeps full precision rather than bottoming out in 1 - (1 - x). The clamp is for y
    underflowing to zero at extreme negative drive, where ln Phi is -inf and only multiplies an
    event that did not happen. """
    if inplace:
        y = _poisson_exact_y(h, beta, nu0, dt, inplace=True)
        return y.neg_().expm1_().neg_().clamp_min_(torch.finfo(y.dtype).tiny).log_()
    y = _poisson_exact_y(h, beta, nu0, dt)
    return torch.log((-torch.expm1(-y)).clamp_min(torch.finfo(y.dtype).tiny))


def poisson_exact_log1m_phi(h, beta, nu0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -y exactly: the survival the kernel is defined by, read off before it is
    exponentiated. Finite wherever y is, including the drives at which Phi itself rounds to
    one. """
    if inplace:
        return _poisson_exact_y(h, beta, nu0, dt, inplace=True).neg_()
    return -_poisson_exact_y(h, beta, nu0, dt)


# The registry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Hazard(NamedTuple):
    """ A per-bin kernel, its field-derivative, the rate it represents, and the two branches
    of its log-probability.

    dphi is carried because the tangent dynamics needs dPhi/dh and it is not recoverable from
    Phi alone; rate because the models differ in the limit precisely through it, so comparing
    rates is comparing models while comparing Phi is comparing models-and-resolutions. There
    are TWO distinct rate functions here, bernoulli's and poisson's; poisson_exact shares
    poisson's, which is exactly how one reads off that the two are one model.

    logphi/log1m_phi are carried for the same reason as dphi: ln Phi and ln(1 - Phi) each have
    a closed form that stays exact where Phi itself has lost its digits -- at strong drive
    poisson_exact rounds to one in float32, and ln(1 - Phi) computed from it is -inf, while the
    direct form -lambda*dt is perfectly finite. The spin EPR estimator reads single spikes
    rather than an age distribution, so it pays that cost per neuron and needs the exact
    branches.

    step_cap is the largest dt the kernel's ARITHMETIC admits at a given nu0, or None when
    nothing caps it. It is not an accuracy criterion: see step_error. All five callables take
    the same trailing `inplace` flag; see the note above the kernels. """

    phi:  Callable[..., torch.Tensor]
    dphi: Callable[..., torch.Tensor]
    rate: Callable[..., torch.Tensor]
    logphi:     Callable[..., torch.Tensor]
    log1m_phi:  Callable[..., torch.Tensor]
    step_cap: Callable[[float], float | None]

    def neg_log_prob(self, s, h, beta, nu0, dt, inplace: bool = False):
        """ -ln p(s | h) for one Bernoulli spike, s in {0, 1}, broadcast over h.

        Selected with where() rather than weighted as -(s*lnPhi + (1-s)*ln(1-Phi)): a branch
        that is multiplied by zero may still be -inf, and 0 * -inf is nan, which is exactly the
        saturated-drive case this is wanted for.

        Both branches need h, so inplace can only be handed to the second one -- which is why
        this saves one buffer rather than the two the kernels manage on their own. """
        if inplace:
            lp = self.logphi(h, beta, nu0, dt)
            l1 = self.log1m_phi(h, beta, nu0, dt, inplace=True)
            return torch.where(s > 0, lp, l1).neg_()
        return -torch.where(s > 0,
                            self.logphi(h, beta, nu0, dt),
                            self.log1m_phi(h, beta, nu0, dt))


# The step caps, as functions of nu0. Each returns the largest admissible dt, or None.
#
#   bernoulli      Phi = nu0*dt*sigma would exceed 1 above dt = 1/nu0: an update probability
#                  above one. The edge itself is admissible and is the legacy discrete model.
#   poisson        c = nu0*dt is a probability and ell = logit(c) diverges as c -> 1, where
#                  the reference measure fires every bin and the tilt has nothing left to
#                  tilt. The cap is STRICT: c < 1, not c <= 1.
#   poisson_exact  Phi = 1 - e^{-y} is a probability for every y, so nothing caps dt here. The
#                  step is set by accuracy alone -- see step_error -- which is the other half
#                  of why this is the kernel the dt -> 0 limit is taken with.

def _cap_bernoulli(nu0: float) -> float: return 1.0 / nu0
def _cap_poisson(nu0: float) -> float:   return math.nextafter(1.0 / nu0, 0.0)
def _cap_none(nu0: float) -> None:       return None


HAZARDS: dict[str, Hazard] = {
    "bernoulli":     Hazard(bernoulli_hazard, d_bernoulli_hazard, bernoulli_rate,
                            bernoulli_logphi, bernoulli_log1m_phi, _cap_bernoulli),
    "poisson":       Hazard(poisson_hazard, d_poisson_hazard, poisson_rate,
                            poisson_logphi, poisson_log1m_phi, _cap_poisson),
    "poisson_exact": Hazard(poisson_exact_hazard, d_poisson_exact_hazard, poisson_exact_rate,
                            poisson_exact_logphi, poisson_exact_log1m_phi, _cap_none),
}

#: The hazard a measurement is run with unless it says otherwise (route B, 2026-11-07). It is
#: still its own axis and every script states it; this is what `presets.params` falls back to.
DEFAULT_HAZARD = "poisson_exact"


# Accuracy of the step
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def step_error(rate_max: float, dt: float) -> float:
    """ The local RELATIVE error bound of the exact-survival map at a peak rate: lambda*dt/2.

    Phi = 1 - e^{-lambda dt} = lambda*dt (1 - lambda*dt/2 + ...), so freezing the rate across a
    bin loses a relative lambda*dt/2 at the fastest-firing cohort, and that is the quantity to
    size dt against once no arithmetic cap binds. It is a LOCAL bound: the error of an
    observable accumulated over a trajectory is this per step, and the dt -> 0 reporting rule
    is what settles the global one.

    rate_max is read off the attractor, not from nu0: the rate is unbounded and exponential, so
    nu0*dt says nothing about it. RDMNetwork.hazard_rate() over a trajectory is where it comes
    from, exactly as max_step_prob is. """
    return 0.5 * rate_max * dt


def check_accuracy(rate_max: float, dt: float, tol: float) -> None:
    """ Raise unless the exact-survival step error at `rate_max` is within `tol`.

    Separated from check_step because it needs the realised rate, which is an output of the
    run: construction can only check the arithmetic, a settled trajectory can check this. """
    err = step_error(rate_max, dt)
    if err > tol:
        raise ValueError(
            f"step error {err:.3g} exceeds tolerance {tol:.3g} at dt={dt:g} with peak rate "
            f"{rate_max:.4g}/ms; refine dt below {2.0 * tol / rate_max:.4g} ms.")


def check_step(hazard: str, nu0: float, dt: float) -> None:
    """ Raise unless `dt` is arithmetically admissible for `hazard` at this `nu0`.

    The construction-time half of sizing the step: it sees only (nu0, dt) and so can only
    enforce what the kernel's algebra requires -- that Phi be a probability, and that the
    Poisson reference's log-odds be finite. Nothing here says the step is ACCURATE; that is
    check_accuracy, which needs the attractor. """
    if hazard not in HAZARDS:
        raise ValueError(f"hazard must be one of {sorted(HAZARDS)}, got {hazard!r}")
    cap = HAZARDS[hazard].step_cap(nu0)
    if cap is not None and dt > cap:
        raise ValueError(
            f"hazard {hazard!r} admits dt <= {cap:.6g} ms at nu0 = {nu0:g}/ms "
            f"(c = nu0*dt = {nu0 * dt:.6g}), got dt = {dt:g}. Refine dt.")


# The (dt, t_abs, nu0) convention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resolve_timesteps(dt: float, t_abs: float, nu0: float,
                      hazard: str) -> tuple[float, float, float]:
    """ Settle (dt, t_abs, nu0) for a model. Shared by RDMNetwork, RDMNetworkBatch and
    SpinModel so the convention is stated once rather than three times.

    Under route B there is no time quantum and no `dt0`. The three quantities are:

      nu0   the RATE UNIT, in 1/ms: the rate of the Poisson reference measure the kernel is a
            tilt of. Fixed at 1 ms^-1. It is not a free parameter -- (nu0, theta) ->
            (nu0 e^{beta delta}, theta + delta) is an exact reparametrization for
            poisson_exact, so only nu0 e^{-beta theta} is identifiable -- and it is carried
            explicitly anyway, so that a model never has to be read as "whatever 1/dt0 was".

      t_abs the absolute refractory period, in ms: a PHYSICAL dead time, an independent
            parameter of the model, and the only thing that bounds the realised firing rate
            under an exponential hazard (at 1/t_abs). It no longer defaults to the step; a
            model that wants no dead time passes t_abs = 0 explicitly and accepts an unbounded
            rate. kernels.n_abs_bins requires dt to divide it.

      dt    the SOLVER step, and nothing else. REQUIRED: under route B there is no discrete
            model to default to, so a model that has not named its step has not said how it is
            being integrated. It is the quantity to refine until observables stop moving.

    The reporting rule reduces to one case: every number comes from a dt -> 0 check, by
    default Richardson extrapolation from a coarse pair (dt, dt/2) with `poisson_exact`, whose
    convergence is monotone. `poisson`'s per-bin error changes sign at threshold, so Richardson
    on it fails on an attractor that crosses threshold; quote `poisson` at a stated dt, or
    extrapolate the solver and compare.

    Checks only what (dt, t_abs, nu0) can answer on their own -- positivity, and the
    per-kernel arithmetic cap of check_step. Accuracy needs the attractor: see check_accuracy.

    Returns (dt, t_abs, nu0). """
    if hazard not in HAZARDS:
        raise ValueError(f"hazard must be one of {sorted(HAZARDS)}, got {hazard!r}")
    if dt is None:
        raise ValueError("dt is required: under route B there is no time quantum to default "
                         "to, and the solver step is a choice the run has to state.")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not nu0 > 0:
        raise ValueError(f"nu0 must be positive, got {nu0}")
    if t_abs is None or t_abs < 0:
        raise ValueError(f"t_abs must be a non-negative number of ms, got {t_abs}. Pass "
                         "t_abs=0.0 explicitly for no dead time (and an unbounded rate).")
    check_step(hazard, nu0, dt)
    return float(dt), float(t_abs), float(nu0)
