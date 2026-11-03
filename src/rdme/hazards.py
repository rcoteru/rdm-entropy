from __future__ import annotations

from typing import Callable, NamedTuple
import math

import torch

""" Per-bin firing kernels and the rates they represent.

Model-independent: everything here is a pure function of the field h, the gain beta, the rate
calibration lam0 and the step dt. Nothing knows about populations, age distributions or
networks, the same way the root finders in rdme.fixed_points know nothing about the
model whose roots they locate.

The kernels and their relationships are documented below; rdme is what applies them.
"""


# Hazards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Three per-bin kernels, all of which have a dt -> 0 limit and all of which reduce to the
# sigmoid at lam0*dt = 1. Writing x = beta*h, sigma for the logistic function, c = lam0*dt/2
# and ell = logit(c) = ln(c) - ln(1-c):
#
#   async         Phi = lam0*dt * sigma(x)            rate lam0*sigma(x)
#   escape        Phi = 1 - (1 + e^x)^(-lam0*dt)      rate lam0*softplus(x)
#   poisson       Phi = sigma(x + ell)                rate (lam0/2)*e^x
#   exp_surv      Phi = 1 - exp(-(lam0/2) e^x dt)     rate (lam0/2)*e^x
#
# THREE models, FOUR kernels: exp_surv shares poisson's rate function exactly and so is the
# same continuous-time model, reached by exact survival instead of by maximum caliber. The
# pair exists because the two discretization maps have different error structure, which is a
# question about the solver rather than about the model -- see exp_surv_hazard.
#
# poisson is the only one of the three that is MAXIMUM CALIBER at every dt. Maximum caliber
# against a Poisson reference measure -- one that fires with probability c per bin regardless
# of the field -- with a mean-activity constraint gives a logistic kernel with the reference's
# log-odds added to the field, at EVERY step, and c = lam0*dt/2 is the calibration that makes
# the reference unbiased (c = 1/2, ell = 0) exactly at the quantum. async and escape are
# MaxCal only there, at the single step where all three coincide; refining either of them
# leaves the variational family, while refining poisson stays inside it.
#
# At lam0*dt = 1 (dt = dt0, the time quantum) async IS the sigmoid, bit for bit on all four
# branches -- see async_log1m_phi, where that edge is taken deliberately. The synchronous
# model is therefore not a third kernel but async evaluated at the boundary of its own
# validity domain, and is not carried separately.
#
# The two differ in two independent ways (App. "Discretization map and rate function"). The
# discretization map -- linear thinning against exact survival -- differs at second order,
# 1 - e^{-lam*dt} = lam*dt[1 - lam*dt/2 + ...], so it only changes the speed of convergence.
# The rate function persists in the limit, so async and escape converge to DIFFERENT
# continuous-time models. Their rates agree for weak drive, where both go like lam0*e^x -- so
# both are escape-noise models in Gerstner's sense, sharing the Arrhenius tail -- and separate
# for strong drive: the async rate saturates at lam0, one spike per quantum, while the escape
# rate grows like lam0*beta*h without bound.
#
# That ceiling is the substantive difference, and the models put it in different places.
# async caps the hazard itself: a binary spike train binned at dt0 cannot encode a rate above
# 1/dt0, so the saturation is a property of the state space the model is defined on. escape
# leaves the hazard unbounded and delegates the limit to the refractory kernel, which is the
# SRM convention and is why a run without an absolute refractory window can reach >1 kHz.
#
# poisson has no ceiling either, and goes further than escape: its rate is exponential rather
# than linear at strong drive, so it is the stiffest of the three and the one whose step has to
# be checked against max(rate)*dt rather than against lam0*dt. At WEAK drive it carries a
# factor of one half -- async and escape both tend to lam0*e^x, poisson to (lam0/2)*e^x, the
# same Arrhenius tail shifted by ln2/beta in the threshold. That is a consequence of pinning
# the reference at the quantum and is deliberately not corrected for: moving the prefactor
# would move the dt = dt0 agreement, which is the fixed point of the whole construction.
#
# That difference is the point of carrying all three models: it is a statement about the
# model, not about the numerics, and it shows up in the entropy production, which is a
# functional of Phi and so sees the rate function directly. exp_surv is the one member added
# for a reason that IS about the numerics, and it is labelled as such.
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
# (B, K, Q) buffer, so each avoided temporary is the size of that buffer; async_hazard alone
# goes from three of them to none. Only softplus is irreducible, torch exposing no in-place
# form and no `out=`, so the kernels built on it keep exactly one.

def async_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ Phi = lam0*dt * sigma(beta*h): asynchronous updates, i.e. linear thinning.

    Each neuron is updated with probability lam0*dt per step and, if updated, fires with the
    synchronous probability; a neuron that is not updated stays silent. Marginalizing the
    update schedule gives this per-bin kernel, with rate lam0*sigma(beta*h).

    Valid only for lam0*dt <= 1 -- beyond that this is not a probability, since sigma
    approaches 1 and Phi would exceed it. RDMNetwork enforces that at construction. """
    if inplace:
        return h.mul_(beta).sigmoid_().mul_(lam0 * dt)
    return (lam0 * dt) * torch.sigmoid(beta * h)


def d_async_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ dPhi/dh = lam0*dt * beta*sigma*(1 - sigma): the synchronous derivative, thinned. """
    if inplace:
        sig = h.mul_(beta).sigmoid_()
        comp = 1.0 - sig
        return sig.mul_((lam0 * dt) * beta).mul_(comp)
    sig = torch.sigmoid(beta * h)
    return (lam0 * dt) * beta * sig * (1.0 - sig)


def async_rate(h, beta, lam0, dt, inplace: bool = False):
    """ lam0*sigma(beta*h), bounded by lam0: at most one spike per update interval on average,
    which is exactly the assumption the synchronous model rests on. """
    if inplace:
        return h.mul_(beta).sigmoid_().mul_(lam0)
    return lam0 * torch.sigmoid(beta * h)


def async_logphi(h, beta, lam0, dt, inplace: bool = False):
    """ ln Phi = ln(lam0*dt) - softplus(-beta*h): the synchronous log-probability, shifted by
    the thinning factor. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(-beta)).neg_().add_(math.log(lam0 * dt))
    return math.log(lam0 * dt) - torch.nn.functional.softplus(-beta * h)


def async_log1m_phi(h, beta, lam0, dt, inplace: bool = False):
    """ ln(1 - lam0*dt*sigma(beta*h)), written as ln(1 + (1-c)e^x) - softplus(x) with
    c = lam0*dt.

    The algebraic rearrangement, rather than log1p of the hazard: at c < 1 this is bounded
    below by ln(1 - c) -- a thinned neuron cannot fire with probability one -- but at c = 1 the
    silent branch does close, and the naive form reaches it as log1p(-1) = -inf several digits
    before the exact -softplus(x) does. Taking the limit here instead keeps the lam0*dt = 1
    edge, where async coincides with the synchronous kernel, agreeing with it to the last bit
    rather than collapsing. """
    c = lam0 * dt
    log1mc = math.log1p(-c) if c < 1.0 else -math.inf
    if inplace:
        x = h.mul_(beta)
        sp = torch.nn.functional.softplus(x)
        # the zero of the logaddexp is a 0-dim tensor rather than zeros_like(x): it broadcasts
        # to the same values and is the one temporary here that would be buffer-sized
        return torch.logaddexp(x.new_zeros(()), x.add_(log1mc), out=x).sub_(sp)
    x = beta * h
    return torch.logaddexp(torch.zeros_like(x), x + log1mc) - torch.nn.functional.softplus(x)


def escape_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ Phi = 1 - exp(-lam*dt) = 1 - (1 + e^{beta*h})^(-lam0*dt): the exact survival of a
    Poisson process whose rate lam = lam0*softplus(beta*h) is frozen across the bin.

    The rate is calibrated so that the exact survival reproduces the sigmoid at lam0*dt = 1,
    which is what 1 - sigma(x) = e^{-softplus(x)} gives.

    Computed as -expm1(-lam0*dt*softplus(beta*h)) rather than from the power form: torch's
    softplus uses the log1p(exp(-|x|)) + max(x, 0) form and so does not overflow at the large
    beta*h this model reaches, and expm1 keeps the small lam0*dt regime -- the whole point of
    refining dt -- from losing precision to 1 - (1 - x). """
    if inplace:
        sp = torch.nn.functional.softplus(h.mul_(beta))
        return sp.mul_(-lam0 * dt).expm1_().neg_()
    return -torch.expm1(-lam0 * dt * torch.nn.functional.softplus(beta * h))


def d_escape_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ dPhi/dh = (1 - Phi) * lam0*dt * beta*sigma(beta*h).

    Written from (1 - Phi) rather than re-exponentiating, so the two agree to the last bit and
    a saturated hazard gives a derivative of exactly zero instead of a small negative one. """
    if inplace:
        # the two factors both need beta*h, so it is kept and the survival built beside it;
        # add_(1.0) on expm1(-lam*dt) == -Phi is the same 1.0 - Phi the other branch forms
        x = h.mul_(beta)
        comp = torch.nn.functional.softplus(x).mul_(-lam0 * dt).expm1_().add_(1.0)
        return comp.mul_(lam0 * dt).mul_(beta).mul_(x.sigmoid_())
    phi = escape_hazard(h, beta, lam0, dt)
    return (1.0 - phi) * (lam0 * dt) * beta * torch.sigmoid(beta * h)


def escape_rate(h, beta, lam0, dt, inplace: bool = False):
    """ lam0*softplus(beta*h), unbounded: grows like lam0*beta*h at strong drive, describing
    firing faster than one spike per update interval. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(beta)).mul_(lam0)
    return lam0 * torch.nn.functional.softplus(beta * h)


def escape_logphi(h, beta, lam0, dt, inplace: bool = False):
    """ ln Phi = ln(-expm1(-lam*dt)). Through expm1 so a rarely-firing neuron, where
    Phi ~ lam*dt, keeps full precision rather than bottoming out in 1 - (1 - x). The clamp is
    for lam*dt underflowing to zero at extreme negative drive, where ln Phi is -inf and only
    multiplies an event that did not happen. """
    if inplace:
        y = torch.nn.functional.softplus(h.mul_(beta)).mul_(lam0 * dt)
        return y.neg_().expm1_().neg_().clamp_min_(torch.finfo(y.dtype).tiny).log_()
    y = lam0 * dt * torch.nn.functional.softplus(beta * h)
    return torch.log((-torch.expm1(-y)).clamp_min(torch.finfo(y.dtype).tiny))


def escape_log1m_phi(h, beta, lam0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -lam*dt exactly: the survival this hazard is built from, read off before
    it is exponentiated. This is why the escape EPR stays finite at drives where Phi itself
    saturates to one in float32. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(beta)).mul_(-(lam0 * dt))
    return -(lam0 * dt) * torch.nn.functional.softplus(beta * h)


def _poisson_logit(lam0: float, dt: float) -> float:
    """ ell = logit(c) with c = lam0*dt/2, the reference log-odds, in float64 Python.

    Once per call, not per element: it is a property of the (lam0, dt) pair and folding it into
    a tensor op would cost a buffer. c = 1/2 -- i.e. lam0*dt = 1, the time quantum -- is taken
    exactly rather than through log(0.5) - log1p(-0.5), whose two terms cancel to 0.0 in the
    real arithmetic but are not required to by the standard. That edge is the one that has to
    be exact: at ell = 0 every branch below must be async's, bit for bit. """
    c = lam0 * dt / 2.0
    if c == 0.5:
        return 0.0
    return math.log(c) - math.log1p(-c)


def poisson_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ Phi = sigma(beta*h + ell): maximum caliber against a Poisson reference measure.

    The reference fires with probability c = lam0*dt/2 per bin, independently of the field, and
    MaxCal with a mean-activity constraint tilts it by e^{beta*h}, giving a logistic kernel
    whose field is shifted by the reference log-odds ell = logit(c). That is the content of the
    kernel: async and escape are MaxCal only at the quantum, where the reference is unbiased,
    while this one is MaxCal at EVERY dt, which is what makes refining dt here a refinement of
    one variational problem rather than a walk across a family of them.

    The calibration c = lam0*dt/2 puts c = 1/2 at dt = dt0, where ell = 0 and Phi = sigma(x):
    the same discrete model the other two reduce to, so every dt = dt0 result carries over
    unchanged and only the refinement differs. """
    if inplace:
        return h.mul_(beta).add_(_poisson_logit(lam0, dt)).sigmoid_()
    return torch.sigmoid(beta * h + _poisson_logit(lam0, dt))


def d_poisson_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ dPhi/dh = beta*Phi*(1 - Phi): the logistic derivative at the shifted field.

    Written in d_async_hazard's order -- the gain folded in first, then the complement -- so
    that at ell = 0 the two agree to the last bit, async's (lam0*dt)*beta being exactly beta
    there. """
    if inplace:
        sig = h.mul_(beta).add_(_poisson_logit(lam0, dt)).sigmoid_()
        comp = 1.0 - sig
        return sig.mul_(beta).mul_(comp)
    sig = torch.sigmoid(beta * h + _poisson_logit(lam0, dt))
    return beta * sig * (1.0 - sig)


def poisson_rate(h, beta, lam0, dt, inplace: bool = False):
    """ (lam0/2) * e^{beta*h}, the dt -> 0 limit of Phi/dt: an exponential escape rate, the
    exp-link GLM. Unbounded, like escape and unlike async, and growing faster than escape's
    softplus at strong drive.

    The factor one half is real and is not a calibration error. Below threshold async and
    escape both go like lam0*e^{beta*h}, while this goes like (lam0/2)*e^{beta*h} -- the same
    Arrhenius tail displaced by ln2/beta in the threshold. It is the price of holding c = 1/2
    at the quantum: the reference measure that makes the kernel MaxCal at every step is also
    what halves its prefactor, and moving one moves the other.

    Overflow. The exponential has no ceiling, so at the drives a strongly-coupled run reaches
    beta*h of a few hundred sends e^x past float32's 3.4e38. The RESULT is clamped at
    finfo.max, not the exponent: a rate that large is already far outside the regime any
    observable is read in, and clamping the result keeps the clamp visible as a finite number
    rather than silently flattening the field dependence below it. """
    if inplace:
        y = h.mul_(beta).exp_().mul_(0.5 * lam0)
        return y.clamp_max_(torch.finfo(y.dtype).max)
    y = torch.exp(beta * h) * (0.5 * lam0)
    return y.clamp_max(torch.finfo(y.dtype).max)


def poisson_logphi(h, beta, lam0, dt, inplace: bool = False):
    """ ln Phi = -softplus(-(beta*h + ell)), exact at both tails: at strong drive it goes to
    0 from below rather than losing the digits of a Phi that has rounded to one, and at weak
    drive it goes to beta*h + ell linearly rather than to ln(0). """
    if inplace:
        x = h.mul_(beta).add_(_poisson_logit(lam0, dt)).neg_()
        return torch.nn.functional.softplus(x).neg_()
    return -torch.nn.functional.softplus(-(beta * h + _poisson_logit(lam0, dt)))


def poisson_log1m_phi(h, beta, lam0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -softplus(beta*h + ell), the mirror image of logphi and exact for the
    same reason. This is the branch async has to rearrange to keep; here it is the closed form
    already, the logistic kernel being symmetric under x -> -x. """
    if inplace:
        x = h.mul_(beta).add_(_poisson_logit(lam0, dt))
        return torch.nn.functional.softplus(x).neg_()
    return -torch.nn.functional.softplus(beta * h + _poisson_logit(lam0, dt))


# The continuum rate of this kernel IS poisson's, so the function is shared rather than
# duplicated. Comparing the two rates has to compare the same object: two copies that happened
# to agree today could drift tomorrow, and the whole point of the pair is that they are one
# model under two discretization maps.
exp_surv_rate = poisson_rate


def _exp_surv_y(h, beta, lam0, dt, inplace: bool = False):
    """ y = lambda*dt = (lam0/2) e^{beta*h} dt, the exponent the survival is built from.

    Factored out because all four branches below need exactly it, in exactly this order, and
    writing it four times is how the in-place and out-of-place forms drift apart. Clamped at
    finfo.max a second time after the multiplication by dt: poisson_rate already caps the rate
    itself, but rate*dt can still overflow at dt > 1, and an infinite y gives 1 - Phi = 0 and
    then 0 * inf = nan in the derivative. Clamped, that corner reads as a saturated hazard
    with a derivative of exactly zero, which is what it is. """
    if inplace:
        y = poisson_rate(h, beta, lam0, dt, inplace=True).mul_(dt)
        return y.clamp_max_(torch.finfo(y.dtype).max)
    y = poisson_rate(h, beta, lam0, dt) * dt
    return y.clamp_max(torch.finfo(y.dtype).max)


def exp_surv_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ Phi = 1 - exp(-lambda*dt) with lambda = (lam0/2) e^{beta*h}: the exact survival of a
    Poisson process whose EXPONENTIAL rate is frozen across the bin.

    The same continuous-time model as "poisson" -- identical rate function, identical dt -> 0
    limit -- reached by a different discretization map. poisson gets there by maximum caliber
    at each bin, which makes it the right model for binned data but gives it a per-bin error
    that CHANGES SIGN at threshold: writing x = beta*h and c = lam0*dt/2,

        poisson:  Phi/dt = r0 e^x / (1 + c(e^x - 1)),      r0 = lam0/2

    sits above the rate for x < 0 and below it for x > 0. On an attractor that crosses
    threshold the quiet and bursting phases then contribute O(dt) errors of opposite sign, and
    their balance moves with dt -- which is a non-monotone approach to the limit, not a wrong
    limit. Exact survival is single-signed (it always underestimates, 1 - e^{-z} <= z), so it
    approaches the same limit monotonically and is the better SOLVER for this model.

    It does NOT reduce to the sigmoid at lam0*dt = 1, and that is not a defect: it is not
    claiming to be a MaxCal model at any bin width, only to integrate poisson's continuum
    model accurately. Use "poisson" to MODEL binned data, "exp_surv" to SOLVE the model. """
    if inplace:
        return _exp_surv_y(h, beta, lam0, dt, inplace=True).neg_().expm1_().neg_()
    return -torch.expm1(-_exp_surv_y(h, beta, lam0, dt))


def d_exp_surv_hazard(h, beta, lam0, dt, inplace: bool = False):
    """ dPhi/dh = (1 - Phi) * beta * y, with y = lambda*dt.

    Through y rather than re-forming r0*dt*beta*e^x: it is the same product with one rounding
    fewer, and (1 - Phi) is built from the expm1 that produced Phi rather than from Phi
    itself, so a saturated hazard gives exactly zero -- the same reasoning as
    d_escape_hazard. """
    if inplace:
        y = _exp_surv_y(h, beta, lam0, dt, inplace=True)
        comp = y.neg().expm1_().add_(1.0)
        return comp.mul_(beta).mul_(y)
    y = _exp_surv_y(h, beta, lam0, dt)
    comp = 1.0 + torch.expm1(-y)
    return comp * beta * y


def exp_surv_logphi(h, beta, lam0, dt, inplace: bool = False):
    """ ln Phi = ln(-expm1(-y)). Through expm1 for the same reason escape_logphi is, and with
    the same clamp: at extreme negative drive y underflows to zero, ln Phi is -inf, and it
    only ever multiplies an event that did not happen. """
    if inplace:
        y = _exp_surv_y(h, beta, lam0, dt, inplace=True)
        return y.neg_().expm1_().neg_().clamp_min_(torch.finfo(y.dtype).tiny).log_()
    y = _exp_surv_y(h, beta, lam0, dt)
    return torch.log((-torch.expm1(-y)).clamp_min(torch.finfo(y.dtype).tiny))


def exp_surv_log1m_phi(h, beta, lam0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -y exactly: the survival the kernel is defined by, read off before it is
    exponentiated. Finite wherever y is, including the drives at which Phi itself rounds to
    one. """
    if inplace:
        return _exp_surv_y(h, beta, lam0, dt, inplace=True).neg_()
    return -_exp_surv_y(h, beta, lam0, dt)


class Hazard(NamedTuple):
    """ A per-bin kernel, its field-derivative, the rate it represents, and the two branches
    of its log-probability.

    dphi is carried because the tangent dynamics needs dPhi/dh and it is not recoverable from
    Phi alone; rate because the models differ in the limit precisely through it, so comparing
    rates is comparing models while comparing Phi is comparing models-and-resolutions. The
    three distinct rate functions are async's, escape's and poisson's; exp_surv shares
    poisson's, which is exactly how one reads off that the two are one model.

    logphi/log1m_phi are carried for the same reason as dphi: ln Phi and ln(1 - Phi) each have
    a closed form that stays exact where Phi itself has lost its digits -- at strong drive the
    escape hazard rounds to one in float32, and ln(1 - Phi) computed from it is -inf, while the
    direct form -lam*dt is perfectly finite. The spin EPR estimator reads single spikes rather
    than an age distribution, so it pays that cost per neuron and needs the exact branches.

    All five take the same trailing `inplace` flag; see the note above the kernels. """
    
    phi:  Callable[..., torch.Tensor]
    dphi: Callable[..., torch.Tensor]
    rate: Callable[..., torch.Tensor]
    logphi:     Callable[..., torch.Tensor]
    log1m_phi:  Callable[..., torch.Tensor]
    bounded_step: bool   # whether lam0*dt <= 1 is required for Phi to be a probability

    def neg_log_prob(self, s, h, beta, lam0, dt, inplace: bool = False):
        """ -ln p(s | h) for one Bernoulli spike, s in {0, 1}, broadcast over h.

        Selected with where() rather than weighted as -(s*lnPhi + (1-s)*ln(1-Phi)): a branch
        that is multiplied by zero may still be -inf, and 0 * -inf is nan, which is exactly the
        saturated-drive case this is wanted for.

        Both branches need h, so inplace can only be handed to the second one -- which is why
        this saves one buffer rather than the two the kernels manage on their own. """
        if inplace:
            lp = self.logphi(h, beta, lam0, dt)
            l1 = self.log1m_phi(h, beta, lam0, dt, inplace=True)
            return torch.where(s > 0, lp, l1).neg_()
        return -torch.where(s > 0,
                            self.logphi(h, beta, lam0, dt),
                            self.log1m_phi(h, beta, lam0, dt))


HAZARDS: dict[str, Hazard] = {
    "async":       Hazard(async_hazard,  d_async_hazard,  async_rate,
                          async_logphi,  async_log1m_phi,  True),
    "escape":      Hazard(escape_hazard, d_escape_hazard, escape_rate,
                          escape_logphi, escape_log1m_phi, False),
    # bounded_step=True is stricter than this kernel needs: Phi is a sigmoid and so is a
    # probability for every c < 1, i.e. for lam0*dt < 2. It is kept because the constraint
    # worth enforcing is not "Phi <= 1" but "dt <= dt0" -- the reference measure is calibrated
    # to be unbiased at the quantum, and a step above it is a reference that fires more often
    # than half the time, which is the discrete model running backwards rather than a model
    # being refined.
    "poisson":     Hazard(poisson_hazard, d_poisson_hazard, poisson_rate,
                          poisson_logphi, poisson_log1m_phi, True),
    # bounded_step=False: Phi = 1 - e^{-y} is a probability for every y, so nothing here caps
    # dt. The step is set by accuracy alone, max(rate)*dt << 1 on the attractor, exactly as
    # for escape -- and unlike poisson, which is pinned to dt <= dt0 by its reference measure
    # rather than by its arithmetic.
    "exp_surv":    Hazard(exp_surv_hazard, d_exp_surv_hazard, exp_surv_rate,
                          exp_surv_logphi, exp_surv_log1m_phi, False),
}


# The (dt0, dt) convention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resolve_timesteps(dt0: float, dt: float | None, hazard: str) -> tuple[float, float, float]:
    """ Settle (dt0, dt, lam0) for a model. Shared by RDMNetwork, RDMBatch and SpinNetwork so
    the convention is stated once rather than three times.

    dt0 is the MODEL. It fixes lam0 = 1/dt0 and so names one member of the hazard family; it is
    an intrinsic modelling choice -- the time quantum, equivalently the rate scale lam0 -- and
    is held fixed across a study. It is required, because a model that has not named its
    quantum has not been specified.

    dt is the SOLVER. It is the resolution the model is integrated at, and the quantity to
    refine until observables stop moving. It defaults to dt0.

    At the default dt = dt0 we sit at lam0*dt = 1, where all three hazards collapse onto the
    sigmoid: that is the time-discrete model. It is also where Phi can reach 1, so at strong
    drive a whole population fires in one bin and flips back the next -- the two-step artifact
    that refining dt is for. So the default is the baseline to refine away from, not a result.

    Refining is hazard-dependent and lam0*dt << 1 is not by itself the criterion:

      async  rate saturates at lam0, so Phi <= lam0*dt bounds the per-step firing probability
             everywhere and lam0*dt << 1 does suffice.
      escape rate lam0*softplus(beta*h) is unbounded, so at strong drive Phi can still approach
             1 at lam0*dt << 1. The criterion there is max(hazard_rate())*dt << 1, read off the
             actual attractor -- see RDMNetwork.max_step_prob.
      poisson refines like escape, and for the same reason only more so: the rate
             (lam0/2)*e^{beta*h} is unbounded and exponential, so lam0*dt << 1 says nothing
             about the per-step firing probability. The criterion is again
             max(hazard_rate())*dt << 1 on the attractor, NOT lam0*dt << 1.
      exp_surv same rate, same criterion, and no cap on dt at all -- it is not calibrated to
             the quantum, so lam0*dt > 1 is arithmetically fine and merely inaccurate. It is
             also the member whose approach to that criterion is monotone; poisson's per-bin
             error changes sign at threshold, so on an attractor that crosses threshold its
             observables can approach the limit from alternating sides.

    Returns (dt0, dt, lam0). """
    if hazard not in HAZARDS:
        raise ValueError(f"hazard must be one of {sorted(HAZARDS)}, got {hazard!r}")
    if not dt0 > 0:
        raise ValueError(f"dt0 must be positive, got {dt0}")
    dt = dt0 if dt is None else dt
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")

    lam0 = 1.0 / dt0
    if HAZARDS[hazard].bounded_step and lam0 * dt > 1.0 + 1e-12:
        # Phi = lam0*dt*sigma would exceed 1: an update probability above one
        raise ValueError(
            f"hazard {hazard!r} requires lam0*dt <= 1, got lam0*dt = {lam0 * dt:.4g} "
            f"(dt={dt}, dt0={dt0}). Refine dt, or raise dt0.")
    return dt0, dt, lam0
