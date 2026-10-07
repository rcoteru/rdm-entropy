from __future__ import annotations

import math
from typing import Callable, NamedTuple

import torch

""" Per-bin firing kernels and the rates they represent.

Model-independent: everything here is a pure function of the field h, the gain beta, the rate
calibration lam0 and the step dt. Nothing knows about populations, age distributions or
networks, the same way the root finders in rdme.fixed_points know nothing about the
model whose roots they locate.

The three kernels and their relationship are documented below; rdme is what applies them.
"""


# Hazards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Three per-bin kernels, all of which reduce to the synchronous model of rdme.mean_field at
# lam0*dt = 1, and two of which have a dt -> 0 limit. Writing x = beta*h and sigma for the
# logistic function:
#
#   synchronous   Phi = sigma(x)                      rate sigma(x)/dt      (no limit)
#   async         Phi = lam0*dt * sigma(x)            rate lam0*sigma(x)
#   escape        Phi = 1 - (1 + e^x)^(-lam0*dt)      rate lam0*softplus(x)
#
# They differ in two independent ways (App. "Discretization map and rate function"). The
# discretization map -- linear thinning against exact survival -- differs at second order,
# 1 - e^{-lam*dt} = lam*dt[1 - lam*dt/2 + ...], so it only changes the speed of convergence.
# The rate function persists in the limit, so async and escape converge to DIFFERENT
# continuous-time models. Their rates agree for weak drive, where both go like lam0*e^x, and
# separate for strong drive: the async rate saturates at lam0, one spike per update interval,
# while the escape rate grows like lam0*beta*h without bound.
#
# That difference is the point of carrying all three: it is a statement about the model, not
# about the numerics, and it shows up in the entropy production, which is a functional of Phi
# and so sees the rate function directly.
#
# Every kernel takes `inplace`, with the same meaning as in rdme.dynamics: the out-of-place
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

def sync_hazard(h: torch.Tensor, beta: torch.Tensor, lam0: float, dt: float, inplace: bool = False):
    """ Phi = sigma(beta*h): the synchronous model of rdme.mean_field, in which every neuron is
    updated every step. Tied to the step at which it is defined -- its rate sigma(beta*h)/dt
    diverges as dt -> 0 -- so it is the dt = dt0 member of the two families below rather than a
    model that can be refined. """
    if inplace:
        return h.mul_(beta).sigmoid_()
    return torch.sigmoid(beta * h)



def d_sync_hazard(h: torch.Tensor, beta: torch.Tensor, lam0: float, dt: float, inplace: bool = False):
    """ dPhi/dh = beta*Phi*(1 - Phi). This is the derivative baked into
    rdme.mean_field.update_tangent, which is what makes that function sigmoid-specific. """
    if inplace:
        phi = h.mul_(beta).sigmoid_()
        comp = 1.0 - phi                 # the one buffer the product cannot do without
        return phi.mul_(beta).mul_(comp)
    phi = torch.sigmoid(beta * h)
    return beta * phi * (1.0 - phi)


def sync_rate(h: torch.Tensor, beta: torch.Tensor, lam0: float, dt: float, inplace: bool = False):
    """ sigma(beta*h)/dt. Divergent by construction: at fixed field, halving the bin doubles
    the opportunities to fire. Returned anyway so the three hazards share an interface and the
    divergence is visible rather than implicit. """
    if inplace:
        return h.mul_(beta).sigmoid_().div_(dt)
    return torch.sigmoid(beta * h) / dt


def sync_logphi(h, beta, lam0, dt, inplace: bool = False):
    """ ln Phi = -softplus(-beta*h). The logistic identity, so a spike's log-probability is
    exact at any drive instead of going through ln(sigmoid) and losing the tail. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(-beta)).neg_()
    return -torch.nn.functional.softplus(-beta * h)


def sync_log1m_phi(h, beta, lam0, dt, inplace: bool = False):
    """ ln(1 - Phi) = -softplus(beta*h), the same identity on the silent branch. """
    if inplace:
        return torch.nn.functional.softplus(h.mul_(beta)).neg_()
    return -torch.nn.functional.softplus(beta * h)


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


class Hazard(NamedTuple):
    """ A per-bin kernel, its field-derivative, the rate it represents, and the two branches
    of its log-probability.

    dphi is carried because the tangent dynamics needs dPhi/dh and it is not recoverable from
    Phi alone; rate because the three differ in the limit precisely through it, so comparing
    rates is comparing models while comparing Phi is comparing models-and-resolutions.

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
    "synchronous": Hazard(sync_hazard,   d_sync_hazard,   sync_rate,
                          sync_logphi,   sync_log1m_phi,   False),
    "async":       Hazard(async_hazard,  d_async_hazard,  async_rate,
                          async_logphi,  async_log1m_phi,  True),
    "escape":      Hazard(escape_hazard, d_escape_hazard, escape_rate,
                          escape_logphi, escape_log1m_phi, False),
}

HAZARDS["sync"] = HAZARDS["synchronous"]   # alias: the kernel's functional form
HAZARDS["sigmoid"] = HAZARDS["synchronous"]   # alias: the kernel's functional form
