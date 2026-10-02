from __future__ import annotations

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

def sync_hazard(h, beta, lam0, dt):
    """ Phi = sigma(beta*h): the synchronous model of rdme.mean_field, in which every neuron is
    updated every step. Tied to the step at which it is defined -- its rate sigma(beta*h)/dt
    diverges as dt -> 0 -- so it is the dt = dt0 member of the two families below rather than a
    model that can be refined. """
    return torch.sigmoid(beta * h)


def d_sync_hazard(h, beta, lam0, dt):
    """ dPhi/dh = beta*Phi*(1 - Phi). This is the derivative baked into
    rdme.mean_field.update_tangent, which is what makes that function sigmoid-specific. """
    phi = torch.sigmoid(beta * h)
    return beta * phi * (1.0 - phi)


def sync_rate(h, beta, lam0, dt):
    """ sigma(beta*h)/dt. Divergent by construction: at fixed field, halving the bin doubles
    the opportunities to fire. Returned anyway so the three hazards share an interface and the
    divergence is visible rather than implicit. """
    return torch.sigmoid(beta * h) / dt


def async_hazard(h, beta, lam0, dt):
    """ Phi = lam0*dt * sigma(beta*h): asynchronous updates, i.e. linear thinning.

    Each neuron is updated with probability lam0*dt per step and, if updated, fires with the
    synchronous probability; a neuron that is not updated stays silent. Marginalizing the
    update schedule gives this per-bin kernel, with rate lam0*sigma(beta*h).

    Valid only for lam0*dt <= 1 -- beyond that this is not a probability, since sigma
    approaches 1 and Phi would exceed it. RDMNetwork enforces that at construction. """
    return (lam0 * dt) * torch.sigmoid(beta * h)


def d_async_hazard(h, beta, lam0, dt):
    """ dPhi/dh = lam0*dt * beta*sigma*(1 - sigma): the synchronous derivative, thinned. """
    sig = torch.sigmoid(beta * h)
    return (lam0 * dt) * beta * sig * (1.0 - sig)


def async_rate(h, beta, lam0, dt):
    """ lam0*sigma(beta*h), bounded by lam0: at most one spike per update interval on average,
    which is exactly the assumption the synchronous model rests on. """
    return lam0 * torch.sigmoid(beta * h)


def escape_hazard(h, beta, lam0, dt):
    """ Phi = 1 - exp(-lam*dt) = 1 - (1 + e^{beta*h})^(-lam0*dt): the exact survival of a
    Poisson process whose rate lam = lam0*softplus(beta*h) is frozen across the bin.

    The rate is calibrated so that the exact survival reproduces the sigmoid at lam0*dt = 1,
    which is what 1 - sigma(x) = e^{-softplus(x)} gives.

    Computed as -expm1(-lam0*dt*softplus(beta*h)) rather than from the power form: torch's
    softplus uses the log1p(exp(-|x|)) + max(x, 0) form and so does not overflow at the large
    beta*h this model reaches, and expm1 keeps the small lam0*dt regime -- the whole point of
    refining dt -- from losing precision to 1 - (1 - x). """
    return -torch.expm1(-lam0 * dt * torch.nn.functional.softplus(beta * h))


def d_escape_hazard(h, beta, lam0, dt):
    """ dPhi/dh = (1 - Phi) * lam0*dt * beta*sigma(beta*h).

    Written from (1 - Phi) rather than re-exponentiating, so the two agree to the last bit and
    a saturated hazard gives a derivative of exactly zero instead of a small negative one. """
    phi = escape_hazard(h, beta, lam0, dt)
    return (1.0 - phi) * (lam0 * dt) * beta * torch.sigmoid(beta * h)


def escape_rate(h, beta, lam0, dt):
    """ lam0*softplus(beta*h), unbounded: grows like lam0*beta*h at strong drive, describing
    firing faster than one spike per update interval. """
    return lam0 * torch.nn.functional.softplus(beta * h)


class Hazard(NamedTuple):
    """ A per-bin kernel, its field-derivative, and the rate it represents.

    dphi is carried because the tangent dynamics needs dPhi/dh and it is not recoverable from
    Phi alone; rate because the three differ in the limit precisely through it, so comparing
    rates is comparing models while comparing Phi is comparing models-and-resolutions. """
    phi:  Callable[..., torch.Tensor]
    dphi: Callable[..., torch.Tensor]
    rate: Callable[..., torch.Tensor]
    bounded_step: bool   # whether lam0*dt <= 1 is required for Phi to be a probability


HAZARDS: dict[str, Hazard] = {
    "synchronous": Hazard(sync_hazard,   d_sync_hazard,   sync_rate,   False),
    "async":       Hazard(async_hazard,  d_async_hazard,  async_rate,  True),
    "escape":      Hazard(escape_hazard, d_escape_hazard, escape_rate, False),
}

HAZARDS["sync"] = HAZARDS["synchronous"]   # alias: the kernel's functional form
HAZARDS["sigmoid"] = HAZARDS["synchronous"]   # alias: the kernel's functional form
