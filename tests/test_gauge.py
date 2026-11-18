""" The gauge move (nu0, theta) -> (nu0 e^{beta delta}, theta + delta) changes nothing: only
nu0 e^{-beta theta} is identifiable. Exact for poisson_exact; for poisson it holds in the
reference log-odds logit(nu0 dt) instead. """

import math

import pytest
import torch

import rdme.hazards as hzd
from rdme.mean_field import RDMIsingModel


@pytest.mark.parametrize("delta", [-0.3, 0.1, 0.7])
def test_poisson_exact_phi_is_gauge_invariant(delta):
    beta, nu0, dt, theta = 6.0, 1.0, 0.1, 0.5
    S = torch.linspace(-1.0, 2.0, 201)
    hz = hzd.HAZARDS["poisson_exact"]
    a = hz.phi(S - theta, beta, nu0, dt)
    b = hz.phi(S - (theta + delta), beta, nu0 * math.exp(beta * delta), dt)
    assert torch.allclose(a, b, rtol=1e-14, atol=1e-300)


@pytest.mark.parametrize("delta", [-0.3, 0.1, 0.7])
def test_poisson_phi_is_gauge_invariant_in_the_log_odds(delta):
    """ (ell, theta) -> (ell + beta delta, theta + delta); the naive nu0 e^{beta delta} is a
    different kernel at finite dt. """
    beta, nu0, dt, theta = 2.0, 1.0, 0.1, 0.5
    S = torch.linspace(-1.0, 2.0, 201)
    ell = math.log(nu0 * dt) - math.log1p(-nu0 * dt)
    nu0_b = (1.0 / (1.0 + math.exp(-(ell + beta * delta)))) / dt
    hz = hzd.HAZARDS["poisson"]
    a = hz.phi(S - theta, beta, nu0, dt)
    assert torch.allclose(a, hz.phi(S - (theta + delta), beta, nu0_b, dt), rtol=1e-13, atol=1e-300)
    if abs(delta) > 0.2:
        naive = hz.phi(S - (theta + delta), beta, nu0 * math.exp(beta * delta), dt)
        assert not torch.allclose(naive, a, rtol=1e-6, atol=0)


def _pair(delta, beta=6.0):
    P = {"J": 10.0, "E": 0.0, "beta": beta, "tau_int": 15.0, "tau_ref": 15.0, "K_ref": 15.0,
             "dt": 0.25, "t_abs": 1.0, "hazard": "poisson_exact"}
    return (RDMIsingModel(**P, theta=0.5, nu0=1.0),
            RDMIsingModel(**P, theta=0.5 + delta, nu0=math.exp(beta * delta)))


@pytest.mark.parametrize("delta", [-0.2, 0.3])
def test_the_trajectory_is_gauge_invariant(delta):
    """ Exact up to rounding, absolute window included: it is a mask, Phi = 0 on both sides,
    not a floor placed relative to theta. """
    a, b = _pair(delta)
    n = int(a.n_abs[0])
    assert bool((a.firing_prob()[0][:n] == 0).all()) and bool((b.firing_prob()[0][:n] == 0).all())
    for _ in range(400):
        a.update(); b.update()
        assert torch.allclose(a.rates(), b.rates(), rtol=1e-10, atol=1e-14)


def test_sigma_is_gauge_invariant():
    """ sigma is a difference of two large entropies, so it is checked on its own. """
    a, b = _pair(0.3)
    a.forward(2000); b.forward(2000)
    ta, tb = a.entropy_trajectory(60, pb=False), b.entropy_trajectory(60, pb=False)
    for k in ("m", "sigma", "H_fwd", "H_rev"):
        assert torch.allclose(ta[k], tb[k], rtol=1e-9, atol=1e-14), k
