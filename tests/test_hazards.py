""" The three kernels against their closed forms. Under route B, `poisson` and `poisson_exact`
discretize the same rate nu0 e^{beta h}; `bernoulli` is the legacy route-A kernel. """

import math

import pytest
import torch

import rdme.hazards as hzd


ALL = ("bernoulli", "poisson", "poisson_exact")
ROUTE_B = ("poisson", "poisson_exact")


@pytest.mark.parametrize("name", ALL)
def test_hazards_are_probabilities(name):
    phi = hzd.HAZARDS[name].phi(torch.linspace(-8.0, 8.0, 201), 30.0, 1.0, 0.1)
    assert bool(((phi >= 0) & (phi <= 1)).all())


@pytest.mark.parametrize("name", ALL)
def test_hazard_derivatives_match_autograd(name):
    beta, nu0, dt = 30.0, 1.0, 0.1
    h = torch.linspace(-2.0, 2.0, 101, requires_grad=True)
    hz = hzd.HAZARDS[name]
    auto = torch.autograd.grad(hz.phi(h, beta, nu0, dt).sum(), h)[0]
    assert torch.allclose(hz.dphi(h.detach(), beta, nu0, dt), auto, rtol=0, atol=1e-11)


@pytest.mark.parametrize("name", ROUTE_B)
def test_the_rate_is_the_exponential_tilt_of_the_reference(name):
    beta, nu0 = 2.0, 1.3
    h = torch.linspace(-4.0, 4.0, 81)
    assert torch.allclose(hzd.HAZARDS[name].rate(h.clone(), beta, nu0, 0.1),
                          nu0 * torch.exp(beta * h), rtol=1e-14, atol=0)


@pytest.mark.parametrize("name", ROUTE_B)
def test_hazard_over_dt_converges_to_the_rate_at_first_order(name):
    """ Phi/dt -> nu0 e^{beta h}, and halving dt halves the error. """
    hz, beta, nu0 = hzd.HAZARDS[name], 1.0, 1.0
    for x in (-8.0, -2.0, 0.0, 2.0):
        h = torch.tensor([x])
        want = nu0 * math.exp(x)
        errs = [abs(float(hz.phi(h.clone(), beta, nu0, dt)) / dt - want) / want
                for dt in (1e-3, 5e-4, 2.5e-4)]
        assert errs[1] == pytest.approx(errs[0] / 2, rel=0.05), (x, errs)
        assert errs[2] == pytest.approx(errs[1] / 2, rel=0.05), (x, errs)


def test_poisson_exact_is_the_survival_of_its_rate():
    """ Phi = 1 - exp(-lambda dt). """
    h, beta, nu0, dt = torch.linspace(-3.0, 3.0, 121), 2.0, 1.0, 0.05
    hz = hzd.HAZARDS["poisson_exact"]
    lam = hz.rate(h.clone(), beta, nu0, dt)
    assert torch.allclose(hz.phi(h.clone(), beta, nu0, dt), -torch.expm1(-lam * dt),
                          rtol=0, atol=1e-15)


def test_poisson_is_the_logistic_kernel_at_the_reference_log_odds():
    """ Phi = sigma(beta h + logit(nu0 dt)). """
    h, beta, nu0, dt = torch.linspace(-3.0, 3.0, 121), 2.0, 1.0, 0.125
    ell = math.log(nu0 * dt) - math.log1p(-nu0 * dt)
    assert torch.allclose(hzd.HAZARDS["poisson"].phi(h.clone(), beta, nu0, dt),
                          torch.sigmoid(beta * h + ell), rtol=0, atol=1e-15)


def test_bernoulli_at_the_unit_step_is_the_sigmoid():
    """ At nu0 dt = 1 the thinning factor is one and Phi = sigma(beta h) bit for bit. """
    beta, h = 6.0, torch.linspace(-4.0, 4.0, 801)
    hz = hzd.HAZARDS["bernoulli"]
    for nu0, dt in ((1.0, 1.0), (2.0, 0.5), (4.0, 0.25)):
        sig = torch.sigmoid(beta * h)
        assert torch.equal(hz.phi(h.clone(), beta, nu0, dt), sig)
        assert torch.allclose(hz.dphi(h.clone(), beta, nu0, dt), beta * sig * (1 - sig),
                              rtol=0, atol=1e-15)
