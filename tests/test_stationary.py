""" Stationarity: the closed forms in rdme.equations, and the dynamics sitting on them. """

import math

import pytest
import torch

import rdme.equations as eq
from conftest import SETTLED, WC
from rdme.mean_field import (
    RDMIsingModel,
    RDMNetwork,
    RDMWilsonCowan,
    RDMWilsonCowanPoints,
)


@pytest.mark.parametrize("n_abs", [0, 3])
def test_stationary_field_is_the_geometric_sum(n_abs):
    """ At constant input the synaptic term is a fixed point of its own update, clamped at 0
    through the window and on its first eligible bin, and integrating from there. """
    Q, alpha, I = 12, 0.3, torch.tensor(1.7)
    blocked = torch.arange(Q) < n_abs
    S = eq.compute_stationary_forward_field(I, alpha, blocked)
    assert bool((S[:n_abs + 1] == 0).all())
    assert S[n_abs + 1].item() == pytest.approx(alpha * I.item(), abs=1e-15)
    assert torch.allclose(eq.update_synaptic_term(S, alpha, I, blocked), S, rtol=0, atol=1e-14)


def test_stationary_age_distribution_is_stationary():
    """ p* is normalized, fixed by the renewal update, and its births equal its spikes. """
    phi = torch.rand(15, generator=torch.Generator().manual_seed(0)).clamp(0.05, 0.95)
    p, m = eq.compute_stationary_age_distribution(phi)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-14)
    assert torch.allclose(eq.update_age_distribution(p, phi), p, rtol=0, atol=1e-14)
    assert p[0].item() == pytest.approx(m.item(), abs=1e-14)
    assert (p * phi).sum().item() == pytest.approx(m.item(), abs=1e-14)


def test_stationary_rate_is_the_untruncated_renewal_rate():
    """ With the boundary far out, m* is 1 / sum_n Survival(n). """
    phi = torch.full((400,), 0.05)
    _, m = eq.compute_stationary_age_distribution(phi)
    surv = eq.compute_stationary_survival(phi)
    assert m.item() == pytest.approx(1.0 / surv.sum().item(), rel=1e-6)


@pytest.mark.parametrize("build", [
    lambda: RDMWilsonCowan(**SETTLED, dt=0.2, hazard="poisson_exact"),      # M = 2, Newton
    lambda: RDMIsingModel(J=30.0, E=0.2, beta=15.0, theta=1.0, tau_int=20.0, tau_ref=3.0,
                          K_ref=0.0, dt=0.2, hazard="poisson_exact", t_abs=0.0,
                          Q=461),  # 5 roots; no dead time, so no bounded window: Q by hand
], ids=["wilson-cowan", "ising-multistable"])
def test_fixed_points_do_not_move(build):
    """ Every root, stable or not, is a state the dynamics leaves where it is. The run is
    short because an unstable root amplifies roundoff away from itself, which is correct. """
    roots = build().fixed_points()
    assert roots.numel() > 0
    for r in roots:
        net = build()
        net.set_state(r)
        net.forward(20)
        assert torch.allclose(net.overlaps(), r, rtol=0, atol=1e-12), (net.overlaps(), r)


def test_stationary_rate_matches_the_dynamics_at_a_fixed_point():
    """ Settled by integration rather than placed: rates() equals [int Survival du]^{-1}. """
    net = RDMWilsonCowan(**SETTLED, dt=0.1, hazard="poisson_exact")
    net.forward(80000)
    before = net.rates().clone()
    net.forward(500)
    assert torch.allclose(net.rates(), before, rtol=0, atol=1e-10), "not settled"
    assert torch.allclose(net.rates(), net.stationary_rate(), rtol=1e-6, atol=0)


# The dead-time transfer function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# No coupling, K_ref = 0, no input and threshold -x: a renewal process with dead time t_abs and
# constant hazard nu0 e^x, whose rate is eq.renewal_rate. The drive goes through theta, not
# through E with an instantaneous synapse: S is clamped at 0 on the first eligible bin
# (equations.update_synaptic_term), which would add a first-order error of its own there.

def _free_run(x, dt, *, nu0=1.0, t_abs=1.0, tol=1e-4, eq_ms=80.0, rec_ms=40.0):
    """ Time-averaged rate: the hard window makes the age distribution ring with period
    t_abs, so one instant of rates() is not the stationary rate. """
    net = RDMIsingModel(J=0.0, E=0.0, beta=1.0, theta=-x, tau_int=1.0, tau_ref=1.0,
                        K_ref=0.0, dt=dt, nu0=nu0, t_abs=t_abs, hazard="poisson_exact",
                        tol=tol)
    net.forward(int(eq_ms / dt))
    r = float(net.trajectory(int(rec_ms / dt), pb=False)["m"][:, 0].mean()) / dt
    return r, net.truncation_error()


def test_renewal_rate_has_both_closed_forms():
    for x, nu0, t_abs in ((-1.0, 1.0, 1.0), (0.5, 2.0, 0.5)):
        assert eq.renewal_rate(x, nu0, t_abs) == pytest.approx(
            1.0 / (t_abs * (1.0 + math.exp(-(x + math.log(nu0 * t_abs))))), rel=1e-12)


@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_stationary_rate_is_the_dead_time_transfer_function(x):
    """ The exact-survival map overestimates the rate by at most lambda*dt/2, single-signed,
    and the error halves with dt: first-order convergence onto the closed form. """
    lam, want = math.exp(x), eq.renewal_rate(x, 1.0, 1.0)
    errs = []
    for dt in (0.02, 0.01):
        got, trunc = _free_run(x, dt)
        assert trunc < 1e-3, trunc
        err = (got - want) / want
        assert 0 < err < 1.5 * 0.5 * lam * dt, (dt, err)
        errs.append(err)
    assert errs[1] == pytest.approx(errs[0] / 2, rel=0.1), errs


@pytest.mark.slow
def test_the_dead_time_caps_the_rate():
    """ At a hazard of e^3 = 20/ms the realised rate is the closed form, below 1/t_abs. """
    x, dt = 3.0, 0.002
    want = eq.renewal_rate(x, 1.0, 1.0)
    got, trunc = _free_run(x, dt, tol=1e-7, eq_ms=60.0, rec_ms=20.0)
    assert trunc < 1e-6, trunc
    assert got < 1.0
    assert got == pytest.approx(want, rel=2e-3), (got, want)


def test_each_element_has_its_own_dead_time():
    """ t_abs is a kernel parameter like tau_ref, so a batch may sweep it, and each element is
    then the single system with its own window. """
    T_ABS, dt = [0.5, 1.0, 2.0], 0.25
    b = RDMWilsonCowanPoints(**WC, t_abs=torch.tensor(T_ABS), dt=dt, hazard="poisson_exact")
    assert b.n_abs.tolist() == [[1, 1], [3, 3], [7, 7]]
    tb = b.trajectory(300, pb=False)
    for i, t_abs in enumerate(T_ABS):
        ts = RDMWilsonCowan(**WC, t_abs=t_abs, dt=dt, hazard="poisson_exact",
                            Q=b.Q).trajectory(300, pb=False)
        assert (tb["m"][i] - ts["m"]).abs().max().item() < 1e-14, t_abs


def _two_populations(t_abs, w):
    one = lambda v: [v, v]
    return RDMNetwork(2, torch.tensor(w), [800, 200], one(1.0), one(30.0), one(1.0), one(6.0),
                      one(3.0), one(0.5), t_abs=t_abs, dt=0.25, hazard="poisson_exact", Q=60)


def test_each_population_has_its_own_dead_time():
    """ The window masks exactly n_abs bins of each population's own hazard; a population
    with t_abs = 0 has none. """
    net = _two_populations([2.0, 0.0], [[0.5, -1.0], [1.0, -1.0]])
    assert net.n_abs.tolist() == [7, 0]
    assert net.blocked[0].tolist() == [True] * 7 + [False] * 53 and not net.blocked[1].any()
    assert net.R[0, 7] == -net.K_ref[0]
    assert torch.equal(net.R[1], -net.K_ref[1] * torch.exp(-((torch.arange(60) + 1) * 0.25) / 3.0))
    net.forward(200)


def test_a_window_above_an_unbounded_source_is_allowed():
    """ The window is a mask, so it needs no bound on the drive: a windowed population excited
    by one with no dead time is a valid model (with Q given, since the age window does need
    the source's 1/t_abs ceiling). """
    net = _two_populations([1.0, 0.0], [[0.5, 1.0], [1.0, -1.0]])
    net.forward(200)
    assert bool(torch.isfinite(net.rates()).all())
    assert bool((net.firing_prob()[0, :3] == 0).all())
