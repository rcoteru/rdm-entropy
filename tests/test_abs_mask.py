""" The absolute refractory window is a hard mask, Gerstner's R = -inf taken literally: Phi = 0,
ln(1 - Phi) = 0, dPhi/dh = 0 and ln Phi = -inf on the blocked bins, and every weight those
bins carry in the entropy production is exactly 0, forward and reverse. """

import math

import pytest
import torch

import rdme.epr as epr
import rdme.equations as eq
import rdme.hazards as hzd
from conftest import S1
from rdme.mean_field import RDMIsingModel, RDMNetwork
from rdme.spin_model import SpinIsingModel

HZ = "poisson_exact"


def _cycle(**kw):
    return RDMIsingModel(**S1 | dict(hazard=HZ, J=17.0, dt=0.25) | kw)


def test_blocked_bins_carry_exactly_zero_hazard():
    net = _cycle()
    net.forward(2000)
    b = net.blocked
    assert int(b.sum()) == int(net.n_abs.sum()) > 0
    for f in (net.firing_prob(), net.d_firing_prob(), net.hazard_rate()):
        assert bool((f[b] == 0).all()) and bool((f[~b] > 0).all())


@pytest.mark.parametrize("method", ["sliding", "joint"])
def test_blocked_bins_carry_exactly_zero_epr_weight(monkeypatch, method):
    """ Every call of epr_from_parts, from the initial condition on: the spike weights on
    blocked bins are exactly 0 and meet ln Phi = -inf, not a finite stand-in. """
    seen = []
    real = epr.epr_from_parts

    def spy(p_fwd, p_rev, spike_f, spike_r, lp_fwd, l1_fwd, lp_rev, l1_rev):
        b = net.blocked
        assert bool((spike_f[b] == 0).all()) and bool((spike_r[b] == 0).all())
        assert bool((lp_fwd[b] == -math.inf).all()) and bool((lp_rev[b] == -math.inf).all())
        assert bool((l1_fwd[b] == 0).all()) and bool((l1_rev[b] == 0).all())
        out = real(p_fwd, p_rev, spike_f, spike_r, lp_fwd, l1_fwd, lp_rev, l1_rev)
        seen.append(out)
        return out

    monkeypatch.setattr(epr, "epr_from_parts", spy)
    net = _cycle()
    e = net.entropy_trajectory(80, pb=False, method=method)        # straight from the IC
    assert len(seen) == 80
    assert all(bool(torch.isfinite(v).all()) for v in e.values())


def test_a_nonzero_weight_on_a_blocked_bin_is_not_hidden():
    """ The 0 * -inf = 0 rule applies to a zero weight only: anything else stays infinite. """
    z, one = torch.zeros(3), torch.ones(3) / 3
    lp = torch.tensor([-math.inf, -1.0, -1.0])
    ok = epr.epr_from_parts(one, one, z, z, lp, z, lp, z)
    assert bool(torch.isfinite(ok).all())
    bad = epr.epr_from_parts(one, one, torch.tensor([1e-30, 0, 0]), z, lp, z, lp, z)
    assert bad[1].item() == math.inf


def test_spins_never_fire_in_the_window_and_their_epr_is_finite():
    torch.manual_seed(0)
    sm = SpinIsingModel(500, **S1 | dict(hazard=HZ, J=17.0, dt=0.25), ic="silent")
    for _ in range(400):
        b = sm.blocked(sm.n)
        assert bool((sm.firing_prob()[b] == 0).all())
        sm.update()
    e = sm.entropy_trajectory(300, method="full", lookahead=200, pb=False)
    assert all(bool(torch.isfinite(v).all()) for v in e.values())
    with pytest.raises(ValueError, match="absolute window"):
        sm.entropy_trajectory(10, lookahead=sm.n_abs, pb=False)


def test_zero_gain_with_a_window_is_allowed():
    """ No floor to size, so beta = 0 needs no special case: the hazard is constant outside
    the window and exactly 0 inside it. """
    net = _cycle(beta=0.0)
    net.forward(400)
    phi = net.firing_prob()[0]
    assert bool((phi[net.blocked[0]] == 0).all()) and bool(torch.isfinite(net.rates()).all())
    torch.manual_seed(0)
    sm = SpinIsingModel(200, **S1 | dict(hazard=HZ, J=17.0, dt=0.25) | {"beta": 0.0},
                        ic="silent")
    sm.forward(100)


def test_no_window_is_the_unmasked_model():
    """ t_abs = 0: nothing is blocked, and the hazard is the kernel's own, bit for bit. """
    net = _cycle(t_abs=0.0, Q=600)
    assert not bool(net.blocked.any()) and int(net.n_abs) == 0
    net.forward(400)
    want = eq.per_population(hzd.HAZARDS[HZ].phi, net.field(), net.beta, net.nu0, net.dt)
    assert torch.equal(net.firing_prob(), want)
    a = net.entropy_trajectory(40, pb=False)
    b = _cycle(t_abs=0.0, Q=600)
    b.forward(400)
    j = b.entropy_trajectory(40, pb=False, method="joint")
    assert torch.allclose(a["sigma"], j["sigma"], rtol=1e-9, atol=1e-14)


def test_q_must_reach_past_the_window():
    with pytest.raises(ValueError, match="absolute window"):
        RDMNetwork(1, torch.tensor([[1.0]]), [100], [0.0], [6.0], [0.5], [15.0], [15.0], [15.0],
                   t_abs=1.0, dt=0.25, hazard=HZ, Q=3)
