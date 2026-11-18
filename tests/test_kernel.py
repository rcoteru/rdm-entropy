""" The refractory kernel is Gerstner's SRM_0 kernel: dead time, then -K_ref exp(-(u - t_abs)/tau_ref),
evaluated at the interspike interval u = (n+1)*dt each age bin stands for. """

import math

import pytest
import torch

import rdme.equations as eq
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel

T_ABS, K, TAU = 1.0, 2.0, 3.0
DTS = [1.0, 0.5, 0.25, 0.2, 0.125, 0.1, 0.05]


def _kernel(dt, K=K, tau=TAU, t_abs=T_ABS):
    return eq.refractory_kernel(int(20 / dt), K, tau, dt=dt, t_abs=t_abs, R_abs=100.0)


@pytest.mark.parametrize("dt", DTS)
def test_first_eligible_bin_is_minus_K(dt):
    R = _kernel(dt)
    n_abs = eq.n_abs_bins(T_ABS, dt)
    assert bool((R[:n_abs] == -100.0).all())
    assert R[n_abs].item() == -K


@pytest.mark.parametrize("u", [1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
def test_kernel_at_fixed_age_does_not_depend_on_dt(u):
    on_grid = [dt for dt in DTS if abs(u / dt - round(u / dt)) < 1e-9]
    vals = [_kernel(dt)[round(u / dt) - 1].item() for dt in on_grid]
    assert vals == pytest.approx([-K * math.exp(-(u - T_ABS) / TAU)] * len(on_grid), rel=1e-12)


@pytest.mark.parametrize("K_ref,tau_ref", [(0.0, 3.0), (2.0, 0.0), (0.0, 0.0)])
def test_no_relative_term_is_pure_dead_time(K_ref, tau_ref):
    """ K_ref = 0 or tau_ref = 0: no nan anywhere, and exactly the dead-time-only model. """
    dt = 0.25
    R = _kernel(dt, K=K_ref, tau=tau_ref)
    n_abs = eq.n_abs_bins(T_ABS, dt)
    assert bool((R[:n_abs] == -100.0).all()) and bool((R[n_abs:] == 0.0).all())

    one = torch.tensor([1.0])
    tail = eq.field_tail(5.0, K_ref * one, tau_ref * one, T_ABS * one, 6.0 * one, 1.0 * one)
    assert tail.item() == pytest.approx(math.exp(-5.0 / 6.0))
    U = eq.age_window(1.0 * one, K_ref * one, tau_ref * one, T_ABS * one, 6.0 * one, 1.0 * one,
                      1e-6)
    assert bool(torch.isfinite(U).all())

    kw = dict(J=2.0, E=0.5, beta=2.0, theta=1.0, tau_int=6.0, dt=dt, t_abs=T_ABS,
              hazard="poisson_exact")
    mf = RDMIsingModel(tau_ref=tau_ref, K_ref=K_ref, **kw)
    dead = RDMIsingModel(tau_ref=3.0, K_ref=0.0, **kw)          # the dead-time-only model
    assert torch.equal(mf.R, dead.R)
    a, b = mf.trajectory(400, pb=False), dead.trajectory(400, pb=False)
    assert bool(torch.isfinite(a["m"]).all()) and torch.equal(a["m"], b["m"])
    assert math.isfinite(mf.truncation_error().item())

    torch.manual_seed(0)
    sm = SpinIsingModel(500, tau_ref=tau_ref, K_ref=K_ref, ic="silent", **kw)
    for _ in range(400):
        sm.update()
        assert bool(torch.isfinite(sm.R).all())
        assert bool((sm.R_eff(sm.R, sm.n)[sm.n >= sm.n_abs] == 0.0).all())
