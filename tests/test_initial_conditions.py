""" The mean field and the spins start from the same initial conditions (equations.
INITIAL_CONDITIONS): at t = 0 the spins' age histogram is the mean field's p -- exactly when
silent, to sampling error when random -- and a random start's R is what update() would have
grown at that age. """

import pytest
import torch

import conftest
import rdme.equations as eq
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel

Q = 300


def _pair(ic, dt=0.25, t_abs=1.0, N=200_000, **over):
    kw = conftest.S1 | dict(hazard="poisson_exact", dt=dt, t_abs=t_abs, J=10.0) | over
    torch.manual_seed(0)
    return SpinIsingModel(N, **kw, ic=ic), RDMIsingModel(**kw, Q=Q, ic=ic)


def test_silent_starts_agree_exactly():
    sm, mf = _pair("silent", N=1000)
    assert torch.equal(sm.fdist(Q)[0].double(), mf.p[0])
    assert mf.p[0, -1] == 1.0 and mf.m.item() == 0.0
    assert not sm.s.any() and not sm.S.any() and not sm.R.any() and not mf.S.any()


@pytest.mark.parametrize("age_max", [25.0, 100.0])        # 100 ms runs past the Q-bin grid
def test_random_starts_agree_in_distribution(age_max):
    sm, mf = _pair("random", age_max=age_max)
    K = eq.random_age_bins(age_max, 0.25)
    assert torch.allclose(mf.p.sum(-1), torch.ones(1, dtype=torch.float64))
    assert mf.m.item() == pytest.approx(1 / K)
    # per bin, the count is binomial(N, p)
    p, N = mf.p[0], sm.N
    sd = (p * (1 - p) / N).sqrt().clamp_min(1e-12)
    assert ((sm.fdist(Q)[0].double() - p).abs() <= 5 * sd).all()
    assert sm.activity().item() == pytest.approx(1 / K, abs=5 * (1 / K / N) ** 0.5)
    assert not sm.S.any() and not mf.S.any()


@pytest.mark.parametrize("t_abs", [0.0, 1.0])
@pytest.mark.parametrize("dt", [0.5, 0.1])
def test_random_start_R_follows_the_spin_recursion(dt, t_abs):
    """ With the drive held far below threshold nobody fires, so every neuron just ages: at
    each step its R must still be the kernel at its age, which it is only if the start was. """
    sm, mf = _pair("random", dt=dt, t_abs=t_abs, N=2000, E=-100.0, J=0.0, age_max=10.0)
    for _ in range(20):
        assert torch.allclose(sm.R.double(), mf.R[0][sm.n], rtol=1e-5, atol=1e-6)
        sm.update()
    assert not sm.s.any()
