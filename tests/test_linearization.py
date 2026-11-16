""" The tangent map is the Jacobian of the state map, and at a fixed point the Lyapunov
spectrum is log|eig(J)|/dt. These guard the Lyapunov results. """

import pytest
import torch

import rdme.lyapunov as lyap
from conftest import HAZ_DT, WC
from rdme.mean_field import RDMWilsonCowan


@pytest.mark.parametrize("hazard,dt", HAZ_DT)
def test_tangent_is_the_jacobian_of_the_state_map(hazard, dt):
    net = RDMWilsonCowan(**WC, dt=dt, hazard=hazard)
    net.forward(137)
    x = net.flatten_state(net.p.clone(), net.S.clone())
    k = 3
    V = torch.randn(k, x.numel(), generator=torch.Generator().manual_seed(0))
    want = V @ torch.func.jacfwd(net.state_map)(x).T

    got = V.clone()
    net.step_tangent(got)
    assert torch.allclose(got, want, rtol=0, atol=1e-13), (got - want).abs().max()


def test_lyapunov_spectrum_at_a_fixed_point_is_log_eig_over_dt():
    """ The leading pair is complex-conjugate, so only their sum is basis-independent. """
    dt = 0.2
    net = RDMWilsonCowan(**WC, dt=dt, hazard="poisson_exact")
    net.forward(4000)
    x = net.flatten_state(net.p.clone(), net.S.clone())
    assert torch.allclose(net.state_map(x), x, rtol=0, atol=1e-11), "not a fixed point"
    ev = torch.linalg.eigvals(torch.func.jacfwd(net.state_map)(x)).abs()
    want = (torch.log(ev.clamp_min(1e-300)) / dt).sort(descending=True).values[:2]

    blocks = net.init_tangent(2, generator=torch.Generator().manual_seed(0))
    net.step_tangent(blocks)                 # project out the unphysical directions
    lyap.orthonormalize(blocks)
    out = lyap.benettin(net.step_tangent, blocks, n_steps=3000, dt=dt,
                        cadence=20, n_blocks=10, warmup=2000)
    assert out.partial[1] == pytest.approx(want.sum().item(), abs=1e-4)
