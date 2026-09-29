""" Tests that the mean-field kernels agree with what the spin model realizes. """

import pytest
import torch

import rdme.shared as shrd
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2, 0.05])
def test_refractory_kernel_matches_spin_model(dt):
    """ The spin model decays X by (1 - alpha_ref) per step and resets it to -K_ref on a
    spike, so a neuron of age n carries X = -K_ref (1 - alpha_ref)^n. The mean field reads
    that off the R kernel, so R(n) must reproduce it bin for bin -- which it only does
    if the kernel folds in the bin width deltaT. """
    tau_ref, K_ref, tau_int = 3.0, 2.0, 12.0

    sm = SpinIsingModel(1, 0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, dt=dt, ic="silent")
    mf = RDMIsingModel(0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, dt=dt)

    n = torch.arange(mf.Qm[0], dtype=mf.R[0].dtype)
    eta_spin = -K_ref * (1.0 - sm.a_ref.item()) ** n

    assert torch.allclose(mf.R[0], eta_spin, atol=1e-5)


def test_refractory_kernel_deltaT_scaling():
    """ deltaT rescales the age axis: R(n; deltaT) == R(n*deltaT; 1). """
    Q, K, tau_r, dt = 50, 2.0, 3.0, 0.2
    got = shrd.refractory_kernel(Q, K, tau_r, dt)
    expected = shrd.refractory_kernel(Q, K, tau_r / dt)
    assert torch.allclose(got, expected)
    # default is the dt=1 kernel, unchanged
    assert torch.allclose(shrd.refractory_kernel(Q, K, tau_r),
                          -K * torch.exp(-torch.arange(Q) / tau_r))


def test_stationary_activity_matches_spin_model():
    """ End-to-end: with matching kernels both models settle on the same fixed point. """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    N, J, E, beta, theta, dt = 20000, 1.0, 0.0, 20.0, 0.0, 0.2
    tau_int, tau_ref, K_ref = 12.0, 3.0, 2.0

    sm = SpinIsingModel(N, J / dt, E, beta, theta, tau_int, tau_ref, K_ref, dt=dt, ic="silent")
    mf = RDMIsingModel(J / dt, E, beta, theta, tau_int, tau_ref, K_ref, dt=dt)
    sm.forward(6000)
    mf.forward(6000)

    m_sm = torch.stack([(sm.update(), sm.activity())[1] for _ in range(2000)]).mean()
    m_mf = torch.stack([(mf.update(), mf.activity())[1] for _ in range(2000)]).mean()

    # finite-N sampling noise on m_sm is ~1e-3 relative at N=2e4
    assert m_mf == pytest.approx(m_sm.item(), rel=0.02)
