""" Entropy production: zero at a fixed point, the sliding observer equal to the full-joint
definition, and the Poisson reference carrying the dt divergence. """

import math

import pytest
import torch

import rdme.presets as pre
from conftest import HAZ_DT, SETTLED, WC
from rdme.epr import anchor_reverse_synaptic_term, poisson_reference_entropy
from rdme.mean_field import RDMIsingModel, RDMWilsonCowan, RDMWilsonCowanPoints


@pytest.mark.parametrize("hazard,dt", HAZ_DT)
def test_sigma_vanishes_at_a_fixed_point(hazard, dt):
    """ A stationary state is time-reversal symmetric. The grid is sized tight enough that the
    lumped last bin carries no mass: at truncation error 1e-2 it leaves sigma ~ 1e-7. """
    net = RDMWilsonCowan(**SETTLED, dt=dt, hazard=hazard, tol=1e-9)
    for r in net.fixed_points():
        net.set_state(r)
        assert float(net.truncation_error()) < 1e-8
        t = net.entropy_trajectory(50, pb=False)
        assert float(t["H_fwd"].min()) > 1e-3, "sigma = 0 must not be 0 - 0"
        assert float(t["sigma"].abs().max()) < 1e-14


def test_reverse_field_is_anchored_at_t_plus_one():
    """ The reverse field both EPR paths share, against its closed form:
    S^dagger_{t+1}(n) = sum_{tau=1..n} alpha (1-alpha)^{tau-1} I_{t+1+tau}, with the censoring
    lag clamped to the last resolvable one. Mis-anchoring moves cycle sigma by ~4%, which
    the joint comparison cannot see and the spin comparison cannot resolve. """
    K, alpha = 7, 0.45
    I = torch.randn(K, generator=torch.Generator().manual_seed(11))   # I[j] = input at t + j
    S = anchor_reverse_synaptic_term(I, alpha)
    assert S[0] == 0
    for n in range(1, K - 1):
        want = sum(alpha * (1 - alpha) ** (tau - 1) * I[1 + tau] for tau in range(1, n + 1))
        assert S[n].item() == pytest.approx(float(want), abs=1e-15)
    assert S[K - 1] == S[K - 2]


@pytest.mark.parametrize("hazard,dt", HAZ_DT)
def test_sliding_epr_matches_the_joint_definition(hazard, dt):
    """ On a moving state, the O(Q) observer equals the EPR read off the explicit joint. """
    a = RDMWilsonCowan(**WC, dt=dt, hazard=hazard).entropy_trajectory(40, pb=False)
    b = RDMWilsonCowan(**WC, dt=dt, hazard=hazard).entropy_trajectory(40, pb=False,
                                                                     method="joint")
    assert float(a["sigma"].abs().max()) > 1e-8, "the cell must be off its fixed point"
    for k in ("m", "sigma", "H_fwd", "H_rev"):
        assert (a[k] - b[k]).abs().max().item() < 1e-14, k


@pytest.mark.parametrize("hazard,dt", HAZ_DT)
def test_batch_matches_single_systems(hazard, dt):
    """ The sweeps run batched: each batch element is the single system with its parameters. """
    W_EE = [1.5, 2.0, 2.5]
    b = RDMWilsonCowanPoints(**{**WC, "w_EE": torch.tensor(W_EE)}, dt=dt, hazard=hazard)
    tb = b.entropy_trajectory(30, pb=False)
    for i, w in enumerate(W_EE):
        ts = RDMWilsonCowan(**{**WC, "w_EE": w}, dt=dt, hazard=hazard).entropy_trajectory(
            30, pb=False)
        for k in ("m", "sigma", "H_fwd", "H_rev"):
            assert (tb[k][i] - ts[k]).abs().max().item() < 1e-14, (k, i)


def test_saturated_sigma_does_not_depend_on_the_dtype():
    """ On S1's bursting cell the reverse hazard saturates, and H_rev is made of ln(1 - Phi)
    there. Read off Phi, that log is a floor set by a dtype-dependent clamp, and float32 gave a
    sigma 40% below float64's (both far below the spins'); off the kernel's exact branch the
    two agree. """
    P = pre.params("S1", hazard="poisson_exact", dt=0.5)
    sig = []
    for dtype in (torch.float32, torch.float64):
        net = RDMIsingModel(60.0, **P, dtype=dtype)
        net.forward(3000)
        sig.append(float(net.entropy_trajectory(400, pb=False)["sigma_tot"].double().mean()))
    assert sig[1] > 100, "the cell must be saturating, where the clamp used to set sigma"
    assert sig[0] == pytest.approx(sig[1], rel=1e-3), sig


def test_the_poisson_reference_carries_the_dt_divergence():
    """ Raw H_fwd/dt grows by r ln 2 per halving of dt; referenced to the Poisson process of
    rate nu0 it converges; sigma, from which the reference cancels, is untouched. """
    raw, ref, rate = [], [], []
    for dt in (0.1, 0.05, 0.025):
        net = RDMIsingModel(J=2.0, E=0.5, beta=4.0, theta=0.5, tau_int=8.0, tau_ref=6.0,
                            K_ref=3.0, dt=dt, nu0=1.0, t_abs=1.0, hazard="poisson_exact")
        net.forward(int(600 / dt))
        t = net.entropy_trajectory(int(40 / dt), pb=False)
        m = t["m"][:, 0]
        off = poisson_reference_entropy(m, net.nu0, dt)
        raw.append(float(t["H_fwd"][:, 0].mean()) / dt)
        ref.append(float((t["H_fwd"][:, 0] - off).mean()) / dt)
        rate.append(float(m.mean()) / dt)
    for i in (0, 1):
        assert raw[i + 1] - raw[i] == pytest.approx(rate[i] * math.log(2.0), rel=0.05), raw
    assert abs(ref[2] - ref[1]) < 0.6 * abs(ref[1] - ref[0]), ref
