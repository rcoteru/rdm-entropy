""" Tests for the per-step primitives in rdme.rdm: compute_total_input,
update_synaptic_term, compute_firing_rate, update_age_distribution, check_age_normalization.

These are the kernel-independent half of the model -- none of them evaluates a hazard, which
is why one copy serves every member of the family. The hazards themselves are tested in
test_srm.py. """

import pytest
import torch

import rdme.hazards as hzd
import rdme.rdm as rdm
from rdme.rdm_single import RDMWilsonCowan


def test_population_input():
    torch.manual_seed(0)
    # small hand-checkable case
    w = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
    m = torch.tensor([0.5, -0.25])
    E = torch.tensor([0.1, 0.2])
    # row = target, column = source: I^a = sum_b w[a, b] m^b + E^a
    expected = torch.tensor([
        1.0 * 0.5 + 2.0 * (-0.25) + 0.1,
        3.0 * 0.5 + 4.0 * (-0.25) + 0.2,
    ])
    assert torch.allclose(rdm.compute_total_input(m, w, E), expected)

    # random case, checked against a plain matmul rather than re-deriving the call
    M = 5
    w = torch.randn(M, M, dtype=torch.float64)
    m = torch.randn(M, dtype=torch.float64)
    E = torch.randn(M, dtype=torch.float64)
    got = rdm.compute_total_input(m, w, E)
    assert torch.allclose(got, w @ m + E)


def test_wilson_cowan_weight_orientation_matches_the_spin_model():
    """ w_XY is source-first (X -> Y) in every constructor's signature, while the matrix the
    models store is target-first (row = target), so the constructors transpose. Both model
    families must land on the same layout: an asymmetric off-diagonal is the only thing that
    can catch a slip, which is why w_EI and w_IE are deliberately different here. """
    from rdme.spin_model import SpinWilsonCowan

    kw = dict(E_ratio=0.8, w_EE=5.0, w_EI=3.0, w_IE=-6.0, w_II=-0.5,
              E_exc=1.0, E_inh=0.5, beta_E=10.0, beta_I=10.0,
              theta_E=1.0, theta_I=1.0, tau_int_E=10.0, tau_int_I=5.0,
              tau_ref_E=3.0, tau_ref_I=3.0)

    net = RDMWilsonCowan(K_ref_E=0.0, K_ref_I=0.0, dt0=0.5, hazard="async", **kw)
    spin = SpinWilsonCowan(N=1000, K_ref1=0.0, K_ref2=0.0, dt0=0.5, **kw)

    assert torch.allclose(net.w.float(), spin.w)

    # and the layout is the one the source-first names imply: I to E is w_IE, not w_EI
    assert net.w[0, 0].item() == pytest.approx(kw["w_EE"])   # E -> E
    assert net.w[0, 1].item() == pytest.approx(kw["w_IE"])   # I -> E, the inhibitory one
    assert net.w[1, 0].item() == pytest.approx(kw["w_EI"])   # E -> I
    assert net.w[1, 1].item() == pytest.approx(kw["w_II"])   # I -> I

    # the inhibitory column must actually inhibit E: raising m_I lowers I_E
    E = torch.zeros(2, dtype=net.w.dtype)
    quiet = rdm.compute_total_input(torch.tensor([0.1, 0.0], dtype=net.w.dtype), net.w, E)
    loud  = rdm.compute_total_input(torch.tensor([0.1, 0.5], dtype=net.w.dtype), net.w, E)
    assert loud[0] < quiet[0]


def test_update_synaptic_term():
    torch.manual_seed(1)
    Q = 6
    alpha = 0.3
    I = 0.7
    S = torch.rand(Q, dtype=torch.float64)

    S_new = rdm.update_synaptic_term(S, alpha, I)

    # bin 0 is always reset -- there is no age-0 field yet
    assert S_new[0] == 0
    # every other bin ages the previous bin's value by one step, injecting the
    # current I weighted by alpha
    assert torch.allclose(S_new[1:], alpha * I + (1 - alpha) * S[:-1])
    # out-of-place: S itself is untouched
    assert not torch.allclose(S, S_new)

    # edge cases: alpha=0 keeps only the aged-in history, alpha=1 forgets it entirely
    S_no_input = rdm.update_synaptic_term(S, 0.0, I)
    assert torch.allclose(S_no_input[1:], S[:-1])
    S_no_memory = rdm.update_synaptic_term(S, 1.0, I)
    assert torch.allclose(S_no_memory[1:], torch.full_like(S[:-1], I))

    # in-place variant must match the out-of-place one bit-for-bit
    S_inplace = S.clone()
    rdm.update_synaptic_term(S_inplace, alpha, I, inplace=True)
    assert torch.equal(S_inplace, S_new)


def test_async_hazard_at_the_quantum_is_the_sigmoid_of_the_field():
    """ The kernel the old compute_firing_prob hard-coded, now one member of the family.
    Its derivative and its relatives are tested in test_srm.py; this just pins the form. """
    torch.manual_seed(2)
    Q = 5
    S = torch.randn(Q, dtype=torch.float64)
    R = torch.randn(Q, dtype=torch.float64)
    beta, theta = 2.5, 0.3
    got = hzd.async_hazard(S + R - theta, beta, lam0=1.0, dt=1.0)
    assert torch.allclose(got, torch.sigmoid(beta * (S + R - theta)))
    # and it does not touch the caller's tensors
    S_before, R_before = S.clone(), R.clone()
    hzd.async_hazard(S + R - theta, beta, 1.0, 1.0)
    assert torch.equal(S, S_before) and torch.equal(R, R_before)


def test_update_age_distribution():
    torch.manual_seed(3)
    Q = 6
    P = torch.rand(Q, dtype=torch.float64)
    P /= P.sum()
    fprobs = torch.rand(Q, dtype=torch.float64) * 0.8  # stay away from 1

    P_new = rdm.update_age_distribution(P, fprobs)

    # renewal update stays a valid distribution
    assert torch.allclose(P_new.sum(), torch.tensor(1.0, dtype=torch.float64))
    # births into bin 0 are exactly the firing rate
    assert torch.allclose(P_new[0], rdm.compute_firing_rate(P, fprobs))
    # interior bins age by one step, decayed by survival
    assert torch.allclose(P_new[1:-1], P[:-2] * (1 - fprobs[:-2]))
    # the boundary bin absorbs whatever remains, so P_new stays normalized by construction
    assert torch.allclose(P_new[-1], 1 - P_new[:-1].sum())

    rdm.check_age_normalization(P_new)  # should not raise
    with pytest.raises(ValueError):
        rdm.check_age_normalization(torch.tensor([0.5, 0.6]))

    # in-place variant must match the out-of-place one to floating-point rounding: both
    # compute the boundary bin as 1 minus the same two quantities, but grouped differently
    # (1 - (frate + interior.sum()) vs (1 - frate) - interior.sum()), so they are not
    # guaranteed bit-exact
    P_inplace = P.clone()
    rdm.update_age_distribution(P_inplace, fprobs, inplace=True)
    assert torch.allclose(P_inplace, P_new)


@pytest.mark.parametrize("shape", [(9,), (4, 9)])
def test_inplace_branches_match_out_of_place(shape):
    """ Both functions carrying an `inplace` flag must give the same answer either way, and
    must do so for BATCHED input too -- the in-place branches were once written with
    first-axis indexing (P[0], S[:-1]), which silently addresses the batch axis instead of
    the age axis once P is (B, Q). """
    torch.manual_seed(0)
    batched = len(shape) == 2
    alpha = torch.full((shape[0], 1), 0.3, dtype=torch.float64) if batched else 0.3
    I     = torch.full((shape[0], 1), 0.7, dtype=torch.float64) if batched else 0.7

    P = torch.rand(*shape, dtype=torch.float64)
    P /= P.sum(dim=-1, keepdim=True)
    fprobs = torch.rand(*shape, dtype=torch.float64)

    P_ref = rdm.update_age_distribution(P.clone(), fprobs)
    P_ip  = P.clone(); rdm.update_age_distribution(P_ip, fprobs, inplace=True)
    assert torch.allclose(P_ref, P_ip, atol=1e-12)
    assert torch.allclose(P_ip.sum(dim=-1), torch.ones(shape[:-1], dtype=torch.float64), atol=1e-12)

    S = torch.randn(*shape, dtype=torch.float64)
    S_ref = rdm.update_synaptic_term(S.clone(), alpha, I)
    S_ip  = S.clone(); rdm.update_synaptic_term(S_ip, alpha, I, inplace=True)
    assert torch.equal(S_ref, S_ip)

    # (the hazards grew their own in-place branches later, for the entropy production rather
    # than for the step: see test_epr.test_hazard_inplace_branches_are_bitwise_identical)


@pytest.mark.parametrize("shape", [(9,), (4, 9)])
def test_compute_firing_rate_reduces_over_the_age_axis(shape):
    """ 1-D goes through torch.dot, higher rank through the explicit reduction; both must
    reduce the LAST axis and agree. """
    torch.manual_seed(1)
    P = torch.rand(*shape, dtype=torch.float64)
    fprobs = torch.rand(*shape, dtype=torch.float64)
    assert torch.allclose(rdm.compute_firing_rate(P, fprobs), (P * fprobs).sum(dim=-1), atol=1e-12)


# The weight convention
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_input_is_built_from_the_rate_in_all_three_models():
    """ The convention of rdm.compute_total_input: I = w @ r + E with r the RATE in 1/ms, not
    the per-bin overlap m = r*dt. It has to hold in all three classes, because a coupling that
    meant different things in the mean field and the spin model would make every comparison
    between them wrong by a factor of dt. """
    from rdme.rdm_single import RDMIsingModel
    from rdme.rdm_batch import RDMIsingModelBatch
    from rdme.spin_model import SpinIsingModel

    J, E = 4.0, 0.5
    base = dict(beta=3.0, theta=0.4, tau_int=5.0, tau_ref=4.0, K_ref=0.0)

    for dt in (1.0, 0.5):
        s = RDMIsingModel(J=J, E=E, **base, dt0=1.0, dt=dt, hazard="async",
                          t_abs=0.0, Qm=[60])
        s.forward(200)
        assert s.total_input()[0].item() == pytest.approx(J * s.rates()[0].item() + E, abs=1e-12)

    b = RDMIsingModelBatch(J=torch.tensor([J]), E=E, **base, dt0=1.0, dt=0.5,
                           hazard="async", t_abs=0.0, Qm=[60])
    b.forward(200)
    assert b.total_input()[0, 0].item() == pytest.approx(J * b.rates()[0, 0].item() + E, abs=1e-12)

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        m = SpinIsingModel(N=5000, J=J, E=E, **base, dt0=1.0, dt=0.5,
                           hazard="async", ic="silent")
        m.forward(600)
        assert m.total_input()[0].item() == pytest.approx(J * m.rates()[0].item() + E, rel=1e-5)
    finally:
        torch.set_default_dtype(prev)


def test_a_coupling_does_not_have_to_be_rescaled_when_dt_changes():
    """ The invariant, measured through the dynamics rather than inferred from the rates.

    The coupling contribution to the input is I - E. Divided by the RATE it is w, at every dt;
    divided by the per-bin overlap m = r*dt it would be w*dt and would halve whenever dt did.
    So (I - E)/r is the quantity that has to come back equal to J, and it is what a caller
    relies on when it moves a script from dt = 0.5 to dt = 0.1 without touching J.

    Checking the rates instead is NOT enough: a first version of this test asserted only that
    the rates stay within 50% of each other across dt, and it PASSED with the input rebuilt
    from m -- the regressed rates happened to land inside the band. """
    from rdme.rdm_single import RDMIsingModel
    J, E = 4.0, 0.5
    base = dict(beta=3.0, theta=0.4, tau_int=5.0, tau_ref=4.0, K_ref=0.0)
    for dt in (1.0, 0.5, 0.25, 0.125):
        m = RDMIsingModel(J=J, E=E, **base, dt0=1.0, dt=dt, hazard="async",
                          t_abs=0.0, Qm=[int(60 / dt)])
        m.forward(int(2000 / dt))
        r = m.rates()[0].item()
        assert r > 1e-3, f"dt={dt}: silent, the test proves nothing"
        effective_J = (m.total_input()[0].item() - E) / r
        assert effective_J == pytest.approx(J, rel=1e-12), (
            f"dt={dt}: effective coupling {effective_J} != {J}; the input is being built "
            "from the per-bin overlap rather than the rate")
