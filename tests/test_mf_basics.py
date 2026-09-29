""" Tests for the basic mean-field functions in mean_field.py: 
    compute_total_input, update_synaptic_term, update_age_distribution, compute_firing_prob. """

import pytest
import torch

import rdme.mean_field as mf


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
    assert torch.allclose(mf.compute_total_input(m, w, E), expected)

    # random case, checked against a plain matmul rather than re-deriving the call
    M = 5
    w = torch.randn(M, M, dtype=torch.float64)
    m = torch.randn(M, dtype=torch.float64)
    E = torch.randn(M, dtype=torch.float64)
    got = mf.compute_total_input(m, w, E)
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

    net = mf.RDMWilsonCowan(K_ref_E=0.0, K_ref_I=0.0, dt=0.5, **kw)
    spin = SpinWilsonCowan(N=1000, K_ref1=0.0, K_ref2=0.0, dt=0.5, **kw)

    assert torch.allclose(net.w.float(), spin.w)

    # and the layout is the one the source-first names imply: I to E is w_IE, not w_EI
    assert net.w[0, 0].item() == pytest.approx(kw["w_EE"])   # E -> E
    assert net.w[0, 1].item() == pytest.approx(kw["w_IE"])   # I -> E, the inhibitory one
    assert net.w[1, 0].item() == pytest.approx(kw["w_EI"])   # E -> I
    assert net.w[1, 1].item() == pytest.approx(kw["w_II"])   # I -> I

    # the inhibitory column must actually inhibit E: raising m_I lowers I_E
    E = torch.zeros(2, dtype=net.w.dtype)
    quiet = mf.compute_total_input(torch.tensor([0.1, 0.0], dtype=net.w.dtype), net.w, E)
    loud  = mf.compute_total_input(torch.tensor([0.1, 0.5], dtype=net.w.dtype), net.w, E)
    assert loud[0] < quiet[0]


def test_update_synaptic_term():
    torch.manual_seed(1)
    Q = 6
    alpha = 0.3
    I = 0.7
    S = torch.rand(Q, dtype=torch.float64)

    S_new = mf.update_synaptic_term(S, alpha, I)

    # bin 0 is always reset -- there is no age-0 field yet
    assert S_new[0] == 0
    # every other bin ages the previous bin's value by one step, injecting the
    # current I weighted by alpha
    assert torch.allclose(S_new[1:], alpha * I + (1 - alpha) * S[:-1])
    # out-of-place: S itself is untouched
    assert not torch.allclose(S, S_new)

    # edge cases: alpha=0 keeps only the aged-in history, alpha=1 forgets it entirely
    S_no_input = mf.update_synaptic_term(S, 0.0, I)
    assert torch.allclose(S_no_input[1:], S[:-1])
    S_no_memory = mf.update_synaptic_term(S, 1.0, I)
    assert torch.allclose(S_no_memory[1:], torch.full_like(S[:-1], I))

    # in-place variant must match the out-of-place one bit-for-bit
    S_inplace = S.clone()
    mf.update_synaptic_term(S_inplace, alpha, I, inplace=True)
    assert torch.equal(S_inplace, S_new)


def test_compute_firing_prob():
    torch.manual_seed(2)
    Q = 5
    S = torch.randn(Q, dtype=torch.float64)
    R = torch.randn(Q, dtype=torch.float64)
    beta, theta = 2.5, 0.3

    expected = torch.sigmoid(beta * (S + R - theta))
    got = mf.compute_firing_prob(S, R, beta, theta)
    assert torch.allclose(got, expected)

    # the no-autograd (in-place-chained) variant must agree numerically...
    got_noag = mf.compute_firing_prob(S.clone(), R.clone(), beta, theta, inplace=True)
    assert torch.allclose(got, got_noag)

    # ...and, despite chaining in-place internally, must not clobber the caller's S/R:
    # the first op is `S + R`, which allocates a fresh tensor, so the in-place chain
    # after that only ever mutates that fresh tensor, never the inputs
    S_before, R_before = S.clone(), R.clone()
    mf.compute_firing_prob(S, R, beta, theta, inplace=True)
    assert torch.equal(S, S_before)
    assert torch.equal(R, R_before)


def test_update_age_distribution():
    torch.manual_seed(3)
    Q = 6
    P = torch.rand(Q, dtype=torch.float64)
    P /= P.sum()
    fprobs = torch.rand(Q, dtype=torch.float64) * 0.8  # stay away from 1

    P_new = mf.update_age_distribution(P, fprobs)

    # renewal update stays a valid distribution
    assert torch.allclose(P_new.sum(), torch.tensor(1.0, dtype=torch.float64))
    # births into bin 0 are exactly the firing rate
    assert torch.allclose(P_new[0], mf.compute_firing_rate(P, fprobs))
    # interior bins age by one step, decayed by survival
    assert torch.allclose(P_new[1:-1], P[:-2] * (1 - fprobs[:-2]))
    # the boundary bin absorbs whatever remains, so P_new stays normalized by construction
    assert torch.allclose(P_new[-1], 1 - P_new[:-1].sum())

    mf.check_age_normalization(P_new)  # should not raise
    with pytest.raises(ValueError):
        mf.check_age_normalization(torch.tensor([0.5, 0.6]))

    # in-place variant must match the out-of-place one to floating-point rounding: both
    # compute the boundary bin as 1 minus the same two quantities, but grouped differently
    # (1 - (frate + interior.sum()) vs (1 - frate) - interior.sum()), so they are not
    # guaranteed bit-exact
    P_inplace = P.clone()
    mf.update_age_distribution(P_inplace, fprobs, inplace=True)
    assert torch.allclose(P_inplace, P_new)


@pytest.mark.parametrize("shape", [(9,), (4, 9)])
def test_inplace_branches_match_out_of_place(shape):
    """ Every function carrying an `inplace` flag must give the same answer either way, and
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

    P_ref = mf.update_age_distribution(P.clone(), fprobs)
    P_ip  = P.clone(); mf.update_age_distribution(P_ip, fprobs, inplace=True)
    assert torch.allclose(P_ref, P_ip, atol=1e-12)
    assert torch.allclose(P_ip.sum(dim=-1), torch.ones(shape[:-1], dtype=torch.float64), atol=1e-12)

    S = torch.randn(*shape, dtype=torch.float64)
    S_ref = mf.update_synaptic_term(S.clone(), alpha, I)
    S_ip  = S.clone(); mf.update_synaptic_term(S_ip, alpha, I, inplace=True)
    assert torch.equal(S_ref, S_ip)

    R = torch.randn(*shape, dtype=torch.float64)
    beta  = torch.full((shape[0], 1), 2.5, dtype=torch.float64) if batched else 2.5
    theta = torch.full((shape[0], 1), 0.4, dtype=torch.float64) if batched else 0.4
    phi_ref = mf.compute_firing_prob(S, R, beta, theta)
    phi_ip  = mf.compute_firing_prob(S.clone(), R.clone(), beta, theta, inplace=True)
    assert torch.allclose(phi_ref, phi_ip, atol=1e-12)


@pytest.mark.parametrize("shape", [(9,), (4, 9)])
def test_compute_firing_rate_reduces_over_the_age_axis(shape):
    """ 1-D goes through torch.dot, higher rank through the explicit reduction; both must
    reduce the LAST axis and agree. """
    torch.manual_seed(1)
    P = torch.rand(*shape, dtype=torch.float64)
    fprobs = torch.rand(*shape, dtype=torch.float64)
    assert torch.allclose(mf.compute_firing_rate(P, fprobs), (P * fprobs).sum(dim=-1), atol=1e-12)
