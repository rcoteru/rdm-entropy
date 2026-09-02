""" Tests for the basic mean-field functions in mean_field.py: 
    compute_population_input, update_integration_variable, update_age_distribution, compute_firing_prob. """

import pytest
import torch

import rdme.mean_field as mf


def test_population_input():
    torch.manual_seed(0)
    # small hand-checkable case
    w = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
    m = torch.tensor([0.5, -0.25])
    I = torch.tensor([0.1, 0.2])
    expected = torch.tensor([
        1.0 * 0.5 + 3.0 * (-0.25) + 0.1,
        2.0 * 0.5 + 4.0 * (-0.25) + 0.2,
    ])
    assert torch.allclose(mf.compute_population_input(m, w, I), expected)

    # random case, checked against a plain matmul rather than re-deriving the call
    M = 5
    w = torch.randn(M, M, dtype=torch.float64)
    m = torch.randn(M, dtype=torch.float64)
    I = torch.randn(M, dtype=torch.float64)
    got = mf.compute_population_input(m, w, I)
    assert torch.allclose(got, w.T @ m + I)


def test_update_integration_variable():
    torch.manual_seed(1)
    Q = 6
    alpha = 0.3
    pop_input = 0.7
    y = torch.rand(Q, dtype=torch.float64)

    y_new = mf.update_integration_variable(y, alpha, pop_input)

    # bin 0 is always reset -- there is no age-0 field yet
    assert y_new[0] == 0
    # every other bin ages the previous bin's value by one step, injecting the
    # current pop_input weighted by alpha
    assert torch.allclose(y_new[1:], alpha * pop_input + (1 - alpha) * y[:-1])
    # out-of-place: y itself is untouched
    assert not torch.allclose(y, y_new)

    # edge cases: alpha=0 keeps only the aged-in history, alpha=1 forgets it entirely
    y_no_input = mf.update_integration_variable(y, 0.0, pop_input)
    assert torch.allclose(y_no_input[1:], y[:-1])
    y_no_memory = mf.update_integration_variable(y, 1.0, pop_input)
    assert torch.allclose(y_no_memory[1:], torch.full_like(y[:-1], pop_input))

    # in-place variant must match the out-of-place one bit-for-bit
    y_inplace = y.clone()
    mf.update_integration_variable_inplace(y_inplace, alpha, pop_input)
    assert torch.equal(y_inplace, y_new)


def test_compute_firing_prob():
    torch.manual_seed(2)
    Q = 5
    y = torch.randn(Q, dtype=torch.float64)
    eta = torch.randn(Q, dtype=torch.float64)
    beta, theta = 2.5, 0.3

    expected = torch.sigmoid(beta * (y + eta - theta))
    got = mf.compute_firing_prob(y, eta, beta, theta)
    assert torch.allclose(got, expected)

    # the no-autograd (in-place-chained) variant must agree numerically...
    got_noag = mf.compute_firing_prob_noautograd(y.clone(), eta.clone(), beta, theta)
    assert torch.allclose(got, got_noag)

    # ...and, despite chaining in-place internally, must not clobber the caller's y/eta:
    # the first op is `y + eta`, which allocates a fresh tensor, so the in-place chain
    # after that only ever mutates that fresh tensor, never the inputs
    y_before, eta_before = y.clone(), eta.clone()
    mf.compute_firing_prob_noautograd(y, eta, beta, theta)
    assert torch.equal(y, y_before)
    assert torch.equal(eta, eta_before)


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
    mf.update_age_distribution_inplace(P_inplace, fprobs)
    assert torch.allclose(P_inplace, P_new)
