""" Tests for rdme.helpers -- the script-side utilities.

The EPR memory model is the part worth pinning. `epr_peak_bytes` carries coefficients that
were MEASURED against the allocator rather than derived, so what these tests protect is the
shape of the formula and the planner's contract (never exceed the budget, never return a
chunk larger than the tile). The coefficients themselves are checked against a live run in
test_epr_peak_matches_the_allocator, which needs a GPU and skips without one. """

import math

import pytest
import torch

import rdme.helpers as hlp


# epr_peak_bytes / epr_plan
# ~~~~~~~~~~~~~~~~~~~~~~~~~

def test_epr_peak_is_quadratic_in_q_and_linear_in_b():
    """ The whole point of the helper: Q costs four times as much when it doubles. """
    assert hlp.epr_peak_bytes(100, 200) == 4 * hlp.epr_peak_bytes(100, 100)
    assert hlp.epr_peak_bytes(200, 100) == 2 * hlp.epr_peak_bytes(100, 100)


def test_epr_peak_out_of_place_costs_one_extra_ring():
    """ inplace=False pins the (B, Q, Q) gather phi_end views for the whole run. """
    B, Q = 64, 50
    ring = B * Q * Q * 8
    assert (hlp.epr_peak_bytes(B, Q, inplace=False)
            - hlp.epr_peak_bytes(B, Q, inplace=True)) == ring


def test_epr_peak_chunk_caps_the_transient_not_the_ring():
    B, Q = 1000, 40
    full, small = hlp.epr_peak_bytes(B, Q), hlp.epr_peak_bytes(B, Q, 1)
    assert small < full
    assert small > B * Q * Q * 8          # the ring survives any chunk
    # a chunk at or above B is the same as no chunk at all
    assert hlp.epr_peak_bytes(B, Q, B) == full == hlp.epr_peak_bytes(B, Q, 10 * B)


@pytest.mark.parametrize("B,Q,budget_gib", [(300, 213, 8.0), (6561, 213, 4.0),
                                            (6561, 550, 4.0), (300, 2000, 0.5)])
def test_epr_plan_stays_inside_the_budget(B, Q, budget_gib):
    budget = int(budget_gib * 2**30)
    B_tile, chunk = hlp.epr_plan(B, Q, device="cpu", budget=budget, verbose=False)
    assert 1 <= B_tile <= B
    assert 1 <= chunk <= B_tile
    if B_tile > 1:                        # a tile of 1 is the "does not fit at all" answer
        assert hlp.epr_peak_bytes(B_tile, Q, chunk) <= budget


def test_epr_plan_fits_the_whole_scan_when_it_can():
    B, Q = 300, 100
    whole = hlp.epr_peak_bytes(B, Q, B)
    B_tile, _ = hlp.epr_plan(B, Q, device="cpu", budget=10 * whole, verbose=False)
    assert B_tile == B


def test_epr_plan_tiles_when_it_cannot():
    B, Q = 6561, 550
    B_tile, _ = hlp.epr_plan(B, Q, device="cpu", budget=2**31, verbose=False)
    assert B_tile < B


def test_device_budget_is_a_positive_fraction_of_free_memory():
    half = hlp.device_budget("cpu", 0.5)
    tenth = hlp.device_budget("cpu", 0.1)
    assert half > 0 and tenth > 0
    assert tenth < half


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_epr_peak_matches_the_allocator():
    """ The coefficients against a real run. They were fitted on this path, so a drift here
    means the allocation pattern of entropy_trajectory changed and the fit has to be redone --
    which is a correctness problem for every script that sizes a tile with it. """
    from rdme.rdm_batch import RDMIsingModelBatch
    torch.set_default_dtype(torch.float64)
    # a realistic (B, Q): the estimate drops the model's own O(B * Q) state, which is a
    # noticeable share of a shallow grid and negligible on the deep ones the scans use
    B, Q, chunk = 300, 250, 150
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mf = RDMIsingModelBatch(J=torch.linspace(5.0, 25.0, B), E=0.0, beta=8.0, theta=0.5,
                            tau_int=15.0, tau_ref=15.0, K_ref=15.0, dt=1.0, nu0=1.0,
                            hazard="poisson_exact", Qm=[Q], device="cuda")
    mf.forward(30, pb=False)
    mf.entropy_trajectory(3, pb=False, chunk=chunk, inplace=True)
    real = torch.cuda.max_memory_allocated()
    pred = hlp.epr_peak_bytes(B, Q, chunk, inplace=True)
    assert pred <= real <= 1.10 * pred, f"predicted {pred}, measured {real}"


# n_steps
# ~~~~~~~

def test_n_steps_is_the_duration_divided_by_the_step():
    assert hlp.n_steps(10000.0, 0.5) == 20000
    assert hlp.n_steps(10000.0, 1.0) == 10000


def test_n_steps_keeps_the_window_when_dt_is_refined():
    """ The point of specifying runs in ms: halving dt doubles the steps and measures the same
    physical window, where a fixed step count would have halved it. """
    ms = 6000.0
    assert hlp.n_steps(ms, 0.25) == 2 * hlp.n_steps(ms, 0.5)


def test_n_steps_rounds_to_the_nearest_step():
    assert hlp.n_steps(1.0, 0.3) == 3          # 3.33 steps, so at most half a step is lost
    assert hlp.n_steps(1.1, 0.3) == 4


@pytest.mark.parametrize("ms, dt", [(0.1, 0.3), (0.0, 1.0), (-5.0, 1.0), (5.0, 0.0)])
def test_n_steps_rejects_a_window_that_is_not_a_window(ms, dt):
    with pytest.raises(ValueError):
        hlp.n_steps(ms, dt)


# The dt -> 0 limit: orders / richardson / converged_at
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# These three are CLAUDE.md's reporting rule in code, and every study in the current brief
# passes its numbers through them. The REFUSAL path is the load-bearing one -- it is what
# refused the cycle amplitude, which has no dt -> 0 limit -- so it is tested directly rather
# than through a script. Every oracle here is a closed form built in the test: a sequence
# x_k = L + a * h_k^p with h halving is exactly first/second-order convergent by construction,
# so the order and the extrapolated limit are both known before the call.

def _converging(limit, amp, p, n=5):
    """ x_k = limit + amp * (1/2^k)^p: a sequence of exact order p, converging from above. """
    return torch.tensor([limit + amp * (0.5 ** k) ** p for k in range(n)])


@pytest.mark.parametrize("p", [1.0, 2.0])
def test_orders_reads_the_order_it_was_built_with(p):
    o = hlp.orders(_converging(3.0, 0.5, p))
    assert torch.allclose(o, torch.full_like(o, p), atol=1e-9)


def test_orders_is_one_shorter_than_the_differences():
    """ n values -> n-1 differences -> n-2 orders. A length that drifts here silently
    misaligns every table that prints an order beside a dt. """
    assert len(hlp.orders(_converging(1.0, 1.0, 1.0, n=6))) == 4


def test_richardson_recovers_a_known_limit():
    """ First order, so the (x_n - x_{n-1})/(2^p - 1) correction is exactly the remaining
    error and the limit comes back to machine precision. """
    lim, p = hlp.richardson(_converging(2.5, 1.0, 1.0))
    assert p == pytest.approx(1.0, abs=1e-6)
    assert lim == pytest.approx(2.5, abs=1e-12)


def test_richardson_recovers_a_second_order_limit():
    lim, p = hlp.richardson(_converging(-0.75, 4.0, 2.0))
    assert p == pytest.approx(2.0, abs=1e-6)
    assert lim == pytest.approx(-0.75, abs=1e-12)


def test_richardson_extrapolates_a_bare_pair_at_first_order():
    """ A pair has no order to read and no sign change to see, so it is first-order
    unconditionally -- which is why the standing rule asks for the order from a third step. """
    lim, p = hlp.richardson(torch.tensor([1.2, 1.1]))
    assert p == 1.0
    assert lim == pytest.approx(1.1 + (1.1 - 1.2) / (2**1 - 1))


def test_richardson_needs_at_least_a_pair():
    with pytest.raises(ValueError):
        hlp.richardson(torch.tensor([1.0]))


def test_richardson_refuses_a_sign_change():
    """ The cycle-amplitude / poisson-at-threshold case: a sequence that falls, turns and
    climbs has no error estimate in its last difference. nan is the required answer -- a
    four-digit limit read off a turning point is the failure this guard exists for. """
    vals = torch.tensor([1.00, 0.90, 0.85, 0.88, 0.93])
    with pytest.warns(RuntimeWarning, match="differences change sign"):
        lim, p = hlp.richardson(vals)
    assert math.isnan(lim) and math.isnan(p)


def test_richardson_refuses_an_order_out_of_range():
    """ Same disease caught one step earlier: a monotone sequence whose differences collapse
    far faster than any order the method can have. Here they fall by 2^8 per step, so the
    observed order is 8 -- which measures something other than the discretization error. """
    vals = torch.tensor([1.0, 0.5, 0.5 - 2.0**-9, 0.5 - 2.0**-9 - 2.0**-18])
    with pytest.warns(RuntimeWarning, match="observed order out of range"):
        lim, p = hlp.richardson(vals)
    assert math.isnan(lim) and math.isnan(p)


def test_richardson_refuses_a_negative_order():
    """ Differences GROWING as dt is refined: monotone, so no sign change, but diverging.
    Caught by the p < 0 arm of the same guard. """
    vals = torch.tensor([1.0, 1.1, 1.3, 1.7, 2.5])
    with pytest.warns(RuntimeWarning, match="observed order out of range"):
        lim, p = hlp.richardson(vals)
    assert math.isnan(lim) and math.isnan(p)


def test_richardson_falls_back_to_first_order_when_the_order_is_unreadable():
    """ Near the limit the differences are of nearly equal numbers and the order read off them
    is noise. An order outside [0.2, 4] but inside the refusal band falls back to p = 1 rather
    than refusing -- the extrapolation is a small correction exactly there. """
    vals = torch.tensor([1.0, 0.5, 0.25, 0.01])     # differences -0.5, -0.25, -0.24
    lim, p = hlp.richardson(vals)
    assert p == 1.0
    assert not math.isnan(lim)


# converged_at
# ~~~~~~~~~~~~

def test_converged_at_returns_the_coarsest_step_inside_the_threshold():
    dts  = [1.0, 0.5, 0.25, 0.125]
    vals = torch.tensor([1.5, 1.1, 1.01, 1.001])          # rel err 0.5, 0.1, 0.01, 0.001
    assert hlp.converged_at(dts, vals, 1.0, 0.05) == 0.25
    assert hlp.converged_at(dts, vals, 1.0, 0.5) == 0.5   # strict: 0.5 is not < 0.5
    assert hlp.converged_at(dts, vals, 1.0, 0.2) == 0.5


def test_converged_at_requires_it_to_stay_converged():
    """ 'And stays': a curve that dips under the threshold and comes back out is not
    converged at the dip. Only the unbroken tail counts. """
    dts  = [1.0, 0.5, 0.25, 0.125]
    vals = torch.tensor([1.5, 1.001, 1.2, 1.001])         # inside at dt = 0.5, out at 0.25
    assert hlp.converged_at(dts, vals, 1.0, 0.05) == 0.125


def test_converged_at_is_nan_when_nothing_converges():
    dts = [1.0, 0.5, 0.25]
    assert math.isnan(hlp.converged_at(dts, torch.tensor([3.0, 2.0, 1.5]), 1.0, 0.01))
