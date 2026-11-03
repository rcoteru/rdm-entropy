""" Tests for rdme.helpers -- the script-side utilities.

The EPR memory model is the part worth pinning. `epr_peak_bytes` carries coefficients that
were MEASURED against the allocator rather than derived, so what these tests protect is the
shape of the formula and the planner's contract (never exceed the budget, never return a
chunk larger than the tile). The coefficients themselves are checked against a live run in
test_epr_peak_matches_the_allocator, which needs a GPU and skips without one. """

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
                            tau_int=15.0, tau_ref=15.0, K_ref=15.0, dt0=1.0, dt=1.0,
                            hazard="async", Qm=[Q], device="cuda")
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
