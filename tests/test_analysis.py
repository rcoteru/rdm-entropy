""" The reporting helpers on signals with known answers: a limit cycle, a decaying oscillation
and a constant. """

import math

import pytest
import torch

import rdme.analysis as an

DT, T, PERIOD = 0.1, 3000.0, 25.0          # ms


def _traj(r_hz: torch.Tensor) -> dict:
    """ An entropy_trajectory dict whose rate is r_hz (Hz) and sigma is zero. """
    m = r_hz * DT * 1e-3
    return {"m_tot": m, "sigma_tot": torch.zeros_like(m)}


def _sine(decay_per_ms: float) -> torch.Tensor:
    t = torch.arange(int(T / DT), dtype=torch.float64) * DT
    return 40.0 + 10.0 * torch.exp(-decay_per_ms * t) * torch.sin(2 * math.pi * t / PERIOD)


def test_limit_cycle_counts_its_periods():
    o = an.rate_and_epr(_traj(_sine(0.0)), DT, 4.0, cycles=True)
    assert o["state"] == "cycle"
    assert o["amp_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert o["n_cycles"] == int(T / PERIOD) - 1
    assert o["r_mean"] == pytest.approx(40.0, rel=1e-6)
    assert (o["r_min"], o["r_max"]) == (pytest.approx(30.0, rel=1e-4), pytest.approx(50.0, rel=1e-4))


def test_damped_sinusoid_is_settling_not_a_cycle():
    """ Ringing onto a fixed point crosses its mean every period, so cycle_window finds as many
    cycles as on a limit cycle; the amplitude ratio is what tells them apart. """
    gamma = 1e-3                                         # per ms: 0.05 of the amplitude per 2 periods
    o = an.rate_and_epr(_traj(_sine(gamma)), DT, 4.0, cycles=True)
    assert o["n_cycles"] > 100
    assert o["state"] == "settling"
    # first and last whole cycles sit ~ (n_cycles - 1) periods apart
    expect = math.exp(-gamma * PERIOD * (o["n_cycles"] - 1))
    assert o["amp_ratio"] == pytest.approx(expect, rel=0.02)


def test_amplitude_tolerance_is_the_boundary():
    """ A decay just inside amp_tol over the window still counts as a cycle. """
    gamma = -math.log(0.97) / T                          # 3 % over the whole window
    o = an.rate_and_epr(_traj(_sine(gamma)), DT, 4.0, cycles=True, amp_tol=0.05)
    assert o["state"] == "cycle"
    assert an.rate_and_epr(_traj(_sine(gamma)), DT, 4.0, cycles=True, amp_tol=0.01)["state"] \
        == "settling"


def test_constant_is_settled_and_saturation_uses_the_mean():
    r = torch.full((int(T / DT),), 40.0, dtype=torch.float64)
    o = an.rate_and_epr(_traj(r), DT, 4.0, cycles=True)
    assert (o["state"], o["n_cycles"], o["amp_ratio"]) == ("settled", 0, None)
    assert o["dead_frac"] == pytest.approx(40.0 * 4.0 * 1e-3)


def test_without_cycles_no_state():
    o = an.rate_and_epr(_traj(_sine(0.0)), DT, 4.0)
    assert (o["state"], o["n_cycles"], o["amp_ratio"]) == (None, None, None)
    assert o["dead_frac"] == pytest.approx(o["r_mean"] * 4.0 * 1e-3)
