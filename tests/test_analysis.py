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


@pytest.mark.parametrize("dt", [0.2, 0.1, 0.05])
def test_phase_profile_recovers_the_waveform(dt):
    """ A rate with a known period and a second signal locked to it: the profile is that
    signal's shape at each phase, whatever the step -- crossings interpolated between steps,
    so a period that is not a whole number of steps does not smear it. """
    period, n_bins = 22.37, 50
    t = torch.arange(int(T / dt), dtype=torch.float64) * dt + 0.3   # not starting at a crossing
    ph = (t / period) % 1.0
    r = 40.0 + 10.0 * torch.sin(2 * math.pi * ph)
    x = torch.cos(2 * math.pi * ph) ** 2
    prof = an.phase_profile(x, r, n_bins)
    centre = (torch.arange(n_bins, dtype=torch.float64) + 0.5) / n_bins
    expect = torch.cos(2 * math.pi * centre) ** 2
    assert not prof.isnan().any()
    assert (prof - expect).abs().max() < 0.07                 # a bin is 1/50 of a period wide
    assert float(prof.mean()) == pytest.approx(float(x.mean()), abs=5e-3)


def test_phase_profile_needs_a_cycle():
    r = torch.full((100,), 40.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        an.phase_profile(r, r, 10)


def test_observed_order_and_richardson():
    """ A first-order sequence x(dt) = 1 + dt at dt = 0.4, 0.2, 0.1: order 1 and limit 1. A
    turning point and an order outside the band are refused, and richardson returns nan. """
    p, why = an.observed_order([1.4, 1.2, 1.1])
    assert why == "" and p == pytest.approx(1.0)
    assert an.richardson([1.4, 1.2, 1.1]) == pytest.approx((1.0, 1.0))
    assert an.richardson([1.4, 1.2, 1.1], p=2)[0] == pytest.approx(1.1 - 0.1 / 3)
    assert an.observed_order([1.4, 1.2, 1.3])[1] == "sign change"
    assert an.observed_order([1.4, 1.2, 1.1], p_range=(1.5, 3.0)) == (pytest.approx(1.0),
                                                                       "out of range")
    with pytest.warns(RuntimeWarning, match="refused"):
        assert math.isnan(an.richardson([1.4, 1.2, 1.3])[0])


def test_running_mean_is_centred_and_keeps_the_length():
    x = torch.arange(10, dtype=torch.float64)
    y = an.running_mean(x, 3)
    assert len(y) == 10 and torch.allclose(y[1:-1], x[1:-1])
    assert y[0] == 0.5 and y[-1] == 8.5            # truncated windows at the ends


def test_noisy_cycles_counts_the_cycles_not_the_noise():
    """ A 25 ms sine under noise that crosses the mean many times per cycle: the period hint
    gives the right count where the raw signal does not. """
    g = torch.Generator().manual_seed(0)
    r = 20.0 + 10.0 * torch.sin(2 * math.pi * torch.arange(30000) * DT / PERIOD)
    r = r + 2.0 * torch.randn(30000, generator=g, dtype=torch.float64)
    assert an.cycle_window(r)[2] > 1000                # 120 periods in the window
    _, per, n = an.noisy_cycles(r, DT, PERIOD)
    assert n == 118 and per == pytest.approx(PERIOD, rel=1e-2)
