""" Golden literals from the dumps, reproduced from the current library at the dumps' settings.

Step 0 of `P1 2026-11-13 code rebuild`: the numbers three dumps printed, pinned so the rebuild
cannot move them. Each test re-runs, through library calls only, the measurement path of
`scripts/S1-nu0_test.py` (`measure` / `measure_full`): T_EQUI = 8000 ms, T_MEAS = 4000 ms,
`poisson_exact`, dt = 0.25, default initial condition, whole-cycle averages between upward
mean-crossings (`helpers.cycle_window`), and the step diagnostics probed over one further
period after the record.

The age window Q is the one the script fitted (`helpers.fit_age_grid` at tol 1e-7 over
T_EQUI + T_MEAS), held here as a literal rather than refitted: the fit is a solver setting the
script chose, and these tests pin what the model does at that setting.

`lam_max` is the PEAK HAZARD RATE (1/ms) over refractory-free ages, not a Lyapunov exponent.
On all three points it is attained at the last age bin, Q - 1, so it (and max Phi) depends on
Q: that is why the re-anchored cycle reads 6.528 at Q = 1296 (`..._e-restore`) and 5.900 at
Q = 411 (`..._ladder-and-cost`).

Literals are the full float64 reprs of the run; at the time of writing the reproduction is
bit for bit (and identical to the scripts' caches). Asserted at rel 1e-12.
"""

import pytest
import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.network import RDMIsingModel

HAZARD = "poisson_exact"
DT = 0.25
T_EQUI, T_MEAS = 8000.0, 4000.0   # ms
REL = 1e-12


@pytest.fixture(autouse=True)
def float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _measure(P: dict, J: float, Q: int) -> dict:
    """ `S1-nu0_test.py`'s `measure`, the columns the dumps quote. """
    model = {k: v for k, v in P.items() if k not in ("dt", "hazard")}
    net = RDMIsingModel(J, **model, dt=DT, hazard=HAZARD, Qm=[Q], device="cpu")
    net.forward(hlp.n_steps(T_EQUI, DT))
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, DT), pb=False)

    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    period = (z - a) * DT / n_cyc if n_cyc else float("nan")

    n_probe = max(1, hlp.n_steps(period, DT) if n_cyc else len(r) // 10)
    phi_max, lam_max, n_abs = 0.0, 0.0, net.n_abs
    for _ in range(n_probe):
        phi_max = max(phi_max, net.max_step_prob())
        lam_max = max(lam_max, max(float(l[n_abs:].max()) for l in net.hazard_rate()))
        net.update()

    return {"r": float(r[a:z].mean()),
            "sigma": float(traj["sigma_tot"][a:z].mean()) / DT,
            "period": period,
            "ptp": float(r[a:z].max() - r[a:z].min()),
            "phi_max": phi_max,
            "lam_max": lam_max}


def _check(got: dict, want: dict):
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=REL, abs=0.0), (k, got[k], v)


@pytest.mark.slow
def test_s1_fixed_point_J10():
    """ P1_2026-11-13_S1-nu0_test_gauge-control-and-retune.md §1, fp column.
    Dump: r = 0.013124117661485, period = 76.547619047619, lam_max = 0.10930602061432. """
    got = _measure(pre.params("S1", hazard=HAZARD, dt=DT), 10.0, Q=683)   # U_max 170.75 ms
    _check(got, {"r":       0.013124117661485284,
                 "period":  76.54761904761905,
                 "lam_max": 0.1093060206143206})


@pytest.mark.slow
def test_s1_cycle_J17():
    """ Same dump, cycle column. Dump: r = 0.014635194999677, sigma = 0.014990859463552,
    period = 68.614035087719, ptp = 0.15971051787548, max Phi = 0.75257152746213,
    lam_max = 5.5865349543556. """
    got = _measure(pre.params("S1", hazard=HAZARD, dt=DT), 17.0, Q=554)   # U_max 138.5 ms
    _check(got, {"r":       0.014635194999676993,
                 "sigma":   0.01499085946355222,
                 "period":  68.6140350877193,
                 "ptp":     0.1597105178754808,
                 "phi_max": 0.7525715274621332,
                 "lam_max": 5.586534954355592})


@pytest.mark.slow
def test_reanchored_cycle_J17():
    """ P1_2026-11-13_S1-nu0_test_ladder-and-cost.md §1, row dt = 1/4, side 're': S1 with
    nu0 = 0.01, theta = 0.5 held, E tuned at dt = 0.25 (0.780717523 in the dump; full value
    from the erest stage), J = 17, window fitted at dt = 0.25 and held: 102.75 ms -> Q = 411.
    Dump: sigma = 1.606919e-02, ptp = 0.168992. """
    P = pre.params("S1", hazard=HAZARD, dt=DT, nu0=0.01, E=0.7807175232690611)
    got = _measure(P, 17.0, Q=411)
    _check(got, {"sigma": 0.016069190712935534,
                 "ptp":   0.16899191196539967})
