import inspect

import pytest
import torch

import rdme.presets as pre
from rdme.hazards import HAZARDS
from rdme.rdm_single import RDMIsingModel
from rdme.spin_model import SpinIsingModel

""" What these check is that the presets are a SPECIFICATION and not a note: that every name
builds the model it claims to, that MODEL_KEYS is the constructor's own signature rather than a
list that drifted from it, and that the guards (hazard required, typo rejected) hold. The
numbers themselves are not tested -- they are the choice being recorded. """


ONE_POP = sorted(k for k, m in pre.POPS.items() if m == 1)


def test_model_keys_match_the_constructor():
    """ MODEL_KEYS[1] is exactly what RDMIsingModel takes, minus the solver keys. A parameter
    added to the model and not here would silently become an unknown override. """
    sig = set(inspect.signature(RDMIsingModel).parameters) - set(pre.SOLVER_KEYS)
    assert sig == set(pre.MODEL_KEYS[1]) | {"hazard"}


@pytest.mark.parametrize("name", ONE_POP)
def test_every_preset_keys_are_model_parameters(name):
    assert set(pre.resolve(name)) <= pre.MODEL_KEYS[1]


@pytest.mark.parametrize("name", ONE_POP)
@pytest.mark.parametrize("hazard", sorted(HAZARDS))
def test_every_preset_builds_and_runs(name, hazard):
    """ Both models, at a step every kernel admits, for a handful of steps: a preset that
    cannot be handed to a constructor is not a preset. J is supplied for the sets that sweep
    it, and dt = 0.5 divides every preset's t_abs. """
    p = pre.params(name, hazard=hazard, dt=0.5,
                   **({} if "J" in pre.resolve(name) else {"J": 10.0}))

    mf = RDMIsingModel(**p, device="cpu")
    mf.forward(5, pb=False)
    assert torch.isfinite(mf.m).all()

    sm = SpinIsingModel(N=64, **p, device="cpu")
    sm.forward(5)
    assert torch.isfinite(sm.s.float()).all()


def test_points_resolve_onto_their_base():
    assert pre.resolve("S1/fp") == pre.resolve("S1") | {"J": 10.0}


def test_resolve_returns_a_copy():
    p = pre.resolve("S1")
    p["beta"] = 999.0
    assert pre.PRESETS["S1"]["beta"] != 999.0


def test_the_default_hazard_is_the_route_b_solver():
    """ The hazard is still its own axis -- every script states it -- but a run that does not
    gets the kernel a dt -> 0 limit is taken with, not an arbitrary one. """
    assert pre.DEFAULT_HAZARD == "poisson_exact"
    assert pre.params("S1", J=10.0, dt=0.5)["hazard"] == "poisson_exact"


def test_a_run_must_state_its_step():
    """ D7. Under route B there is no time quantum to default to, so dt is the one solver key
    params() insists on: a run that has not named its step has not said how it was solved,
    and the dump has nothing to record. """
    with pytest.raises(ValueError, match="needs dt="):
        pre.params("S1", hazard="poisson_exact", J=10.0)


def test_a_typo_is_rejected_rather_than_passed_on():
    with pytest.raises(TypeError, match="tau_m"):
        pre.params("S1", hazard="poisson", J=10.0, dt=0.5, tau_m=20.0)


def test_solver_keys_pass_through():
    p = pre.params("S1", hazard="poisson", J=10.0, dt=0.25, device="cpu")
    assert p["dt"] == 0.25 and p["device"] == "cpu"
    assert p["nu0"] == 1.0 and p["t_abs"] == 1.0


def test_pending_presets_say_why():
    with pytest.raises(KeyError, match="not encoded yet"):
        pre.resolve("EI")


def test_t_abs_is_a_physical_parameter_that_the_step_does_not_move():
    """ Under route B the dead time is 1 ms whatever dt is, which is what makes a dt -> 0
    limit a limit of one model. It reaches the model, and it reaches the age grid as a
    different number of BINS at each step, which is the same window either way. """
    for dt, n_abs in ((1.0, 0), (0.5, 1), (0.25, 3)):
        mf = RDMIsingModel(**pre.params("S1", hazard="poisson_exact", J=10.0, dt=dt),
                           device="cpu")
        assert mf.t_abs == pytest.approx(1.0)
        assert mf.n_abs == n_abs
        assert (mf.n_abs + 1) * mf.dt == pytest.approx(mf.t_abs)


def test_diff_reports_only_what_differs():
    p = pre.params("S1", hazard="poisson", J=torch.linspace(0, 20, 5), beta=8.0, dt=0.2)
    assert pre.diff("S1", p) == {"beta": 8.0}          # J is a swept axis, dt is the solver


def test_banner_states_preset_hazard_and_dt():
    line = pre.banner("S1/fp", pre.params("S1/fp", hazard="poisson", dt=0.2))
    assert "S1/fp" in line and "poisson" in line and "dt=0.2" in line and "t_abs=1" in line
