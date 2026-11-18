""" The suite checks physics with exact answers: conservation, closed forms at stationarity,
symmetries, the EPR against its definition, and the spins against the mean field as N grows.
Every oracle it uses is in the library; this file holds only the shared operating points. """

from typing import Any

import pytest
import torch

# Two-population Wilson-Cowan cell. From its default start it rings for a long transient before
# settling on a focus, so short runs exercise moving dynamics; small time constants keep the
# age grids short.
WC: dict[str, Any] = {"E_ratio": 0.8, "w_EE": 3.67, "w_EI": 15.910, "w_IE": -7.955, "w_II": -2.50,
      "E_exc": 1.16, "E_inh": 1.16, "beta_E": 30.0, "beta_I": 30.0, "theta_E": 1.0, "theta_I": 1.0,
      "tau_int_E": 6.0, "tau_int_I": 8.0, "tau_ref_E": 3.0, "tau_ref_I": 3.0,
      "K_ref_E": 0.5, "K_ref_I": 0.0}

# The same cell moved onto a stable fixed point.
SETTLED = dict(WC, w_EE=0.5, w_EI=1.0, w_IE=-1.0, w_II=-1.0, E_exc=1.0, E_inh=1.0)

# Every kernel, each at a step it admits that divides t_abs = 1 ms.
HAZ_DT = [("poisson_exact", 0.2), ("poisson", 0.1), ("bernoulli", 0.5)]


@pytest.fixture(autouse=True)
def float64():
    """ For the tensors the tests build themselves: the exact checks are asserted near
    roundoff, which float32 cannot reach. The library sets its own dtypes and does not need
    this -- the mean field is float64 whatever the default (test_dtype), and the spin model
    builds its own float32 state. """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)
