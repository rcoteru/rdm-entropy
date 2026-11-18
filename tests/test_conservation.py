""" The age distribution is a probability: the transport conserves its mass exactly. """

import pytest

from conftest import HAZ_DT, WC
from rdme.mean_field import RDMWilsonCowan


@pytest.mark.parametrize("hazard,dt", HAZ_DT)
def test_age_mass_is_conserved(hazard, dt):
    net = RDMWilsonCowan(**WC, dt=dt, hazard=hazard)
    for _ in range(4):
        net.forward(500)
        for p in net.p:
            assert p.sum().item() == pytest.approx(1.0, abs=1e-12)
            assert bool((p >= 0).all())
