""" The library's precision is its own, whatever torch's global default is, so no script has
to set one: the mean field is float64, the spin model float32 state with float64 draws and
accumulators. """

import torch

from conftest import S1, WC
from rdme.mean_field import RDMIsingModelBatch, RDMWilsonCowan, RDMWilsonCowanPoints
from rdme.spin_model import SpinIsingModel


def _run(global_dtype):
    torch.set_default_dtype(global_dtype)    # the autouse fixture restores float64
    wc = RDMWilsonCowan(**WC, dt=0.2, hazard="poisson_exact")
    out = {"roots": wc.fixed_points(),
           "tangent": wc.init_tangent(2, generator=torch.Generator().manual_seed(0))}
    out |= {f"wc.{k}": v for k, v in wc.entropy_trajectory(30, pb=False).items()}
    pts = RDMWilsonCowanPoints(**{**WC, "w_EE": [1.5, 2.5]}, dt=0.2, hazard="poisson_exact")
    out["points.m"] = pts.trajectory(30)["m"]
    ising = RDMIsingModelBatch([1.2375, 17.0], 0.0, 6.0, 0.5, 15.0, 15.0, 15.0,
                               dt=0.5, hazard="poisson_exact")
    out["ising.sigma"] = ising.entropy_trajectory(30, pb=False)["sigma"]
    out["ising.w"] = ising.w
    return out


def test_the_mean_field_ignores_the_global_default_dtype():
    a, b = _run(torch.float32), _run(torch.float64)
    for k in b:
        assert a[k].dtype == torch.float64, k
        assert torch.equal(a[k], b[k]), k


def _spins(global_dtype):
    torch.set_default_dtype(global_dtype)
    torch.manual_seed(0)
    sm = SpinIsingModel(500, 17.0, **S1 | dict(hazard="poisson_exact", dt=0.5),
                        device="cpu", ic="random")
    sm.forward(100)
    return sm.trajectory(100) | sm.entropy_trajectory(100, chunk=64, lookahead=64)


def test_the_spin_model_ignores_the_global_default_dtype():
    """ Same draws and the same float64 accumulators either way: the scripts used to set a
    float64 default, and the spins must not have depended on it. """
    a, b = _spins(torch.float32), _spins(torch.float64)
    for k in b:
        assert torch.equal(a[k], b[k]), k
