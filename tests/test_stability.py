""" rdme.stability: the matrix-free spectra against the dense oracle, the batched and chunked
solvers against their one-shot forms, the fixed points a lattice alone misses, a bifurcation
on a cut, and the attractor classes on signals with known answers. """

import math

import pytest
import torch

import rdme.stability as stab
from conftest import SETTLED
from rdme.mean_field import RDMWilsonCowan, RDMWilsonCowanBatch

# The Schwalger 2017 Fig. 5 neuron, identical for E and I; dt = 0.2 keeps Q ~ 1000
FIG5 = dict(beta_E=0.4, beta_I=0.4, theta_E=15.0, theta_I=15.0, tau_int_E=20.0, tau_int_I=20.0,
            tau_ref_E=1.0, tau_ref_I=1.0, K_ref_E=0.0, K_ref_I=0.0, nu0=0.01, t_abs=4.0,
            dt=0.2, hazard="poisson_exact")


@pytest.fixture(scope="module")
def settled():
    net = RDMWilsonCowan(**SETTLED, dt=0.2, hazard="poisson_exact")
    return net, net.fixed_points()[0]


def test_leading_modes_match_the_dense_spectrum(settled):
    """ The leading modes against the dense spectrum; a warm start from the leading
    eigenvector converges in fewer matvecs. """
    net, m = settled
    dt = net.dt
    dense = net.fp_eigenvalues(m)
    dense = dense[dense.abs().argsort(descending=True)][:2]
    got = net.leading_modes(m, k=2)
    assert torch.allclose(got["mu"], dense.abs().log() / dt, rtol=0, atol=1e-7)
    # a conjugate pair's order is arbitrary: compare |omega|
    assert torch.allclose(got["omega"].abs().sort().values,
                          (dense.angle() / dt).abs().sort().values, rtol=0, atol=1e-7)
    warm = net.leading_modes(m, k=2, v0=got["v"])
    assert torch.allclose(warm["mu"], got["mu"], rtol=0, atol=1e-7)
    assert warm["n_matvec"] < got["n_matvec"]


def test_leading_growth_matches_the_leading_mode(settled):
    net, m = settled
    want = float(net.leading_modes(m, k=2)["mu"][0])
    assert float(net.leading_growth(m, t_ms=1000.0)) == pytest.approx(want, abs=1e-4)


def test_batched_leading_growth_is_each_systems():
    """ A batch iterates each row on its own: the same rates as the systems one at a time. """
    W = torch.tensor([860.0, 960.0])
    mf = RDMWilsonCowanBatch(0.8, w_EE=960., w_EI=W, w_IE=-1200., w_II=-1200., E_exc=24.,
                             E_inh=24., **FIG5)
    m = torch.stack([mf.select(b).fixed_points()[0] for b in range(2)])
    got = mf.leading_growth(m, t_ms=200.0)
    one = torch.stack([mf.select(b).leading_growth(m[b], t_ms=200.0) for b in range(2)])
    assert torch.allclose(got, one, rtol=0, atol=1e-10)


def test_fp_sweep_in_chunks_is_the_one_shot_sweep():
    W = torch.tensor([0.0, 400.0, 900.0])
    mf = RDMWilsonCowanBatch(0.8, w_EE=960., w_EI=W, w_IE=-1200., w_II=-1200., E_exc=24.,
                             E_inh=24., **FIG5)
    r1, ok1 = mf.fp_sweep(n_per_axis=4)
    r2, ok2 = mf.fp_sweep(n_per_axis=4, chunk=5)
    assert torch.equal(ok1, ok2)
    assert torch.allclose(r1[ok1], r2[ok2], rtol=0, atol=1e-12)


def test_run_chunked_halves_on_out_of_memory():
    """ A function that only fits 3 rows at a time: the chunk halves from 10 to 2, and the
    result is the whole range in order. """
    def fn(sl):
        if sl.stop - sl.start > 3:
            raise torch.OutOfMemoryError("too big")
        return torch.arange(sl.start, sl.stop)
    out, chunk = stab.run_chunked(fn, 10)
    assert torch.equal(out, torch.arange(10))
    assert chunk == 2


def test_fixed_points_finds_what_the_lattice_alone_misses():
    """ The w(I→I) = 0 E-I family at w(E→I) = 0: a stable low state, a saddle and an unstable
    focus. The default starts include the exact 1-D reduction and find all three. """
    mf = RDMWilsonCowan(0.8, w_EE=960., w_EI=0., w_IE=-1200., w_II=0., E_exc=24., E_inh=24.,
                        **FIG5)
    roots = mf.fixed_points()
    assert len(roots) == 3
    assert len(stab.reduced_guesses(mf)) == 3


def test_a_hopf_on_the_fig5_cut():
    """ Along w(E→I) through Fig. 5 the fixed point loses stability below ~886 (dt = 0.1) in a
    Hopf at ~20 Hz: one crossing, unstable below, with a ~zero growth rate where it is
    located. """
    def build(v, device):
        return RDMWilsonCowanBatch(0.8, w_EE=960., w_EI=v, w_IE=-1200., w_II=-1200.,
                                   E_exc=24., E_inh=24., **FIG5, device=device)
    scan = stab.scan_cut(build, [840.0, 940.0], device="cpu", n_per_axis=6, t_ms=500.0)
    assert [len(s["m"]) for s in scan] == [1, 1]
    lm = stab.find_bifurcations(scan, build, tol=1.0)
    assert len(lm) == 1 and lm[0]["kind"] == "hopf" and lm[0]["side"] == "below"
    assert 860.0 < lm[0]["value"] < 912.0
    assert 15.0 < lm[0]["freq_hz"] < 25.0
    assert abs(lm[0]["mu"]) < 2e-3


def _rates(r_E: torch.Tensor) -> torch.Tensor:
    """ (T, 2) Hz with I a copy of E. """
    return torch.stack([r_E, r_E], -1)


def test_attractor_classes():
    t = torch.arange(20000, dtype=torch.float64) * 0.1
    s = torch.sin(2 * math.pi * t / 50.0)
    r = torch.stack([_rates(20.0 + 0 * t),                                # async
                     _rates(20.0 + 10.0 * s),                             # cycle
                     _rates(20.0 + 300.0 * s.clamp(min=0) ** 8),          # burst
                     _rates(20.0 + 10.0 * torch.exp(-t / 300.0) * s)])    # settling
    assert stab.classify_attractor(r, ceiling=250.0) == ["async", "cycle", "burst", "settling"]
    diff = stab.compare_starts(r, r[[0, 1, 1, 3]], ceiling=250.0)
    assert diff.tolist() == [False, False, True, False]
