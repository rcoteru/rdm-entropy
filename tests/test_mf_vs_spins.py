""" The mean field is the N -> infinity limit of the spin network. The two share kernels and
the dead time exactly; the spins' fluctuations vanish as 1/N about the mean field; and their
entropy production converges onto the mean field's, zero at a fixed point and not on a cycle.

S1 at dt = 0.25. The age grid is pinned at Q = 600: the preset's default (277 bins) truncates
enough mass at both points to move the mean field's rate and sigma by percent. """

import pytest
import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.mean_field import RDMIsingModel
from rdme.spin_model import SpinIsingModel

DT = 0.25
S1 = pre.params("S1", hazard="poisson_exact", dt=DT)
J_FP, J_CYC = 10.0, 17.0


@pytest.mark.parametrize("t_abs", [0.0, 1.0])
@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2, 0.05])
def test_refractory_kernel_matches_spin_model(dt, t_abs):
    """ The refractory term a spin's hazard sees at age n, simulated through update(), is the
    mean field's R(n): same hold through the dead time, same decay after it. """
    tau_ref, K_ref, tau_int = 3.0, 2.0, 12.0
    kw = {"dt": dt, "t_abs": t_abs, "hazard": "poisson_exact"}
    torch.manual_seed(0)
    sm = SpinIsingModel(200, 0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, ic="silent", **kw)
    mf = RDMIsingModel(0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, **kw)
    ref = mf.R[0].float()
    seen = torch.zeros(sm.N, dtype=torch.bool)
    for _ in range(int(60 / dt)):
        sm.update()
        seen |= sm.s == 1                      # R is a function of age only after a spike
        ok = seen & (sm.n < mf.Q)
        got = sm.R_eff(sm.R, sm.n)[ok]
        assert torch.allclose(got, ref[sm.n[ok]], rtol=1e-4, atol=1e-6)
    assert seen.all()


@pytest.mark.parametrize("dt", [0.5, 0.2, 0.1])
def test_the_realised_dead_time_is_t_abs(dt):
    """ The shortest interspike interval the spins realise is t_abs, at every dt. """
    t_abs, N = 1.0, 4000
    torch.manual_seed(0)
    sm = SpinIsingModel(N=N, J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0,
                        K_ref=0.0, dt=dt, t_abs=t_abs, hazard="poisson_exact", ic="silent")
    sm.forward(int(400 / dt))
    last = torch.full((N,), -10**9, dtype=torch.long)
    worst, seen = 10**9, 0
    for t in range(int(600 / dt)):
        sm.update()
        idx = (sm.s == 1).nonzero(as_tuple=True)[0]
        gap = t - last[idx]
        prior = gap < 10**8
        if prior.any():
            worst = min(worst, gap[prior].min().item()); seen += int(prior.sum())
        last[idx] = t
    assert seen > 10_000
    assert worst * dt == pytest.approx(t_abs)


def _mf(J, ms):
    """ Mean-field rate and sigma per ms, averaged over whole cycles. """
    mf = RDMIsingModel(J, **S1)
    mf.forward(int(4000 / DT))
    t = mf.entropy_trajectory(int(ms / DT), pb=False)
    a, b, _ = hlp.cycle_window(t["r_tot"])
    return float(t["r_tot"][a:b].mean()), float(t["sigma_tot"][a:b].mean()) / DT


@pytest.mark.slow
def test_spin_fluctuations_vanish_as_one_over_N():
    """ At the fixed point the spins' mean rate is the mean field's, and the variance of
    their population rate falls exactly as 1/N. """
    r_mf, _ = _mf(J_FP, 2000)
    varN = []
    for N in (1000, 4000, 16000):
        torch.manual_seed(0)
        sm = SpinIsingModel(N, J_FP, **S1, device="cpu", ic="random")
        sm.forward(int(2000 / DT))
        r = sm.trajectory(int(8000 / DT))["m_tot"].double() / DT
        assert float(r.mean()) == pytest.approx(r_mf, rel=2e-3), N
        varN.append(float(r.var()) * N)
    assert max(varN) / min(varN) < 1.05, varN


def _spin_sigma(J, N, ms=4000.0, blocks=40):
    """ Spin sigma per ms, with a blocked standard error: the per-step estimate is a sampled
    log-ratio, correlated across a burst. """
    torch.manual_seed(0)
    sm = SpinIsingModel(N, J, **S1, device="cpu", ic="random")
    sm.forward(int(2000 / DT))
    sg = sm.entropy_trajectory_chunked(int(ms / DT), chunk=4096, overlap=4096)["sigma_tot"]
    sg = sg.double() / DT
    blk = sg[:len(sg) // blocks * blocks].reshape(blocks, -1).mean(dim=1)
    return float(sg.mean()), float(blk.std()) / blocks ** 0.5


@pytest.mark.slow
@pytest.mark.parametrize("J", [J_FP, J_CYC], ids=["fixed-point", "cycle"])
def test_spin_sigma_converges_to_the_mean_field(J):
    _, sig_mf = _mf(J, 4000)
    sig, sem = _spin_sigma(J, 16000)
    assert sem < 1e-3, sem
    assert abs(sig - sig_mf) < 4 * sem, (sig, sig_mf, sem)
    if J == J_CYC:
        assert sig_mf > 10 * sem, "on the cycle sigma must be resolvably nonzero"
