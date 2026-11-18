""" The age window: the grid sized from tol is a bound on what the truncation costs, at every
step, independent of dt. """

import pytest
import torch

import rdme.hazards as hzd
from conftest import S1
from rdme import epr
from rdme.mean_field import RDMIsingModel

TOL = 1e-2

# (hazard, dt, J, beta) on S1: a healthy cycle under each kernel, the bursting corner of the
# (beta, J) map (CV ~ 10), and a near-silent fixed point, where the boundary bin holds almost
# all the mass and the bound is tight.
CASES = [("poisson_exact", 0.5, 17.0, 6.0), ("poisson", 0.25, 17.0, 6.0),
         ("bernoulli", 0.5, 17.0, 6.0), ("poisson_exact", 0.5, 60.0, 15.25),
         ("poisson_exact", 0.5, 10.0, 20.0)]


def _worst_step_error(hazard, dt, J, beta, tol=TOL):
    """ Largest relative firing error, over a stretch of a long-grid trajectory, of lumping
    every cohort older than the tol-sized window into one bin at the field of its age -- which
    is exactly what one step of the truncated model does from that state. """
    P = S1 | dict(hazard=hazard, J=J, beta=beta, dt=dt)
    Q = RDMIsingModel(**P, tol=tol).Q
    ref = RDMIsingModel(**P, tol=1e-12)
    assert ref.Q > 2 * Q
    ref.forward(int(3000 / dt))
    worst = 0.0
    for _ in range(int(1500 / dt)):
        p, phi = ref.p, ref.firing_prob()
        m = (p * phi).sum(-1)
        lumped = p[..., :Q].clone()
        lumped[..., -1] = p[..., Q - 1:].sum(-1)
        m_trunc = (lumped * phi[..., :Q]).sum(-1)
        worst = max(worst, float(((m_trunc - m).abs() / m).max()))
        ref.update()
    return worst


@pytest.mark.parametrize("hazard,dt,J,beta", CASES)
def test_the_window_bounds_the_truncation_at_every_step(hazard, dt, J, beta):
    assert _worst_step_error(hazard, dt, J, beta) <= TOL


def test_the_bound_is_tight_where_the_boundary_holds_the_mass():
    """ Near silence nearly every cohort sits in the last bin, and the realised error comes
    within a factor of two of tol: the window is not padded. """
    assert _worst_step_error("poisson_exact", 0.5, 10.0, 20.0) > TOL / 4


@pytest.mark.slow
@pytest.mark.parametrize("J,beta", [(17.0, 6.0), (60.0, 15.25)], ids=["cycle", "bursting"])
def test_the_window_bounds_the_entropy_production(J, beta):
    """ sigma from the observer a tol-sized model runs -- lumped marginal, Q-long window --
    against the long-grid observer, on ONE reference trajectory, so only the observer's own
    two boundaries (the forward lumped bin and the reverse censoring bin) can differ. Averaged
    over 300 ms, longer than the bursting cell's 255 ms period. """
    dt = 0.5
    P = S1 | dict(hazard="poisson_exact", J=J, beta=beta, dt=dt)
    Q = RDMIsingModel(**P, tol=TOL).Q
    ref = RDMIsingModel(**P, tol=1e-8)
    QR, steps = ref.Q, int(300 / dt)
    ref.forward(int(3000 / dt))
    ps, Ss, Is = [], [], []
    for k in range(steps + QR):
        if k < steps:
            ps.append(ref.p.clone())
            Ss.append(ref.S.clone())
        Is.append(ref.total_input().clone())
        ref.update()
    I = torch.stack(Is, -1)

    def sigma(q):
        prm = (ref.R[..., :q], ref.blocked[..., :q], ref.alpha_int[..., None], ref.beta[..., None],
               ref.theta[..., None], hzd.HAZARDS[ref.hazard], ref.nu0, dt)
        total = 0.0
        for t in range(steps):
            p = ps[t][..., :q].clone()
            p[..., -1] = ps[t][..., q - 1:].sum(-1)
            _, s, _, _ = epr.epr_step_joint((p, Ss[t][..., :q], I[..., t:t + q - 1]), None,
                                            I[..., t + q - 1], *prm)
            total += float(s)
        return total

    full = sigma(QR)
    assert abs(sigma(Q) - full) <= TOL * abs(full)


def test_the_window_does_not_depend_on_dt():
    """ U_max = (Q-1)*dt is a property of the model and tol; only its rounding to bins moves. """
    U = [(RDMIsingModel(**S1 | dict(hazard="poisson_exact", J=17.0, dt=dt)).Q - 1) * dt for dt in (0.5, 0.25, 0.1)]
    assert max(U) - min(U) <= 0.5, U
