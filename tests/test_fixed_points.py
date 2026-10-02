""" Tests for the fixed-point machinery: the closed-form stationary quantities in
rdme.fixed_points -- the closed forms at stationarity, the generic root finders, and the
RDMNetwork(Batch) methods built on top of them. The reference everything is checked against is the dynamics itself --
a fixed point must be a state forward() does not move, and its spectral radius must predict
whether a small kick decays.

Every model here uses hazard="synchronous", so these are statements about the fixed-point
machinery rather than about a particular kernel; the hazards are compared in test_srm.py. """

import pytest
import torch

import rdme.dynamics as dyn
import rdme.fixed_points as fpts
from rdme.batch import RDMIsingModelBatch
from rdme.single import RDMIsingModel, RDMWilsonCowan


torch.manual_seed(0)

DT = 0.2
ISING = dict(beta=30.0, theta=1.0, tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
             hazard="synchronous")
# J/E chosen inside the multi-root region: five fixed points, stable and unstable mixed
BISTABLE = dict(J=30.0, E=0.4, **ISING)

# weights in physical units: srm builds the input from the rate, so no 1/DT here
WC = dict(E_ratio=0.8, w_EE=3.0, w_EI=3.0, w_IE=-1.0, w_II=-1.0,
          E_exc=1.0, E_inh=1.0, beta_E=30.0, beta_I=30.0, theta_E=1.0, theta_I=1.0,
          tau_int_E=20.0, tau_int_I=20.0, tau_ref_E=3.0, tau_ref_I=3.0,
          K_ref_E=0.0, K_ref_I=0.0, dt=DT, hazard="synchronous")


@pytest.fixture(autouse=True)
def float64():
    """ The whole fixed-point path is a root find; float32 residuals bottom out around
    1e-7 and make every tolerance here meaningless. """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# Closed-form stationary quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_forward_field_constant_input():
    Q, alpha, I = 12, 0.3, torch.tensor(1.7)

    S = fpts.compute_stationary_forward_field(I, alpha, Q)

    # against the recursion it solves, not against the geometric sum it was derived from
    ref = torch.zeros(Q, dtype=S.dtype)
    for n in range(1, Q):
        ref[n] = alpha * I + (1 - alpha) * ref[n - 1]
    assert torch.allclose(S, ref, rtol=0, atol=1e-14)
    assert S[0] == 0

    # and it is a genuine fixed point of update_synaptic_term, boundary bin included
    assert torch.allclose(dyn.update_synaptic_term(S, alpha, I), S, rtol=0, atol=1e-14)

    # batched: (B,) inputs with (B,) alphas give (B, Q), row by row
    Ib = torch.tensor([0.5, 1.7, -2.0])
    ab = torch.tensor([0.1, 0.3, 0.8])
    Sb = fpts.compute_stationary_forward_field(Ib, ab, Q)
    assert Sb.shape == (3, Q)
    assert torch.allclose(Sb[1], S, rtol=0, atol=1e-14)


def test_survival():
    phi = torch.rand(10).clamp(0.05, 0.95)

    S = fpts.compute_stationary_survival(phi)

    assert S[0] == 1.0
    ref = torch.tensor([torch.prod(1 - phi[:n]) for n in range(10)])
    assert torch.allclose(S, ref, rtol=0, atol=1e-14)

    # log form is the same quantity, and is the reason a saturated hazard does not
    # collapse the whole tail to an unusable zero
    assert torch.allclose(fpts.compute_stationary_survival(phi, log=True), S.log(), rtol=0, atol=1e-14)
    hot = torch.full((200,), 1.0)
    assert torch.isfinite(fpts.compute_stationary_survival(hot, log=True)).all()


def test_stationary_age_distribution():
    phi = torch.rand(15).clamp(0.05, 0.95)

    p, m = fpts.compute_stationary_age_distribution(phi)

    assert torch.allclose(p.sum(), torch.tensor(1.0), rtol=0, atol=1e-14)
    assert torch.allclose(p[0], m, rtol=0, atol=1e-14)          # births are spikes
    # stationary under the renewal update it was derived from
    assert torch.allclose(dyn.update_age_distribution(p, phi), p, rtol=0, atol=1e-14)
    # the reset branch is an identity, not an extra condition (telescoping)
    assert torch.allclose((p * phi).sum(), m, rtol=0, atol=1e-14)


def test_stationary_distribution_matches_renewal_rate():
    """ With the boundary bin pushed far out, m* must approach the isolated-neuron
    renewal rate 1/sum_n S_0(n) -- the untruncated limit of the appendix. """
    phi = torch.full((400,), 0.05)
    _, m = fpts.compute_stationary_age_distribution(phi)
    surv = fpts.compute_stationary_survival(phi)
    # the two differ by the boundary bin's resummed tail, ~S_0(Q)/Phi, which the truncation
    # criterion drives to zero -- it is not an identity, only a limit
    assert m.item() == pytest.approx(1.0 / surv.sum().item(), rel=1e-6)


# Generic solvers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _cubic(m):
    """ Scalar residual with three roots at 0.2, 0.5, 0.8, in (M,)->(M,) form. """
    return (m - 0.2) * (m - 0.5) * (m - 0.8)


def test_bracket_roots_finds_every_root():
    roots = fpts.bracket_roots(_cubic, 0.0, 1.0, n_grid=101).flatten()
    assert torch.allclose(roots, torch.tensor([0.2, 0.5, 0.8]), rtol=0, atol=1e-12)


def test_newton_roots_converges_and_deduplicates():
    guesses = torch.linspace(0.0, 1.0, 21).unsqueeze(-1)
    roots = fpts.newton_roots(_cubic, guesses, tol=1e-12).flatten()
    # 21 starting points, 3 distinct roots left after deduplication
    assert torch.allclose(roots, torch.tensor([0.2, 0.5, 0.8]), rtol=0, atol=1e-9)


def test_newton_roots_vector_valued():
    """ A genuinely coupled M=2 residual, to check the Jacobian solve is not silently
    treating the problem as two scalar ones. """
    target = torch.tensor([0.3, 0.7])

    def residual(m):
        return torch.stack([m[0] + m[1] ** 2 - (target[0] + target[1] ** 2),
                            m[0] * m[1] - target[0] * target[1]])

    roots = fpts.newton_roots(residual, fpts.fp_guess_lattice(2, 7), tol=1e-12)
    assert any(torch.allclose(r, target, rtol=0, atol=1e-8) for r in roots)


def test_newton_params_sweep():
    """ Per-row parameters: one Newton trajectory per (parameter, guess) pair, all in one
    vmapped call. Root of m^2 - c is sqrt(c). """
    c = torch.tensor([0.04, 0.25, 0.64]).repeat_interleave(2).unsqueeze(-1)
    m0 = torch.tensor([[0.1], [0.9]]).repeat(3, 1)

    m = fpts.newton_iterate(lambda x, ci: x ** 2 - ci, m0, (c,), tol=1e-14)

    assert torch.allclose(m.flatten(), c.flatten().sqrt(), rtol=0, atol=1e-10)
    assert (fpts.residual_norms(lambda x, ci: x ** 2 - ci, m, (c,)) < 1e-12).all()


def test_deduplicate():
    roots = torch.tensor([[0.5], [0.5 + 1e-9], [0.2], [0.8], [0.2 - 1e-9]])
    assert torch.allclose(fpts.deduplicate(roots, 1e-6).flatten(),
                          torch.tensor([0.2, 0.5, 0.8]), rtol=0, atol=1e-8)
    # distinct roots closer than the default merge tolerance must survive a tighter one
    assert fpts.deduplicate(roots, 1e-12).shape[0] == 5
    # empty in, empty out -- a sweep cell where nothing converged is not an error
    assert fpts.deduplicate(torch.zeros(0, 1)).numel() == 0


# Model fixed points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_ising_fixed_point_is_stationary():
    """ The defining property: set_state on a root, integrate, and nothing moves. """
    model = RDMIsingModel(J=2.0, E=1.0, **ISING)
    roots = model.fixed_points()
    assert roots.shape[1] == 1

    for r in roots:
        assert model.fp_residual(r).abs().max() < 1e-12
        run = RDMIsingModel(J=2.0, E=1.0, **ISING)
        run.set_state(r)
        run.forward(2000)
        assert run.m.item() == pytest.approx(r.item(), abs=1e-12)


def test_ising_free_run_lands_on_a_stable_root():
    """ J is kept small on purpose: the slowest mode here has |lambda| ~ 0.9989, so the
    relaxation already takes ~10^4 steps, and closer to the Hopf boundary (J/dt ~ 10) it
    takes longer than any test wants to run. """
    model = RDMIsingModel(J=0.5, E=1.0, **ISING)
    roots = model.fixed_points()
    stable = [r for r in roots if model.is_stable(r)]
    assert stable

    model.forward(40000)
    assert any(model.m.item() == pytest.approx(r.item(), abs=1e-6) for r in stable)


def test_bracketing_finds_the_unstable_branch():
    """ The multi-root region is the whole point of the scalar bracketing path: Newton
    from a coarse lattice reaches only the wide basins, while the scan is exhaustive. """
    model = RDMIsingModel(**BISTABLE)
    roots = model.fixed_points(method='bracket', n_grid=2001)

    assert roots.shape[0] >= 3
    assert (torch.stack([model.fp_residual(r).abs().max() for r in roots]) < 1e-10).all()
    # roots are distinct and ordered, and both stability classes are represented
    assert (torch.diff(roots.flatten()) > 1e-6).all()
    stability = [model.is_stable(r) for r in roots]
    assert any(stability) and not all(stability)


def test_stability_predicts_the_response_to_a_kick():
    """ The spectral radius is not decoration: perturb a fixed point and the deviation
    must shrink at a stable root and grow at an unstable one. """
    model = RDMIsingModel(**BISTABLE)
    roots = model.fixed_points(method='bracket', n_grid=2001)

    kick = 1e-7

    def kick_grows(r):
        run = RDMIsingModel(**BISTABLE)
        run.set_state(r)
        run.p[0][0] += kick
        run.p[0][1] -= kick
        run.m = run.p[0][0:1].clone()
        dev = (run.trajectory(5000)["m"].squeeze(-1) - r.item()).abs()
        # an unstable root leaves the linear regime long before the run ends -- the most
        # unstable one here multiplies the kick by ~1.7 per step -- so "grew" has to be
        # read off the escape, not off a ratio of endpoint deviations that has already
        # saturated against whatever attractor caught the trajectory
        return bool((dev > 1e3 * kick).any())

    for r in roots:
        rho = fpts.spectral_radius(model.fp_eigenvalues(r)).item()
        if abs(rho - 1.0) < 1e-3:
            continue                      # marginal: the linearization decides nothing
        assert kick_grows(r) == (rho > 1.0), f"root {r.item()}: rho={rho}"


def test_state_map_matches_update():
    """ state_map is the autograd-friendly twin of update() and must step identically,
    or the Jacobian it is differentiated for describes a different model. """
    model = RDMIsingModel(J=2.0, E=1.0, **ISING)
    model.forward(50)                                  # away from any special state
    x = model.flatten_state(model.p, model.S)

    x_next = model.state_map(x)
    model.update()

    p, S = model.unflatten_state(x_next)
    assert torch.allclose(p[0], model.p[0], rtol=0, atol=1e-14)
    assert torch.allclose(S[0], model.S[0], rtol=0, atol=1e-14)


def test_flatten_unflatten_roundtrip():
    model = RDMWilsonCowan(**WC)
    p, S = model.unflatten_state(model.flatten_state(model.p, model.S))
    assert all(torch.equal(a, b) for a, b in zip(p, model.p))
    assert all(torch.equal(a, b) for a, b in zip(S, model.S))


def test_wilson_cowan_fixed_point():
    """ M=2: the Newton path, on ragged per-population age grids. """
    model = RDMWilsonCowan(**WC)
    roots = model.fixed_points()
    assert roots.shape[1] == 2
    assert roots.shape[0] >= 1

    for r in roots:
        assert model.fp_residual(r).abs().max() < 1e-10
        run = RDMWilsonCowan(**WC)
        run.set_state(r)
        run.forward(1000)
        assert torch.allclose(run.m, r, rtol=0, atol=1e-10)


# Batched sweep
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_fp_sweep_matches_the_single_model():
    J = torch.tensor([1.0, 2.0, 3.0])
    E = torch.tensor([0.9, 1.1])
    batch = RDMIsingModelBatch(J=J, E=E, beta=30.0, theta=1.0,
                                  tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
                                   hazard="synchronous")

    roots, converged = batch.fp_sweep(n_per_axis=6)
    assert roots.shape == (batch.B, 6, 1)
    assert converged.shape == (batch.B, 6)

    # the grid is the outer product of J and E, in that order (see flatten_param_grid)
    for b, (j, e) in enumerate([(j, e) for j in J for e in E]):
        single = RDMIsingModel(J=j.item(), E=e.item(), beta=30.0, theta=1.0,
                                   tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
                                   hazard="synchronous")
        expected = single.fixed_points(n_grid=2001)
        found = fpts.deduplicate(roots[b][converged[b]])
        # every Newton trajectory that converged must have landed on a genuine root
        assert all(min((f - x).abs().item() for x in expected) < 1e-7 for f in found)


def test_batch_set_state_is_stationary():
    batch = RDMIsingModelBatch(J=torch.tensor([1.0, 2.0]), E=1.0, beta=30.0,
                                  theta=1.0, tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
                                   hazard="synchronous")
    roots, converged = batch.fp_sweep(n_per_axis=4)
    m = roots[:, 0, :]                          # first guess converges for every cell here
    assert converged[:, 0].all()

    batch.set_state(m)
    assert torch.allclose(batch.fp_residual(m).abs().max(), torch.tensor(0.0), atol=1e-12)
    batch.forward(1000, pb=False)
    assert torch.allclose(batch.m, m, rtol=0, atol=1e-12)


def test_select_matches_the_batch_element():
    """ select() is what lets a cached sweep be analysed one system at a time; it must
    reproduce the batch's own step exactly, kernels and age grid included. """
    batch = RDMIsingModelBatch(J=torch.tensor([1.0, 2.0, 3.0]),
                                  E=torch.tensor([0.9, 1.1]), beta=30.0, theta=1.0,
                                  tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
                                   hazard="synchronous")
    batch.forward(200, pb=False)

    for b in (0, 3, batch.B - 1):
        one = batch.select(b)
        assert one.Qm == batch.Qm
        assert torch.equal(one.R[0], batch.R[0][b])
        assert torch.equal(one.m, batch.m[b])
        assert torch.allclose(one.fp_residual(one.m), batch.fp_residual(batch.m)[b],
                              rtol=0, atol=1e-14)

    batch.update()
    for b in (0, 3, batch.B - 1):
        one = RDMIsingModelBatch(J=torch.tensor([1.0, 2.0, 3.0]),
                                    E=torch.tensor([0.9, 1.1]), beta=30.0, theta=1.0,
                                    tau_int=20.0, tau_ref=3.0, K_ref=0.0, dt=DT,
                                   hazard="synchronous")
        one.forward(200, pb=False)
        sel = one.select(b)
        sel.update()
        assert torch.allclose(sel.m, batch.m[b], rtol=0, atol=1e-14)
        assert torch.allclose(sel.p[0], batch.p[0][b], rtol=0, atol=1e-14)
        assert torch.allclose(sel.S[0], batch.S[0][b], rtol=0, atol=1e-14)
