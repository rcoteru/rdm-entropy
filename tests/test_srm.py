""" Tests for the mean field: the scheme, its two classes, and the machinery on top of them.

The kernels themselves are tested in test_hazards.py; what is checked here is the MODEL built
on one. Three references, none of them a stored number:

  a second representation -- a batch of B systems against B single systems, the sliding EPR
  recursions against the explicit joint, the tangent map against autodiff on the state map it
  linearizes;

  an invariant of the scheme -- age mass conserved, the density integrating to one, the input
  built from rates, a fixed point not moving;

  convergence -- with the model pinned in physical units, refining dt must settle. Under route
  B that is the only reporting rule there is, so it is asserted directly rather than against a
  discrete model that no longer exists.
"""

import math

import pytest
import torch

import rdme.hazards as hzd
import rdme.lyapunov as lyap
from rdme.rdm_batch import (RDMIsingModelBatch, RDMNetworkBatch, RDMWilsonCowanBatch,
                        RDMWilsonCowanPoints)
from rdme.rdm_single import RDMIsingModel, RDMWilsonCowan


# A moderate operating point: oscillatory, so the tests exercise moving dynamics rather than a
# state that sits still and would hide a transport bug.
WC = dict(E_ratio=0.8, w_EE=3.67, w_EI=15.910, w_IE=-7.955, w_II=-2.50,
          E_exc=1.16, E_inh=1.16, beta_E=30.0, beta_I=30.0, theta_E=1.0, theta_I=1.0,
          tau_int_E=10.0, tau_int_I=40.0, tau_ref_E=3.0, tau_ref_I=3.0,
          K_ref_E=0.5, K_ref_I=0.0)


# A settled operating point, for the tests that need a genuine fixed point. The nominal WC
# above oscillates, which silently invalidates any stationary claim made about it. The drive
# is 1.0 rather than the 0.9 of the route-A suite: at 0.9 the cell fires at 0.017/ms, a mean
# interval of 58 ms against an age grid whose truncation age is 184 ms, and the resulting 19%
# truncation error is larger than the discretization error the convergence tests measure.
# At 1.0 the truncation error is 6e-6 and the cell is still a fixed point.
SETTLED = dict(w_EE=0.5, w_EI=1.0, w_IE=-1.0, w_II=-1.0, E_exc=1.0, E_inh=1.0)

# The solver steps used below all divide t_abs = 1 ms, as kernels.n_abs_bins requires: a dead
# time that does not land on a bin boundary would move with the step.
HAZ_DT = (("poisson_exact", 0.2), ("poisson_exact", 0.1), ("poisson", 0.1),
          ("bernoulli", 0.5))


@pytest.fixture(autouse=True)
def float64():
    """ The batch-vs-single comparisons assert agreement to 1e-14; in float32 the two differ
    in the last bits purely from operation order. """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _ct(dt, hazard="poisson_exact", **over):
    p = dict(WC); p.update(over)
    return RDMWilsonCowan(**p, dt=dt, hazard=hazard)


def _mean_rate(net, eq_ms=1500.0, rec_ms=600.0):  # noqa: D401
    net.forward(int(eq_ms / net.dt))
    return net.trajectory(int(rec_ms / net.dt), pb=False)["r"][:, 0].mean().item()


# Convergence in dt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.slow
@pytest.mark.parametrize("hazard", ["poisson_exact", "poisson"])
def test_rates_converge_as_dt_refines(hazard):
    """ With the model pinned in physical units (nu0, t_abs and the couplings all in ms),
    halving dt must change the rate by less each time. That is the whole reporting rule under
    route B, and it is a property of the scheme rather than of the kernel, so both route-B
    kernels are asked for it.

    Run at a settled operating point: on an oscillating one the mean over a finite window
    carries a sampling error of order amplitude/n_periods, which swamps the discretization
    error being measured and makes the successive differences non-monotone for reasons that
    have nothing to do with convergence. """
    rates = [_mean_rate(_ct(dt, hazard=hazard, **SETTLED)) for dt in (0.2, 0.1, 0.05)]
    d1, d2 = abs(rates[1] - rates[0]), abs(rates[2] - rates[1])
    # first order, which is what the scheme claims: the only errors are the variation of rate
    # and input within a bin. The band is wide enough not to fail on the pre-asymptotic term
    # and narrow enough to catch a scheme that has stalled or gone second order by accident.
    assert 0.35 < d2 / d1 < 0.65, \
        f"{hazard}: expected first-order convergence, got ratio {d2 / d1:.3f} from {rates}"


@pytest.mark.slow
def test_the_two_route_b_kernels_converge_to_the_same_rate():
    """ One model, two discretization maps: `poisson` and `poisson_exact` must agree in the
    limit, and their gap must close with dt. This is the model-level counterpart of the
    kernel-level D5 check in test_hazards.py -- the kernels differing at O(dt) is only
    interesting if the dynamics built on them does too. """
    gaps = []
    for dt in (0.2, 0.1, 0.05):
        r = {n: _mean_rate(_ct(dt, hazard=n, **SETTLED)) for n in ("poisson", "poisson_exact")}
        gaps.append(abs(r["poisson"] - r["poisson_exact"]))
    assert gaps[2] < 0.5 * gaps[0], f"the two maps should converge together: {gaps}"


# Internal consistency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_age_mass_is_conserved():
    net = _ct(0.1)
    net.forward(2000)
    for p_i in net.p:
        assert p_i.sum().item() == pytest.approx(1.0, abs=1e-12)
        assert bool((p_i >= -1e-15).all())


def test_age_density_integrates_to_one():
    """ q = p/dt is the density of the transport equations, so q du = q dt must sum to 1. """
    net = _ct(0.05)
    net.forward(1000)
    for q_i in net.age_density():
        assert (q_i.sum() * net.dt).item() == pytest.approx(1.0, abs=1e-12)


def test_stationary_rate_matches_the_dynamics_at_a_fixed_point():
    """ At a fixed point the renewal equation reduces to r* = [int Survival du]^{-1}. Run to
    settle, then the closed form and the dynamics must agree. """
    net = _ct(0.1, **SETTLED)
    # 8000 ms: the slow mode here is tau_int_I = 40 ms and the approach is exponential, so
    # 4000 ms still leaves a 5e-10 drift per 50 ms -- above the 1e-10 the premise is asserted
    # at, which would make "a fixed point" a statement about the window rather than the state
    net.forward(80000)
    # the premise: this must actually BE a fixed point, or the relation does not apply
    before = net.rates().clone()
    net.forward(500)
    assert torch.allclose(net.rates(), before, rtol=0, atol=1e-10), "not settled"

    r_dyn, r_fp = net.rates(), net.stationary_rate()
    assert torch.allclose(r_dyn, r_fp, rtol=1e-6, atol=0), (r_dyn - r_fp).abs().max()


def test_input_is_built_from_rates_not_overlaps():
    """ The weight convention: RDMNetwork takes w unscaled and multiplies by r, RDMNetwork takes
    w/dt and multiplies by m. Pinning it because passing w/dt here out of habit would rescale
    the network silently and the dynamics would still look plausible. """
    dt = 0.1
    net = _ct(dt)
    net.forward(500)
    want = net.w @ net.rates() + net.E
    assert torch.allclose(net.total_input(), want, rtol=0, atol=1e-15)


def test_truncation_age_is_dt_independent():
    """ Qm comes from q_from_tau, so U_max = Qm*dt is a physical age and does not move when the
    resolution does -- otherwise a convergence study would be changing two things at once. """
    u_max = [[q * dt for q in _ct(dt).Qm] for dt in (0.2, 0.1, 0.05)]
    for col in zip(*u_max):
        assert max(col) - min(col) < 0.5    # within one coarse bin


def test_rejects_an_unknown_hazard():
    with pytest.raises(ValueError, match="hazard must be one of"):
        _ct(0.2, hazard="logistic")


# A smaller, faster system for the tests that build dense Jacobians or O(Q^2) joints.
SMALL = dict(WC, tau_int_E=6.0, tau_int_I=8.0)


def _sync(dt=0.2, **over):
    """ The default model for the tests that are about machinery rather than about which
    kernel is in play: route B at a coarse-but-admissible step. """
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowan(**p, dt=dt, hazard="poisson_exact")


def test_fixed_points_are_fixed_under_forward():
    """ The reference a root is actually checked against: a fixed point must be a state the
    dynamics does not move. """
    a = _sync()
    roots = a.fixed_points()
    assert roots.numel() > 0, "no fixed point found"
    for r in roots:
        a.set_state(r)
        a.forward(3000)
        assert torch.allclose(a.overlaps(), r, rtol=0, atol=1e-10), (a.overlaps(), r)


def test_tangent_matches_the_jacobian_of_its_own_state_map():
    """ The hazard-general tangent map against autodiff on the map it linearizes. Run for BOTH
    kernels: a test on one would pass even if another's dphi were wrong, and the three have
    genuinely different derivatives. """
    for hazard, dt in (("poisson_exact", 0.1), ("poisson", 0.1), ("bernoulli", 0.5)):
        net = RDMWilsonCowan(**SMALL, dt=dt, hazard=hazard)
        net.forward(137)
        x = net.flatten_state([t.clone() for t in net.p], [t.clone() for t in net.S])
        D, k = x.numel(), 3
        V = torch.randn(D, k)
        want = torch.func.jacfwd(net.state_map)(x) @ V

        blocks = []
        for i in range(net.M):
            cp, cS = [], []
            for j in range(k):
                vp, vS = net.unflatten_state(V[:, j])
                cp.append(vp[i]); cS.append(vS[i])
            blocks.append(torch.stack(cp)); blocks.append(torch.stack(cS))

        net.step_tangent(blocks)
        got = torch.stack([net.flatten_state([blocks[2*i][j] for i in range(net.M)],
                                             [blocks[2*i+1][j] for i in range(net.M)])
                           for j in range(k)], dim=-1)
        assert torch.allclose(got, want, rtol=0, atol=1e-13), \
            f"{hazard}: {(got - want).abs().max()}"


def test_entropy_is_nonnegative_and_sigma_is_the_difference():
    a = _sync()
    t = a.entropy_trajectory(40, pb=False)
    assert bool((t["H_fwd"] >= -1e-15).all()) and bool((t["H_rev"] >= -1e-15).all())
    assert torch.allclose(t["sigma"], t["H_rev"] - t["H_fwd"], rtol=0, atol=1e-18)


def test_sliding_and_joint_epr_agree_for_every_hazard():
    """ The two EPR paths against each other. They share only epr_from_parts -- the joint path
    builds the (Q, K) joint and sums it, the sliding path reaches the same four marginals by
    recursion -- so agreement checks the recursions, which is the part with something to get
    wrong. Run for both hazards, since the recursions evaluate Phi in five places. """
    for name, dt in HAZ_DT:
        a = RDMWilsonCowan(**SMALL, dt=dt, hazard=name)
        b = RDMWilsonCowan(**SMALL, dt=dt, hazard=name)
        ta = a.entropy_trajectory(40, method="sliding", pb=False)
        tb = b.entropy_trajectory(40, method="joint", pb=False)
        for k in ("m", "sigma", "H_fwd", "H_rev"):
            d = (ta[k] - tb[k]).abs().max().item()
            assert d < 1e-14, f"{name}/{k}: {d}"


def test_entropy_rejects_an_unknown_method():
    a = _sync()
    with pytest.raises(ValueError, match="method must be"):
        a.entropy_trajectory(5, method="fast", pb=False)


def test_entropy_runs_for_every_hazard():
    """ The reason the EPR was rewritten in terms of Phi rather than softplus and the logit:
    those two are identities of the logistic kernel only, and the comparison of entropy
    production across the hazards is the point of carrying them. """
    for name, dt in HAZ_DT:
        net = RDMWilsonCowan(**SMALL, dt=dt, hazard=name)
        t = net.entropy_trajectory(30, pb=False)
        for k in ("sigma", "H_fwd", "H_rev"):
            assert torch.isfinite(t[k]).all(), (name, k)
        assert bool((t["H_fwd"] >= -1e-15).all()), name
        assert torch.allclose(t["sigma"], t["H_rev"] - t["H_fwd"], rtol=0, atol=1e-18), name


# The batched class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Two references. A batch of B systems must agree, element by element, with B single systems
# built from the same parameters -- that is what makes the batch axis a bookkeeping device
# rather than a second implementation. Every kernel is run, at a step each admits.

W_EE = [1.5, 2.0, 2.5]


def _bat(hazard="poisson_exact", dt=0.1, **over):
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowanPoints(**{**p, "w_EE": torch.tensor(W_EE)}, dt=dt, hazard=hazard)


def _sing(w, hazard="poisson_exact", dt=0.1, **over):
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowan(**{**p, "w_EE": w}, dt=dt, hazard=hazard)


def test_batch_dynamics_match_single_systems():
    for hz, dt in HAZ_DT:
        b = _bat(hz, dt)
        S = [_sing(w, hz, dt) for w in W_EE]
        b.forward(200, pb=False)
        for s in S: s.forward(200)
        d = max((b.m[i] - S[i].m).abs().max().item() for i in range(len(W_EE)))
        assert d < 1e-14, f"{hz}: {d}"


def test_select_extracts_a_working_single_system():
    """ select() must hand back a model that both sits where the batch does and keeps stepping
    with it -- the second half is what catches a shared buffer or a mis-copied kernel. """
    b, S = _bat(), [_sing(w) for w in W_EE]
    b.forward(150, pb=False)
    for s in S: s.forward(150)
    sel = b.select(1)
    assert torch.allclose(sel.m, S[1].m, rtol=0, atol=1e-14)
    assert sel.Qm == b.Qm, "Qm must be copied, not re-derived from this element's taus"
    sel.forward(60); S[1].forward(60)
    assert torch.allclose(sel.m, S[1].m, rtol=0, atol=1e-14)


def test_batch_fixed_points_match_single_systems():
    b, S = _bat(), [_sing(w) for w in W_EE]
    m = torch.tensor([[0.01, 0.005]] * len(W_EE))
    g = b.fp_residual(m)
    for i in range(len(W_EE)):
        assert torch.allclose(g[i], S[i].fp_residual(m[i]), rtol=0, atol=1e-15)


def test_batch_tangent_matches_single_systems():
    for hz, dt in HAZ_DT:
        b = _bat(hz, dt)
        S = [_sing(w, hz, dt) for w in W_EE]
        b.forward(100, pb=False)
        for s in S: s.forward(100)
        bb = b.init_tangent(2, generator=torch.Generator().manual_seed(0))
        ss = [[t[i].clone() for t in bb] for i in range(len(W_EE))]
        for _ in range(40):
            b.step_tangent(bb)
            for i in range(len(W_EE)): S[i].step_tangent(ss[i])
        d = max((bb[j][i] - ss[i][j]).abs().max().item()
                for i in range(len(W_EE)) for j in range(len(bb)))
        # 1e-12 rather than the 1e-14 the state comparisons carry: 40 tangent steps amplify
        # the last-bit difference between the batch's reduction order and the single system's,
        # and the exponential route-B rate amplifies it faster than the bounded one did
        assert d < 1e-12, f"{hz}: {d}"


def test_batch_epr_sliding_and_joint_agree():
    """ Both paths, batched, for every hazard -- and with the joint chunked, since chunking is
    the batch's own degree of freedom and must not change the answer. """
    for hz, dt in HAZ_DT:
        a = _bat(hz, dt).entropy_trajectory(30, method="sliding", pb=False)
        c = _bat(hz, dt).entropy_trajectory(30, method="joint", pb=False, chunk=2)
        for k in ("m", "sigma", "H_fwd", "H_rev"):
            assert (a[k] - c[k]).abs().max().item() < 1e-14, (hz, k)


def test_batch_epr_matches_single_systems():
    for hz, dt in HAZ_DT:
        b = _bat(hz, dt).entropy_trajectory(30, method="sliding", pb=False)
        for i, w in enumerate(W_EE):
            s = _sing(w, hz, dt).entropy_trajectory(30, method="sliding", pb=False)
            for k in ("m", "sigma", "H_fwd", "H_rev"):
                assert (b[k][i] - s[k]).abs().max().item() < 1e-14, (hz, k, i)


# Batch bookkeeping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_grid_factory_crosses_and_points_factory_zips():
    """ The distinction the two factories exist for, and the reason RDMWilsonCowanPoints is not
    just a convenience: an outer product cannot express an axis that drives two parameters. """
    kw = dict(dt=0.1, hazard="poisson_exact")
    a = torch.tensor([1.0, 2.0, 3.0])
    c = torch.tensor([-1.0, -2.0, -3.0])
    cross = RDMWilsonCowanBatch(**{**SMALL, "w_EE": a, "w_II": c}, **kw)
    zipd  = RDMWilsonCowanPoints(**{**SMALL, "w_EE": a, "w_II": c}, **kw)
    assert cross.B == 9 and zipd.B == 3
    # row = target, column = source: w[:, 0, 0] is w_EE and w[:, 1, 1] is w_II
    assert torch.equal(zipd.w[:, 0, 0], a) and torch.equal(zipd.w[:, 1, 1], c)


def test_grid_axes_round_trip_through_unflatten():
    g = RDMWilsonCowanBatch(**{**SMALL, "w_EE": torch.linspace(1., 3., 4),
                                   "w_II": torch.linspace(-3., -1., 5)},
                                dt=0.1, hazard="poisson_exact")
    assert g.B == 20
    t = g.trajectory(5, pb=False)
    assert t["m"].shape == (4, 5, 5, 2), t["m"].shape   # (n_wEE, n_wII, T, M)
    assert g.unflatten(g.w[:, 0, 0])[:, 0].allclose(torch.linspace(1., 3., 4))


def test_points_factory_rejects_a_mismatched_column():
    with pytest.raises(ValueError, match="length 1 or B=3"):
        RDMWilsonCowanPoints(**{**SMALL, "w_EE": torch.tensor([1.0, 2.0, 3.0]),
                                    "w_II": torch.tensor([-1.0, -2.0])},
                                 dt=0.1, hazard="poisson_exact")


def test_batch_save_load_round_trip(tmp_path):
    b = _bat()
    b.forward(50, pb=False)
    f = tmp_path / "b.pt"
    b.save(f)
    c = RDMNetworkBatch.load(f, device="cpu")
    assert (c.hazard, c.dt, c.nu0, c.t_abs) == (b.hazard, b.dt, b.nu0, b.t_abs)
    assert c.Qm == b.Qm and c.grid_shape == b.grid_shape
    assert torch.equal(c.m, b.m) and all(torch.equal(x, y) for x, y in zip(c.p, b.p))
    b.forward(20, pb=False); c.forward(20, pb=False)
    assert torch.allclose(c.m, b.m, rtol=0, atol=1e-15), "loaded batch must keep stepping"


def test_batch_rejects_an_oversized_step_for_the_bernoulli_hazard():
    """ Phi = nu0*dt*sigma is not a probability above dt = 1/nu0, and the class refuses it at
    construction rather than returning a hazard above one. """
    with pytest.raises(ValueError, match="admits dt <="):
        _bat("bernoulli", dt=2.0)


def test_a_step_that_does_not_divide_the_dead_time_is_refused():
    """ The window would round, and a dead time that rounds moves with the solver step. """
    with pytest.raises(ValueError, match="not an integer number"):
        _bat(dt=0.3)


def test_batch_ising_factory():
    b = RDMIsingModelBatch(J=torch.tensor([5.0, 10.0]), E=1.0, beta=30.0, theta=1.0,
                               tau_int=10.0, tau_ref=3.0, K_ref=0.0,
                               dt=0.1, hazard="poisson_exact")
    assert b.B == 2 and b.M == 1
    assert torch.equal(b.w[:, 0, 0], torch.tensor([5.0, 10.0]))
    s = RDMIsingModel(J=10.0, E=1.0, beta=30.0, theta=1.0, tau_int=10.0, tau_ref=3.0,
                          K_ref=0.0, dt=0.1, hazard="poisson_exact")
    b.forward(100, pb=False); s.forward(100)
    assert torch.allclose(b.m[1], s.m, rtol=0, atol=1e-14)


# Tangent dynamics, against the state it is supposed to follow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_tangent_leaves_the_state_where_update_would():
    """ step_tangent must advance the state exactly as update() does, or the tangent is being
    propagated along a different trajectory than the one being measured. """
    for hz, dt in HAZ_DT:
        ref, run = _bat(hz, dt), _bat(hz, dt)
        ref.forward(50, pb=False); run.forward(50, pb=False)
        blocks = run.init_tangent(2, generator=torch.Generator().manual_seed(0))
        for _ in range(30):
            ref.update()
            run.step_tangent(blocks)
        assert torch.equal(run.m, ref.m), hz
        for a, c in zip(run.p, ref.p): assert torch.equal(a, c)
        for a, c in zip(run.S, ref.S): assert torch.equal(a, c)


def test_tangent_preserves_the_sum_zero_subspace():
    """ The age distributions are normalized, so physical perturbations carry no net mass. The
    map enforces that whatever comes in, which is why the unphysical modes never come back. """
    run = _bat()
    run.forward(50, pb=False)
    blocks = [torch.randn(run.B, 3, q) for q in run.Qm for _ in range(2)]   # NOT centred
    before = [blocks[2 * i].sum(dim=-1).abs().max().item() for i in range(run.M)]
    run.step_tangent(blocks)
    after = [blocks[2 * i].sum(dim=-1).abs().max().item() for i in range(run.M)]
    assert min(before) > 0.1, "the input must violate the constraint, or this proves nothing"
    assert max(after) < 1e-12, after


def test_init_tangent_is_orthonormal_and_sum_zero():
    run = _bat()
    blocks = run.init_tangent(4, generator=torch.Generator().manual_seed(0))
    for i in range(run.M):
        assert blocks[2 * i].sum(dim=-1).abs().max() < 1e-12
    gram = sum(torch.einsum('bia,bja->bij', b, b) for b in blocks)      # (B, k, k)
    assert torch.allclose(gram, torch.eye(4).expand_as(gram), rtol=0, atol=1e-12)


def test_spectrum_matches_fp_eigenvalues_at_a_fixed_point():
    """ The only analytic answer the model itself supplies, and the one that survives
    rdme.mean_field: sitting exactly on a fixed point, the exponents are log|eig(J)|/dt for the
    leading directions. Exercises the tangent map, the driver, the orthonormalization and the
    units together. """
    dt = 0.2
    net = RDMWilsonCowan(**SMALL, dt=dt, hazard="poisson_exact")
    net.forward(4000)
    p = [t.clone() for t in net.p]; S = [t.clone() for t in net.S]
    x = net.flatten_state(p, S)
    assert torch.allclose(net.state_map(x), x, rtol=0, atol=1e-11), "not a fixed point"

    ev = torch.linalg.eigvals(torch.func.jacfwd(net.state_map)(x)).abs()
    want = (torch.log(ev.clamp_min(1e-300)) / dt).sort(descending=True).values[:2]

    run = RDMWilsonCowan(**SMALL, dt=dt, hazard="poisson_exact")
    run.p = [t.clone() for t in p]; run.S = [t.clone() for t in S]
    run.m = torch.stack([t[..., 0] for t in run.p], dim=-1)

    blocks = run.init_tangent(2, generator=torch.Generator().manual_seed(0))
    run.step_tangent(blocks)                 # kill the unphysical directions first
    lyap.orthonormalize(blocks)
    out = lyap.benettin(run.step_tangent, blocks, n_steps=3000, dt=dt,
                        cadence=20, n_blocks=10, warmup=2000)

    # The leading pair is complex-conjugate, hence EXACTLY degenerate: QR fixes the plane they
    # span but not the basis inside it, so the individual exponents are defined only up to that
    # rotation and come back split by ~4e-4 here. Their sum is not, so that is what carries the
    # tight tolerance -- the same reason the sweep reports partial sums alongside the exponents,
    # and asserting the individual values would be contradicting the degenerate flag below.
    assert out.partial[1] == pytest.approx(want.sum().item(), abs=1e-4)
    assert out.lam.mean().item() == pytest.approx(want.mean().item(), abs=1e-4)
    assert bool(out.degenerate[0]), "a conjugate pair should be flagged as unresolved"
    assert abs(float(want[0] - want[1])) < 1e-9, "the reference pair should be degenerate too"


# Points vs Batch, the remaining distinctions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_points_matches_batch_on_a_single_axis():
    """ One swept axis is the case both factories can express; they must agree exactly. """
    kw = dict(dt=0.1, hazard="poisson_exact")
    w = torch.linspace(1.0, 4.0, 7)
    a = RDMWilsonCowanBatch(**{**SMALL, "w_EE": w}, **kw)
    b = RDMWilsonCowanPoints(**{**SMALL, "w_EE": w}, **kw)
    assert a.B == b.B == 7
    for name in ("w", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "alpha_int"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for x, y in zip(a.R, b.R):
        assert torch.equal(x, y)


def test_points_expresses_a_coupled_axis():
    """ Why the zipped factory exists: a loop-gain axis drives two weights at once, which an
    outer product cannot express -- it would return the n**2 cross product instead. """
    g, rho = torch.linspace(0.0, 15.0, 11), 2.0 ** 0.5
    pts = RDMWilsonCowanPoints(**{**SMALL, "w_EI": g * rho, "w_IE": -g / rho},
                                   dt=0.1, hazard="poisson_exact")
    assert pts.B == 11
    # w_EI * |w_IE| == g**2 cell by cell, which is the quantity the sweep is written in
    assert torch.allclose(pts.w[:, 1, 0] * pts.w[:, 0, 1].abs(), g ** 2)


def test_points_and_batch_evolve_identically():
    """ Equal constructed tensors are necessary but not sufficient -- step both. """
    kw = dict(dt=0.1, hazard="poisson_exact")
    w = torch.tensor([2.0, 3.0])
    a = RDMWilsonCowanBatch(**{**SMALL, "w_EE": w}, **kw)
    b = RDMWilsonCowanPoints(**{**SMALL, "w_EE": w}, **kw)
    a.forward(200, pb=False); b.forward(200, pb=False)
    assert torch.equal(a.m, b.m)
    for x, y in zip(a.p, b.p):
        assert torch.equal(x, y)


# Reparametrization invariance, at the level of the model (D2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("delta", [-0.2, 0.3])
def test_the_model_is_invariant_under_the_rate_threshold_reparametrization(delta):
    """ (nu0, theta) -> (nu0 e^{beta delta}, theta + delta) is an exact symmetry of the
    poisson_exact kernel, so the two models must produce the SAME trajectory -- not merely
    the same stationary rate. Only nu0 e^{-beta theta} is identifiable, which is why nu0 is
    carried as a fixed unit rather than fitted.

    The symmetry acts on the FIELD, not on the kernels: h = S + R - theta drops by delta in
    every bin and the rate unit makes it back, nu0 e^{beta(h - delta)} e^{beta delta} =
    nu0 e^{beta h}. The one place it is not exact is the absolute-refractory floor, which is a
    MARGIN rather than part of the model: R_abs = S_max - theta + margin/beta shifts with
    theta, so the blocked field is left where it was and the blocked Phi picks up the factor
    e^{beta delta} that nothing cancels. At margin = 40 that is e^{-40} against e^{-40+1.8},
    i.e. 4e-18 against 2e-17, which is below every tolerance here and is asserted as such
    rather than waved past. """
    beta, nu0, dt = 6.0, 1.0, 0.25
    P = dict(J=10.0, E=0.0, beta=beta, tau_int=15.0, tau_ref=15.0, K_ref=15.0,
             dt=dt, t_abs=1.0, hazard="poisson_exact", Qm=[160])
    a = RDMIsingModel(**P, theta=0.5, nu0=nu0)
    b = RDMIsingModel(**P, theta=0.5 + delta, nu0=nu0 * math.exp(beta * delta))

    n = a.n_abs
    fa, fb = a.field()[0], b.field()[0]
    assert torch.allclose(fb[n:] - fa[n:], torch.full_like(fa[n:], -delta), rtol=0, atol=1e-12)
    assert max(float(a.firing_prob()[0][:n].max()),
               float(b.firing_prob()[0][:n].max())) < 1e-15, "the window is not suppressing"
    for _ in range(400):
        a.update(); b.update()
        assert torch.allclose(a.rates(), b.rates(), rtol=1e-10, atol=1e-14), \
            (float(a.rates()), float(b.rates()))


def test_the_reparametrization_also_leaves_sigma_alone():
    """ The entropy production is a functional of Phi, so an exact symmetry of Phi is an exact
    symmetry of sigma. Asserted separately because sigma is a difference of two large,
    nearly-cancelling entropies: a symmetry that held to 1e-10 in the rate could still move
    the fourth digit of sigma. """
    beta, delta, dt = 6.0, 0.3, 0.25
    P = dict(J=10.0, E=0.0, beta=beta, tau_int=15.0, tau_ref=15.0, K_ref=15.0,
             dt=dt, t_abs=1.0, hazard="poisson_exact", Qm=[160])
    a = RDMIsingModel(**P, theta=0.5, nu0=1.0)
    b = RDMIsingModel(**P, theta=0.5 + delta, nu0=math.exp(beta * delta))
    a.forward(2000); b.forward(2000)
    ta = a.entropy_trajectory(60, pb=False)
    tb = b.entropy_trajectory(60, pb=False)
    for k in ("m", "sigma", "H_fwd", "H_rev"):
        assert torch.allclose(ta[k], tb[k], rtol=1e-9, atol=1e-14), \
            (k, (ta[k] - tb[k]).abs().max().item())
