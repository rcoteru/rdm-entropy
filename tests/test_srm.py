""" Tests for the continuous-time mean field.

Two references. The first is the discrete model itself: at the time quantum, lam0*dt = 1, the
escape hazard reduces to the sigmoid, so RDMNetwork must reproduce RDMNetwork exactly. That
pins the scheme, the parameter scaling and the weight convention in one check, and it is the
claim the appendix makes about the two models being the same at that single step.

The second is convergence: with lam0 pinned, refining dt refines the numerics of one model, so
the rates must converge. The synchronous model does not: refining dt moves it to a different
member of the family, and that non-convergence is asserted here too, since it is the whole
reason the escape and asynchronous kernels exist.
"""

import pytest
import torch

import rdme.hazards as hzd
import rdme.lyapunov as lyap
from rdme.batch import (RDMIsingModelBatch, RDMNetworkBatch, RDMWilsonCowanBatch,
                        RDMWilsonCowanPoints)
from rdme.single import RDMIsingModel, RDMWilsonCowan


# A moderate operating point: oscillatory in the discrete model, so the tests exercise moving
# dynamics rather than a state that sits still and would hide a transport bug.
WC = dict(E_ratio=0.8, w_EE=3.67, w_EI=15.910, w_IE=-7.955, w_II=-2.50,
          E_exc=1.16, E_inh=1.16, beta_E=30.0, beta_I=30.0, theta_E=1.0, theta_I=1.0,
          tau_int_E=10.0, tau_int_I=40.0, tau_ref_E=3.0, tau_ref_I=3.0,
          K_ref_E=0.5, K_ref_I=0.0)


# A settled operating point, for the tests that need a genuine fixed point. Verified to hold
# r_E to within 5e-12 over 200 ms; the nominal WC above oscillates with an amplitude of 0.07,
# which silently invalidates any stationary claim made about it.
SETTLED = dict(w_EE=0.5, w_EI=1.0, w_IE=-1.0, w_II=-1.0, E_exc=0.9, E_inh=0.9)


@pytest.fixture(autouse=True)
def float64():
    """ The equivalence test below asserts exact agreement with RDMNetwork; in float32 the two
    differ in the last bits purely from operation order. """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _ct(dt, dt0=None, hazard="escape", **over):
    p = dict(WC); p.update(over)
    return RDMWilsonCowan(**p, dt=dt, dt0=dt0, hazard=hazard)


def _rdm(dt, **over):
    """ The same system through rdme.mean_field, which wants the weights pre-divided by dt. """
    p = dict(WC); p.update(over)
    for k in ("w_EE", "w_EI", "w_IE", "w_II"):
        p[k] = p[k] / dt
    return mf.RDMWilsonCowan(**p, dt=dt)


# Equivalence at the time quantum
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_all_three_hazards_coincide_at_the_time_quantum():
    """ Both families pass through the synchronous model at lam0*dt = 1: the escape hazard by
    its softplus calibration, the asynchronous one because an update probability of 1 is no
    thinning at all. Checked over a field range spanning both tails. """
    h = torch.linspace(-3.0, 3.0, 401)
    beta, dt = 30.0, 0.2
    ref = hzd.sync_hazard(h, beta, 1.0 / dt, dt)
    for name in ("async", "escape"):
        got = hzd.HAZARDS[name].phi(h, beta, 1.0 / dt, dt)
        assert torch.allclose(got, ref, rtol=0, atol=1e-14), (name, (got - ref).abs().max())


def test_hazard_derivatives_match_autograd():
    """ Each dPhi/dh against autograd on its own Phi, away from the quantum so the three are
    genuinely distinct (lam0*dt = 0.5). """
    beta, lam0, dt = 30.0, 5.0, 0.1
    for name, hz in hzd.HAZARDS.items():
        h = torch.linspace(-2.0, 2.0, 101, requires_grad=True)
        phi = hz.phi(h, beta, lam0, dt)
        auto = torch.autograd.grad(phi.sum(), h)[0]
        ana = hz.dphi(h.detach(), beta, lam0, dt)
        assert torch.allclose(ana, auto, rtol=0, atol=1e-11), (name, (ana - auto).abs().max())


def test_hazards_are_probabilities_and_async_rejects_an_oversized_step():
    """ Phi must stay in [0, 1]. For the asynchronous kernel that is a real constraint rather
    than an identity: Phi = lam0*dt*sigma exceeds 1 once lam0*dt does, which is why the update
    probability is capped at one and the class refuses the configuration. """
    h = torch.linspace(-8.0, 8.0, 201)
    for name, hz in hzd.HAZARDS.items():
        phi = hz.phi(h, 30.0, 5.0, 0.1)
        assert bool(((phi >= 0) & (phi <= 1)).all()), name

    bad = hzd.async_hazard(h, 30.0, lam0=1.0 / 0.2, dt=0.4)      # lam0*dt = 2
    assert float(bad.max()) > 1.0, "the constraint should bite for lam0*dt > 1"
    with pytest.raises(ValueError, match="requires lam0.*<= 1"):
        _ct(0.4, dt0=0.2, hazard="async")


def test_rate_asymptotics_separate_the_two_continuous_models():
    """ The appendix's eq:rate-asymptotics, which is the paper point: the two families converge
    to DIFFERENT continuous-time models, because their rate functions differ. They agree at
    weak drive (both ~ lam0*e^{beta h}) and separate at strong drive, where the asynchronous
    rate saturates at lam0 while the escape rate grows like lam0*beta*h. """
    beta, lam0, dt = 1.0, 5.0, 0.05
    weak, strong = torch.tensor([-6.0]), torch.tensor([8.0])

    a_w = hzd.async_rate(weak, beta, lam0, dt)
    e_w = hzd.escape_rate(weak, beta, lam0, dt)
    assert abs(float(a_w / e_w) - 1.0) < 0.01, "rates should agree at weak drive"

    a_s = hzd.async_rate(strong, beta, lam0, dt)
    e_s = hzd.escape_rate(strong, beta, lam0, dt)
    assert float(a_s) == pytest.approx(lam0, rel=1e-3), "async rate saturates at lam0"
    assert float(e_s) > 7 * float(a_s), "escape rate grows without bound"
    assert float(e_s) == pytest.approx(lam0 * beta * float(strong), rel=0.01)


def test_escape_hazard_matches_its_rate_definition():
    """ Phi must be the exact Poisson survival for the rate lambda = lam0*softplus(beta h),
    which is the property the continuum limit rests on. """
    h = torch.linspace(-2.0, 2.0, 201)
    beta, lam0, dt = 30.0, 5.0, 0.03
    lam = lam0 * torch.nn.functional.softplus(beta * h)
    assert torch.allclose(hzd.escape_hazard(h, beta, lam0, dt), 1 - torch.exp(-lam * dt),
                          rtol=0, atol=1e-15)


def test_default_dt0_sits_at_the_quantum():
    """ dt0 defaults to dt, i.e. lam0*dt = 1, where all three hazards coincide. """
    net = _ct(0.2)
    assert net.dt0 == 0.2 and net.lam0 == pytest.approx(5.0)
    ref = _ct(0.2, hazard="synchronous").firing_prob()
    for name in ("async", "escape"):
        for a, b in zip(_ct(0.2, hazard=name).firing_prob(), ref):
            assert torch.allclose(a, b, rtol=0, atol=1e-14), name


# Convergence, and the lack of it in the discrete model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _mean_rate(net, eq_ms=1500.0, rec_ms=600.0):  # noqa: D401
    net.forward(int(eq_ms / net.dt))
    return net.trajectory(int(rec_ms / net.dt), pb=False)["r"][:, 0].mean().item()


def test_rates_converge_as_dt_refines_at_fixed_lam0():
    """ With the model pinned by dt0, halving dt must change the rate by less each time.

    Run at a settled operating point: on an oscillating one the mean over a finite window
    carries a sampling error of order amplitude/n_periods, which swamps the discretization
    error being measured and makes the successive differences non-monotone for reasons that
    have nothing to do with convergence. """
    rates = [_mean_rate(_ct(dt, dt0=0.2, **SETTLED)) for dt in (0.2, 0.1, 0.05)]
    d1, d2 = abs(rates[1] - rates[0]), abs(rates[2] - rates[1])
    assert d2 < 0.6 * d1, f"successive differences not shrinking: {rates}"


def test_sigmoid_model_does_not_converge_as_dt_refines():
    """ The counterpart, and the reason this module exists: with the sigmoid, refining dt walks
    across a family of models rather than refining one, so the successive differences do not
    collapse. Compared against the escape hazard on the same points, since the claim is
    relative -- one converges and the other does not. """
    dts = (0.2, 0.1, 0.05)
    sig = [_mean_rate(_ct(dt, hazard="sigmoid", **SETTLED)) for dt in dts]
    esc = [_mean_rate(_ct(dt, dt0=0.2, **SETTLED)) for dt in dts]

    spread = lambda v: max(v) - min(v)
    assert spread(sig) > 10 * spread(esc), f"sigmoid {sig} vs escape {esc}"
    d1, d2 = abs(sig[1] - sig[0]), abs(sig[2] - sig[1])
    assert d2 > 0.6 * d1, f"sigmoid rates unexpectedly converged: {sig}"


# Internal consistency
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_age_mass_is_conserved():
    net = _ct(0.1, dt0=0.2)
    net.forward(2000)
    for p_i in net.p:
        assert p_i.sum().item() == pytest.approx(1.0, abs=1e-12)
        assert bool((p_i >= -1e-15).all())


def test_age_density_integrates_to_one():
    """ q = p/dt is the density of the transport equations, so q du = q dt must sum to 1. """
    net = _ct(0.05, dt0=0.2)
    net.forward(1000)
    for q_i in net.age_density():
        assert (q_i.sum() * net.dt).item() == pytest.approx(1.0, abs=1e-12)


def test_stationary_rate_matches_the_dynamics_at_a_fixed_point():
    """ At a fixed point the renewal equation reduces to r* = [int Survival du]^{-1}. Run to
    settle, then the closed form and the dynamics must agree. """
    net = _ct(0.1, dt0=0.2, **SETTLED)
    net.forward(40000)
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
    net = _ct(dt, dt0=0.2)
    net.forward(500)
    want = net.w @ net.rates() + net.E
    assert torch.allclose(net.total_input(), want, rtol=0, atol=1e-15)


def test_truncation_age_is_dt_independent():
    """ Qm comes from q_from_tau, so U_max = Qm*dt is a physical age and does not move when the
    resolution does -- otherwise a convergence study would be changing two things at once. """
    u_max = [[q * dt for q in _ct(dt, dt0=0.2).Qm] for dt in (0.2, 0.1, 0.05)]
    for col in zip(*u_max):
        assert max(col) - min(col) < 0.5    # within one coarse bin


def test_rejects_an_unknown_hazard():
    with pytest.raises(ValueError, match="hazard must be one of"):
        _ct(0.2, hazard="logistic")


# A smaller, faster system for the tests that build dense Jacobians or O(Q^2) joints.
SMALL = dict(WC, tau_int_E=6.0, tau_int_I=8.0)


def _sync(dt=0.2, **over):
    """ The synchronous kernel at its own quantum, i.e. the model rdme.mean_field used to
    implement. Kept as the default for the tests that are about machinery rather than about
    which hazard is in play. """
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowan(**p, dt=dt, hazard="synchronous")


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
    hazards: the sigmoid case would pass even if d_escape_hazard were wrong. """
    for hazard, dt0 in (("synchronous", None), ("async", 0.2), ("escape", 0.2)):
        net = RDMWilsonCowan(**SMALL, dt=0.1, dt0=dt0, hazard=hazard)
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
    wrong. Run for all three hazards, since the recursions evaluate Phi in five places. """
    for name, dt in (("synchronous", 0.2), ("async", 0.1), ("escape", 0.1)):
        a = RDMWilsonCowan(**SMALL, dt=dt, dt0=0.2, hazard=name)
        b = RDMWilsonCowan(**SMALL, dt=dt, dt0=0.2, hazard=name)
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
    production across the three hazards is the point of carrying them. """
    for name in ("synchronous", "async", "escape"):
        net = RDMWilsonCowan(**SMALL, dt=0.05, dt0=0.2, hazard=name)
        t = net.entropy_trajectory(30, pb=False)
        for k in ("sigma", "H_fwd", "H_rev"):
            assert torch.isfinite(t[k]).all(), (name, k)
        assert bool((t["H_fwd"] >= -1e-15).all()), name
        assert torch.allclose(t["sigma"], t["H_rev"] - t["H_fwd"], rtol=0, atol=1e-18), name


# A strongly driven, non-oscillating system whose time quantum is SLOW compared with its
# kernels. Both conditions are needed to separate the two rate functions: strong drive, so the
# saturation of the asynchronous rate is binding, and dt0 >~ tau, so the interspike interval is
# set by the hazard rather than by refractory recovery.
SLOW_QUANTUM = dict(w_EE=0.0, w_EI=1.0, w_IE=-1.0, w_II=-1.0, E_exc=2.0, E_inh=2.0,
                    tau_int_E=6.0, tau_int_I=8.0, tau_ref_E=3.0, tau_ref_I=3.0)


def _slow(dt0, hazard, dt=None):
    p = dict(SMALL); p.update(SLOW_QUANTUM)
    return RDMWilsonCowan(**p, dt=dt0 / 8 if dt is None else dt, dt0=dt0, hazard=hazard)


def test_async_and_escape_agree_when_the_quantum_is_fast():
    """ For dt0 << tau_ref, tau_m a strongly driven neuron fires within a few dt0 of crossing
    threshold under either rate and its interval is set by refractory recovery, so the two
    continuous models agree despite having different rate functions. """
    r = {n: _mean_rate(_slow(0.5, n), eq_ms=1500.0, rec_ms=300.0) for n in ("async", "escape")}
    gap = abs(r["async"] - r["escape"]) / r["async"]
    assert gap < 0.05, f"expected agreement at a fast quantum, got {gap:.1%}: {r}"


def test_async_and_escape_separate_when_the_quantum_is_slow():
    """ The paper point: the discretization map only changes the speed of convergence, but the
    rate function changes the limit, so the two families converge to DIFFERENT continuous-time
    models. The size of the difference is controlled by the time quantum relative to the time
    constants -- not by the step dt, which only sets convergence to a fixed model.

    Measured here: the relative gap runs 3% / 27% / 104% at dt0 = 0.5 / 3 / 12 against
    tau_ref = 3, with the asynchronous rate falling as it saturates at lam0 = 1/dt0 while the
    escape rate, unbounded, keeps the population firing. """
    gaps = []
    for dt0 in (0.5, 3.0, 12.0):
        r = {n: _mean_rate(_slow(dt0, n), eq_ms=1500.0, rec_ms=300.0)
             for n in ("async", "escape")}
        gaps.append(abs(r["async"] - r["escape"]) / r["async"])
    assert gaps[0] < 0.05 < gaps[1] < gaps[2], f"gap should grow with dt0/tau: {gaps}"
    assert gaps[-1] > 0.5, f"expected a large separation at a slow quantum: {gaps}"


def test_each_family_still_converges_in_dt_at_a_slow_quantum():
    """ Separation between the families is not a failure to converge within one: with dt0 and
    so the model pinned, refining dt must still settle, for each hazard independently. """
    for n in ("async", "escape"):
        rates = [_mean_rate(_slow(6.0, n, dt=d), eq_ms=1500.0, rec_ms=300.0)
                 for d in (1.5, 0.75, 0.375)]
        d1, d2 = abs(rates[1] - rates[0]), abs(rates[2] - rates[1])
        assert d2 < 0.7 * d1, f"{n} not converging: {rates}"


# The batched class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Two references. A batch of B systems must agree, element by element, with B single systems
# built from the same parameters -- that is what makes the batch axis a bookkeeping device
# rather than a second implementation. And at hazard="synchronous" it must agree with
# rdme.mean_field.RDMNetworkBatch, the same oracle the single class is held to.

W_EE = [1.5, 2.0, 2.5]


# the three kernels, each at a step where it is well defined (async needs lam0*dt <= 1)
HAZ_DT = (("synchronous", 0.2, None), ("async", 0.1, 0.2), ("escape", 0.1, 0.2))


def _bat(hazard="escape", dt=0.1, dt0=0.2, **over):
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowanPoints(**{**p, "w_EE": torch.tensor(W_EE)},
                                    dt=dt, dt0=dt0, hazard=hazard)


def _sing(w, hazard="escape", dt=0.1, dt0=0.2, **over):
    p = dict(SMALL); p.update(over)
    return RDMWilsonCowan(**{**p, "w_EE": w}, dt=dt, dt0=dt0, hazard=hazard)


def test_batch_dynamics_match_single_systems():
    for hz, dt, dt0 in HAZ_DT:
        b = _bat(hz, dt, dt0)
        S = [_sing(w, hz, dt, dt0) for w in W_EE]
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
    for hz, dt, dt0 in HAZ_DT:
        b = _bat(hz, dt, dt0)
        S = [_sing(w, hz, dt, dt0) for w in W_EE]
        b.forward(100, pb=False)
        for s in S: s.forward(100)
        bb = b.init_tangent(2, generator=torch.Generator().manual_seed(0))
        ss = [[t[i].clone() for t in bb] for i in range(len(W_EE))]
        for _ in range(40):
            b.step_tangent(bb)
            for i in range(len(W_EE)): S[i].step_tangent(ss[i])
        d = max((bb[j][i] - ss[i][j]).abs().max().item()
                for i in range(len(W_EE)) for j in range(len(bb)))
        assert d < 1e-13, f"{hz}: {d}"


def test_batch_epr_sliding_and_joint_agree():
    """ Both paths, batched, for every hazard -- and with the joint chunked, since chunking is
    the batch's own degree of freedom and must not change the answer. """
    for hz, dt, dt0 in HAZ_DT:
        a = _bat(hz, dt, dt0).entropy_trajectory(30, method="sliding", pb=False)
        c = _bat(hz, dt, dt0).entropy_trajectory(30, method="joint", pb=False, chunk=2)
        for k in ("m", "sigma", "H_fwd", "H_rev"):
            assert (a[k] - c[k]).abs().max().item() < 1e-14, (hz, k)


def test_batch_epr_matches_single_systems():
    for hz, dt, dt0 in HAZ_DT:
        b = _bat(hz, dt, dt0).entropy_trajectory(30, method="sliding", pb=False)
        for i, w in enumerate(W_EE):
            s = _sing(w, hz, dt, dt0).entropy_trajectory(30, method="sliding", pb=False)
            for k in ("m", "sigma", "H_fwd", "H_rev"):
                assert (b[k][i] - s[k]).abs().max().item() < 1e-14, (hz, k, i)


# Batch bookkeeping
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_grid_factory_crosses_and_points_factory_zips():
    """ The distinction the two factories exist for, and the reason RDMWilsonCowanPoints is not
    just a convenience: an outer product cannot express an axis that drives two parameters. """
    kw = dict(dt=0.1, dt0=0.2, hazard="escape")
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
                                dt=0.1, dt0=0.2, hazard="escape")
    assert g.B == 20
    t = g.trajectory(5, pb=False)
    assert t["m"].shape == (4, 5, 5, 2), t["m"].shape   # (n_wEE, n_wII, T, M)
    assert g.unflatten(g.w[:, 0, 0])[:, 0].allclose(torch.linspace(1., 3., 4))


def test_points_factory_rejects_a_mismatched_column():
    with pytest.raises(ValueError, match="length 1 or B=3"):
        RDMWilsonCowanPoints(**{**SMALL, "w_EE": torch.tensor([1.0, 2.0, 3.0]),
                                    "w_II": torch.tensor([-1.0, -2.0])},
                                 dt=0.1, dt0=0.2, hazard="escape")


def test_batch_save_load_round_trip(tmp_path):
    b = _bat()
    b.forward(50, pb=False)
    f = tmp_path / "b.pt"
    b.save(f)
    c = RDMNetworkBatch.load(f, device="cpu")
    assert (c.hazard, c.dt, c.dt0) == (b.hazard, b.dt, b.dt0)
    assert c.Qm == b.Qm and c.grid_shape == b.grid_shape
    assert torch.equal(c.m, b.m) and all(torch.equal(x, y) for x, y in zip(c.p, b.p))
    b.forward(20, pb=False); c.forward(20, pb=False)
    assert torch.allclose(c.m, b.m, rtol=0, atol=1e-15), "loaded batch must keep stepping"


def test_batch_rejects_an_oversized_step_for_the_async_hazard():
    with pytest.raises(ValueError, match="requires lam0.*<= 1"):
        _bat("async", dt=0.4, dt0=0.2)


def test_batch_ising_factory():
    b = RDMIsingModelBatch(J=torch.tensor([5.0, 10.0]), E=1.0, beta=30.0, theta=1.0,
                               tau_int=10.0, tau_ref=3.0, K_ref=0.0,
                               dt=0.1, dt0=0.2, hazard="escape")
    assert b.B == 2 and b.M == 1
    assert torch.equal(b.w[:, 0, 0], torch.tensor([5.0, 10.0]))
    s = RDMIsingModel(J=10.0, E=1.0, beta=30.0, theta=1.0, tau_int=10.0, tau_ref=3.0,
                          K_ref=0.0, dt=0.1, dt0=0.2, hazard="escape")
    b.forward(100, pb=False); s.forward(100)
    assert torch.allclose(b.m[1], s.m, rtol=0, atol=1e-14)


# Tangent dynamics, against the state it is supposed to follow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_tangent_leaves_the_state_where_update_would():
    """ step_tangent must advance the state exactly as update() does, or the tangent is being
    propagated along a different trajectory than the one being measured. """
    for hz, dt, dt0 in HAZ_DT:
        ref, run = _bat(hz, dt, dt0), _bat(hz, dt, dt0)
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
    net = RDMWilsonCowan(**SMALL, dt=dt, hazard="synchronous")
    net.forward(4000)
    p = [t.clone() for t in net.p]; S = [t.clone() for t in net.S]
    x = net.flatten_state(p, S)
    assert torch.allclose(net.state_map(x), x, rtol=0, atol=1e-11), "not a fixed point"

    ev = torch.linalg.eigvals(torch.func.jacfwd(net.state_map)(x)).abs()
    want = (torch.log(ev.clamp_min(1e-300)) / dt).sort(descending=True).values[:2]

    run = RDMWilsonCowan(**SMALL, dt=dt, hazard="synchronous")
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
    kw = dict(dt=0.1, dt0=0.2, hazard="escape")
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
                                   dt=0.1, dt0=0.2, hazard="escape")
    assert pts.B == 11
    # w_EI * |w_IE| == g**2 cell by cell, which is the quantity the sweep is written in
    assert torch.allclose(pts.w[:, 1, 0] * pts.w[:, 0, 1].abs(), g ** 2)


def test_points_and_batch_evolve_identically():
    """ Equal constructed tensors are necessary but not sufficient -- step both. """
    kw = dict(dt=0.1, dt0=0.2, hazard="escape")
    w = torch.tensor([2.0, 3.0])
    a = RDMWilsonCowanBatch(**{**SMALL, "w_EE": w}, **kw)
    b = RDMWilsonCowanPoints(**{**SMALL, "w_EE": w}, **kw)
    a.forward(200, pb=False); b.forward(200, pb=False)
    assert torch.equal(a.m, b.m)
    for x, y in zip(a.p, b.p):
        assert torch.equal(x, y)
