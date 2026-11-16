""" Kernel-level tests: the three per-bin hazards, their branches, and the step guards.

The model is route B -- a maximum-caliber tilt of a Poisson reference of rate nu0 -- realised
by two kernels, `poisson` (MaxCal at every bin) and `poisson_exact` (exact survival for the
same continuum). `bernoulli` is route A, kept for comparison, and is tested here only against
closed forms of its own.

Every oracle below is a closed form or a second representation of the same object. Nothing
here reads a cache, and nothing asserts a number that was measured rather than derived.
"""

import math

import pytest
import torch

import rdme.hazards as hzd


ROUTE_B = ("poisson", "poisson_exact")
ALL = ("bernoulli", "poisson", "poisson_exact")

# (nu0, dt) pairs every kernel admits: nu0*dt < 1 strictly, so `poisson`'s reference log-odds
# are finite and `bernoulli`'s Phi is a probability.
STEPS = [(1.0, 0.5), (1.0, 0.25), (0.3, 0.5), (2.0, 0.25), (1.0, 0.03125)]


@pytest.fixture(autouse=True)
def float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# The registry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_the_registry_is_exactly_the_three_kernels():
    """ A hard rename with no aliases (2026-11-07): `async`, `escape` and `exp_surv` are gone,
    not deprecated, so a script carrying an old name fails loudly instead of silently picking
    up a neighbour. """
    assert sorted(hzd.HAZARDS) == ["bernoulli", "poisson", "poisson_exact"]
    for gone in ("async", "escape", "exp_surv", "sync", "sigmoid"):
        assert gone not in hzd.HAZARDS


def test_the_default_hazard_is_the_exact_survival_solver():
    """ The dt -> 0 limit is taken with the kernel whose convergence is monotone. """
    assert hzd.DEFAULT_HAZARD == "poisson_exact"


# Shared contracts
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("name", ALL)
def test_hazards_are_probabilities(name):
    h = torch.linspace(-8.0, 8.0, 201)
    phi = hzd.HAZARDS[name].phi(h, 30.0, 1.0, 0.1)
    assert bool(((phi >= 0) & (phi <= 1)).all()), name


@pytest.mark.parametrize("name", ALL)
def test_hazard_derivatives_match_autograd(name):
    """ Each dPhi/dh against autograd on its own Phi. """
    beta, nu0, dt = 30.0, 1.0, 0.1
    h = torch.linspace(-2.0, 2.0, 101, requires_grad=True)
    hz = hzd.HAZARDS[name]
    phi = hz.phi(h, beta, nu0, dt)
    auto = torch.autograd.grad(phi.sum(), h)[0]
    ana = hz.dphi(h.detach(), beta, nu0, dt)
    assert torch.allclose(ana, auto, rtol=0, atol=1e-11), (name, (ana - auto).abs().max())


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("nu0,dt", STEPS)
def test_inplace_branches_are_bitwise_identical(name, nu0, dt):
    """ The contract every kernel carries: the in-place branch is the out-of-place arithmetic
    in the same order with the temporaries folded away, not merely something close. Over a
    field reaching the saturated and silent tails, where the closed log forms are the whole
    reason they are carried separately. """
    hz = hzd.HAZARDS[name]
    h = torch.cat([torch.linspace(-20.0, 20.0, 97), torch.tensor([-60.0, 60.0])])
    beta = 3.0
    for fn in (hz.phi, hz.dphi, hz.rate, hz.logphi, hz.log1m_phi):
        ref = fn(h.clone(), beta, nu0, dt)
        got = fn(h.clone(), beta, nu0, dt, inplace=True)
        assert torch.equal(ref.nan_to_num(), got.nan_to_num()), (name, fn.__name__)


@pytest.mark.parametrize("name", ALL)
def test_inplace_leaves_the_result_but_not_the_field(name):
    """ inplace=True is a licence to destroy h; the caller must not read it afterwards. """
    hz = hzd.HAZARDS[name]
    h = torch.linspace(-3.0, 3.0, 17)
    keep = h.clone()
    out = hz.phi(h, 2.0, 1.0, 0.25, inplace=True)
    assert torch.equal(out, hz.phi(keep.clone(), 2.0, 1.0, 0.25))


@pytest.mark.parametrize("name", ROUTE_B)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nu0,dt", [(1.0, 0.5), (1.0, 0.0625)])
def test_log_branches_are_finite_and_consistent(name, dtype, nu0, dt):
    """ ln Phi and ln(1 - Phi) must stay finite and correct where Phi itself has lost its
    digits -- at x = +-40 in float32 one of the two rounds to 0 or 1 and the naive log of it
    is -inf, which is exactly the case the closed forms exist for.

    Checked against log(Phi) and log1p(-Phi) only where THOSE are representable; elsewhere
    only finiteness and the sign are asserted, since there is no reference to compare to. """
    hz = hzd.HAZARDS[name]
    h = torch.cat([torch.linspace(-20.0, 20.0, 81, dtype=dtype),
                   torch.tensor([-40.0, 40.0], dtype=dtype)])
    beta = torch.tensor(1.0, dtype=dtype)
    phi = hz.phi(h.clone(), beta, nu0, dt)
    lp, l1 = hz.logphi(h.clone(), beta, nu0, dt), hz.log1m_phi(h.clone(), beta, nu0, dt)

    assert bool(torch.isfinite(lp).all()) and bool(torch.isfinite(l1).all())
    assert bool((lp <= 0).all()) and bool((l1 <= 0).all())

    # A small Phi keeps its relative precision, so log(Phi) is a reference wherever Phi > 0; a
    # Phi near one does not -- 1 - Phi is a cancellation that loses everything below the
    # spacing of 1 -- so log1p(-Phi) is only a reference while 1 - Phi stays well above it.
    # Beyond that there is nothing to compare against, which is why log1m_phi is carried.
    tol = 1e-5 if dtype is torch.float32 else 3e-9
    near_one = 1e-2 if dtype is torch.float32 else 1e-7
    assert torch.allclose(lp[phi > 0], torch.log(phi[phi > 0]), rtol=tol, atol=tol)
    ok = phi < 1.0 - near_one
    assert torch.allclose(l1[ok], torch.log1p(-phi[ok]), rtol=tol, atol=tol)


# The Poisson reference, and its two kernels
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("name", ROUTE_B)
def test_the_rate_is_the_exponential_tilt_of_the_reference(name):
    """ rate = nu0 e^{beta h}, in closed form and with no factor of one half: under route B
    the reference is a Poisson process of rate nu0 and c = nu0*dt, not nu0*dt/2. """
    beta, nu0 = 2.0, 1.3
    h = torch.linspace(-4.0, 4.0, 81)
    want = nu0 * torch.exp(beta * h)
    assert torch.allclose(hzd.HAZARDS[name].rate(h.clone(), beta, nu0, 0.1), want,
                          rtol=1e-14, atol=0)


def test_the_two_route_b_kernels_share_one_rate_function():
    """ Not merely equal: the same object. The two are one continuous-time model under two
    discretization maps, and `rate` is where that is readable -- so this is what makes a
    comparison between them a statement about discretization alone. """
    assert hzd.HAZARDS["poisson_exact"].rate is hzd.HAZARDS["poisson"].rate


@pytest.mark.parametrize("name", ROUTE_B)
def test_hazard_over_dt_converges_to_the_rate_at_first_order(name):
    """ Phi/dt -> nu0 e^{beta h} as dt -> 0, and at first order in dt.

    First order is asserted as well as the limit, because the limit alone would pass for a
    kernel that reached it the wrong way: halving dt must halve the error, which is what says
    the hazard is a consistent one-step map for that rate rather than some other kernel
    sharing its tangent. """
    hz, beta, nu0 = hzd.HAZARDS[name], 1.0, 1.0
    for x in (-8.0, -2.0, 0.0, 2.0):
        h = torch.tensor([x / beta], dtype=torch.float64)
        want = float(hz.rate(h.clone(), beta, nu0, 1.0))
        assert want == pytest.approx(nu0 * math.exp(x), rel=1e-12)
        errs = [abs(float(hz.phi(h.clone(), beta, nu0, dt)) / dt - want) / want
                for dt in (1e-3, 5e-4, 2.5e-4)]
        assert errs[1] == pytest.approx(errs[0] / 2, rel=0.05), (name, x, errs)
        assert errs[2] == pytest.approx(errs[1] / 2, rel=0.05), (name, x, errs)
        # the bound itself, which for the exact-survival map is lambda*dt/2 and for the
        # logistic one is the same order with a field-dependent coefficient
        assert errs[2] < 5e-3, (name, x, errs)


def test_poisson_and_poisson_exact_differ_at_order_dt_and_agree_in_the_limit():
    """ D5. Same rate function, different maps: the per-bin Phi differs, the difference is
    O(dt) with the same prefactor structure, and the two limits coincide.

    The gap is asserted to SHRINK LINEARLY rather than merely to be small, because a pair of
    kernels that agreed to O(dt) without agreeing in the limit would pass the second test and
    fail the first. """
    beta, nu0 = 1.0, 1.0
    h = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=torch.float64)
    po, pe = hzd.HAZARDS["poisson"], hzd.HAZARDS["poisson_exact"]
    # dt small enough that c*e^x << 1 at the strongest field here, so the leading O(dt) term
    # nu0^2 (e^x - e^{2x}/2) is what the differences are actually measuring
    gaps = []
    for dt in (0.01, 0.005, 0.0025):
        d = (po.phi(h.clone(), beta, nu0, dt) - pe.phi(h.clone(), beta, nu0, dt)) / dt
        gaps.append(d.abs().max().item())
    assert gaps[1] == pytest.approx(gaps[0] / 2, rel=0.1), gaps
    assert gaps[2] == pytest.approx(gaps[1] / 2, rel=0.1), gaps


def test_exact_survival_is_single_signed_where_the_logistic_error_changes_sign():
    """ The mechanism `poisson_exact` exists for, and the reason Richardson is run on it.

    Exact survival always UNDERESTIMATES the rate, 1 - e^{-z} <= z, at every field. The
    logistic kernel does not: Phi/dt = nu0 e^x / (1 + c(e^x - 1)) sits above the rate below
    threshold and below it above threshold. On an attractor spanning both, poisson's O(dt)
    error partly cancels against itself and the cancellation moves with dt, which is what
    makes its convergence non-monotone and poisson_exact's monotone. """
    beta, nu0, dt = 1.0, 1.0, 0.25
    x = torch.tensor([-4.0, -1.0, -0.1, 0.1, 1.0, 4.0], dtype=torch.float64)
    rate = hzd.poisson_rate(x.clone(), beta, nu0, dt)

    pe = hzd.HAZARDS["poisson_exact"].phi(x.clone(), beta, nu0, dt) / dt - rate
    assert bool((pe < 0).all()), f"exact survival should never overestimate: {pe}"

    po = hzd.HAZARDS["poisson"].phi(x.clone(), beta, nu0, dt) / dt - rate
    assert bool((po[:3] > 0).all()) and bool((po[3:] < 0).all()), (
        f"the logistic error should change sign at x = 0: {po}")


def test_poisson_exact_is_the_survival_of_its_own_rate():
    """ Phi = 1 - exp(-lambda*dt) with lambda the kernel's own rate: the definition, checked
    against the rate function rather than against a re-typed exponential. """
    h = torch.linspace(-3.0, 3.0, 121)
    beta, nu0, dt = 2.0, 1.0, 0.05
    hz = hzd.HAZARDS["poisson_exact"]
    lam = hz.rate(h.clone(), beta, nu0, dt)
    assert torch.allclose(hz.phi(h.clone(), beta, nu0, dt), -torch.expm1(-lam * dt),
                          rtol=0, atol=1e-15)


def test_poisson_is_the_logistic_kernel_at_the_reference_log_odds():
    """ Phi = sigma(beta*h + logit(nu0*dt)): the MaxCal form, against the closed form. """
    h = torch.linspace(-3.0, 3.0, 121)
    beta, nu0, dt = 2.0, 1.0, 0.125
    ell = math.log(nu0 * dt) - math.log1p(-nu0 * dt)
    assert torch.allclose(hzd.HAZARDS["poisson"].phi(h.clone(), beta, nu0, dt),
                          torch.sigmoid(beta * h + ell), rtol=0, atol=1e-15)


# Reparametrization invariance (D2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# nu0 is a UNIT, not a fitted parameter: only nu0 e^{-beta theta} is identifiable. For
# poisson_exact that is exact at the level of Phi; for poisson it holds in the reference
# LOG-ODDS instead, since logit(nu0*dt) is not linear in ln nu0 at finite c.

@pytest.mark.parametrize("delta", [-0.3, 0.1, 0.7])
def test_poisson_exact_is_invariant_under_the_rate_threshold_reparametrization(delta):
    """ (nu0, theta) -> (nu0 e^{beta delta}, theta + delta) leaves Phi unchanged, exactly. """
    beta, nu0, dt = 6.0, 1.0, 0.1
    S = torch.linspace(-1.0, 2.0, 201)          # the field before the threshold is subtracted
    theta = 0.5
    hz = hzd.HAZARDS["poisson_exact"]
    a = hz.phi(S - theta, beta, nu0, dt)
    b = hz.phi(S - (theta + delta), beta, nu0 * math.exp(beta * delta), dt)
    assert torch.allclose(a, b, rtol=1e-14, atol=1e-300), (a - b).abs().max()


@pytest.mark.parametrize("delta", [-0.3, 0.1, 0.7])
def test_poisson_is_invariant_in_the_reference_log_odds(delta):
    """ For the MaxCal kernel the invariance is (ell, theta) -> (ell + beta*delta, theta +
    delta), ell = logit(nu0*dt) being the quantity the field is shifted by. Written as the nu0
    that REALISES that log-odds at this dt, which is what a script would have to pass -- and
    which is NOT nu0 e^{beta delta}, the difference being the whole content of the test. """
    # beta is modest here so that the "naive" rescaling below stays inside c < 1 and can be
    # evaluated at all: the point is that it gives a DIFFERENT kernel, not that it raises
    beta, nu0, dt = 2.0, 1.0, 0.1
    theta, S = 0.5, torch.linspace(-1.0, 2.0, 201)
    ell = math.log(nu0 * dt) - math.log1p(-nu0 * dt)
    nu0_b = (1.0 / (1.0 + math.exp(-(ell + beta * delta)))) / dt      # sigma(ell')/dt
    hz = hzd.HAZARDS["poisson"]
    a = hz.phi(S - theta, beta, nu0, dt)
    b = hz.phi(S - (theta + delta), beta, nu0_b, dt)
    assert torch.allclose(a, b, rtol=1e-13, atol=1e-300), (a - b).abs().max()
    if abs(delta) > 0.2:
        naive = hz.phi(S - (theta + delta), beta, nu0 * math.exp(beta * delta), dt)
        assert not torch.allclose(naive, a, rtol=1e-6, atol=0), \
            "the exponential rescaling is NOT the invariance for this kernel"


# The legacy Bernoulli kernel (D6)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_bernoulli_at_the_unit_step_is_the_sigmoid_exactly():
    """ At nu0*dt = 1 the thinning factor is one and the per-bin kernel IS sigma(beta*h), bit
    for bit on all four field branches. That is the legacy discrete model every pre-route-B
    result was computed in, and the closed form replaces the cached regression number the old
    suite used for it. """
    beta, h = 6.0, torch.linspace(-4.0, 4.0, 801)
    hz = hzd.HAZARDS["bernoulli"]
    for nu0, dt in ((1.0, 1.0), (2.0, 0.5), (4.0, 0.25)):
        assert torch.equal(hz.phi(h.clone(), beta, nu0, dt), torch.sigmoid(beta * h))
        sig = torch.sigmoid(beta * h)
        assert torch.allclose(hz.dphi(h.clone(), beta, nu0, dt), beta * sig * (1 - sig),
                              rtol=0, atol=1e-15)
        # the log branches are compared only where the naive form is representable: at
        # beta*h = 24 the sigmoid has rounded to 1.0 and log(sig) is exactly 0, while the
        # closed form still carries -softplus(-24) = -3.8e-11. That the closed form is the
        # MORE accurate of the two is why it is carried, so it is asserted rather than
        # tolerated -- the silent branch at c = 1 likewise has to be taken as a limit, since
        # log1p(-Phi) reaches -inf several digits early.
        # The tolerance is torch's softplus `threshold`: above an input of 20 it switches to
        # the identity and drops ln(1 + e^-20) = 2.1e-9. These kernels take that as given
        # rather than overriding it -- the same call is what makes the phi branch above agree
        # with the sigmoid bit for bit -- so 3e-9 is the floor of this comparison, not slack.
        lp, l1 = hz.logphi(h.clone(), beta, nu0, dt), hz.log1m_phi(h.clone(), beta, nu0, dt)
        assert torch.allclose(lp[sig > 0], torch.log(sig[sig > 0]), rtol=0, atol=3e-9)
        # and the silent branch only where 1 - sig is well clear of the spacing of 1: a sigmoid
        # near one keeps no relative precision, so log1p(-sig) is itself wrong by 2e-6 at
        # beta*h = 24 while the closed form is exact. That gap is the branch's reason to exist.
        near = sig < 1.0 - 1e-7
        assert torch.allclose(l1[near], torch.log1p(-sig[near]), rtol=0, atol=3e-9)
        assert bool(torch.isfinite(lp).all()) and bool(torch.isfinite(l1).all())
        assert float(lp[-1]) == pytest.approx(-math.log1p(math.exp(-beta * 4.0)), abs=1e-18)
        assert float(l1[0]) == pytest.approx(-math.log1p(math.exp(-beta * 4.0)), abs=1e-18)


def test_bernoulli_rate_saturates_at_the_unit():
    """ nu0*sigma(beta*h): bounded, unlike the route-B rate, which is the substantive
    difference between the two routes rather than a numerical one. """
    nu0 = 1.7
    strong = torch.tensor([20.0], dtype=torch.float64)
    assert float(hzd.bernoulli_rate(strong, 6.0, nu0, 0.1)) == pytest.approx(nu0, rel=1e-12)
    assert float(hzd.poisson_rate(strong, 6.0, nu0, 0.1)) > 1e40


# Step guards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_bernoulli_admits_the_unit_step_and_refuses_more():
    """ Phi = nu0*dt*sigma exceeds 1 above dt = 1/nu0; the edge itself is the legacy model. """
    hzd.check_step("bernoulli", 1.0, 1.0)
    with pytest.raises(ValueError, match="admits dt <="):
        hzd.check_step("bernoulli", 1.0, 1.0 + 1e-9)
    bad = hzd.bernoulli_hazard(torch.tensor([8.0]), 30.0, nu0=1.0, dt=2.0)
    assert float(bad.max()) > 1.0, "the constraint should bite above the unit step"


def test_poisson_refuses_the_unit_step_strictly():
    """ c = nu0*dt is a probability and ell = logit(c) diverges at c = 1, where the reference
    fires every bin and the tilt has nothing left to tilt. The cap is strict. """
    hzd.check_step("poisson", 1.0, 0.999)
    with pytest.raises(ValueError, match="admits dt <="):
        hzd.check_step("poisson", 1.0, 1.0)
    with pytest.raises(ValueError, match=r"0 < c = nu0\*dt < 1"):
        hzd.poisson_hazard(torch.tensor([0.0]), 1.0, 1.0, 1.0)


def test_poisson_exact_has_no_arithmetic_cap():
    """ Phi = 1 - e^{-y} is a probability for every y, so nothing caps dt here -- the step is
    set by accuracy alone, which is the other half of why this is the kernel a dt -> 0 limit
    is taken with. """
    assert hzd.HAZARDS["poisson_exact"].step_cap(1.0) is None
    hzd.check_step("poisson_exact", 1.0, 25.0)
    phi = hzd.HAZARDS["poisson_exact"].phi(torch.linspace(-5, 5, 51), 2.0, 1.0, 25.0)
    assert bool(((phi >= 0) & (phi <= 1)).all())


def test_step_error_is_the_half_lambda_dt_bound():
    """ The local relative error of the exact-survival map, in closed form, and the tolerance
    check built on it. """
    assert hzd.step_error(4.0, 0.01) == pytest.approx(0.02)
    hzd.check_accuracy(4.0, 0.01, tol=0.05)
    with pytest.raises(ValueError, match="exceeds tolerance"):
        hzd.check_accuracy(4.0, 0.01, tol=0.01)


def test_resolve_timesteps_requires_a_step_and_a_known_hazard():
    assert hzd.resolve_timesteps(0.25, 1.0, 1.0, "poisson_exact") == (0.25, 1.0, 1.0)
    with pytest.raises(ValueError, match="hazard must be one of"):
        hzd.resolve_timesteps(0.25, 1.0, 1.0, "escape")
    with pytest.raises(ValueError, match="dt is required"):
        hzd.resolve_timesteps(None, 1.0, 1.0, "poisson_exact")
    with pytest.raises(ValueError, match="t_abs must be"):
        hzd.resolve_timesteps(0.25, -1.0, 1.0, "poisson_exact")
    with pytest.raises(ValueError, match="nu0 must be positive"):
        hzd.resolve_timesteps(0.25, 1.0, 0.0, "poisson_exact")
