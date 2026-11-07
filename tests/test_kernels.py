""" Tests that the mean-field kernels agree with what the spin model realizes, and that the
absolute refractory window is the physical dead time it claims to be.

Under route B t_abs is a parameter of the model in ms, not a shadow of the solver step, and it
is the only thing that bounds the realised firing rate under an exponential hazard. What is
checked here is that the two representations of it agree -- an R(u) kernel in the mean field,
an age mask in the spin model -- that the realised dead time is t_abs at every dt, and that
the stationary rate it produces is the closed form the dead time implies. """

import math

import pytest
import torch

import rdme.kernels as krn
from rdme.rdm_single import RDMIsingModel
from rdme.spin_model import SpinIsingModel


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2, 0.05])
def test_refractory_kernel_matches_spin_model(dt):
    """ The spin model decays X by (1 - alpha_ref) per step and resets it to -K_ref on a
    spike, so a neuron of age n carries X = -K_ref (1 - alpha_ref)^n. The mean field reads
    that off the R kernel, so R(n) must reproduce it bin for bin -- which it only does
    if the kernel folds in the bin width deltaT. """
    tau_ref, K_ref, tau_int = 3.0, 2.0, 12.0

    # t_abs = 0: this compares the EXPONENTIAL term alone, so the window must be out of the
    # way. With the model default of 1 ms the leading bins carry the floor -R_abs instead.
    kw = dict(dt=dt, t_abs=0.0)
    sm = SpinIsingModel(1, 0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref,
                        hazard="poisson_exact", ic="silent", **kw)
    mf = RDMIsingModel(0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref,
                       hazard="poisson_exact", **kw)

    n = torch.arange(mf.Qm[0], dtype=mf.R[0].dtype)
    eta_spin = -K_ref * (1.0 - sm.a_ref.item()) ** n

    assert torch.allclose(mf.R[0], eta_spin, atol=1e-5)


def test_refractory_kernel_deltaT_scaling():
    """ deltaT rescales the age axis: R(n; deltaT) == R(n*deltaT; 1). """
    Q, K, tau_r, dt = 50, 2.0, 3.0, 0.2
    got = krn.refractory_kernel(Q, K, tau_r, dt)
    expected = krn.refractory_kernel(Q, K, tau_r / dt)
    assert torch.allclose(got, expected)
    # default is the dt=1 kernel, unchanged
    assert torch.allclose(krn.refractory_kernel(Q, K, tau_r),
                          -K * torch.exp(-torch.arange(Q) / tau_r))


def test_stationary_activity_matches_spin_model():
    """ End-to-end: with matching kernels both models settle on the same fixed point. """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    N, J, E, beta, theta, dt = 20000, 1.0, 0.0, 20.0, 0.0, 0.2
    tau_int, tau_ref, K_ref = 12.0, 3.0, 2.0

    kw = dict(dt=dt, hazard="poisson_exact")
    sm = SpinIsingModel(N, J, E, beta, theta, tau_int, tau_ref, K_ref, ic="silent", **kw)
    mf = RDMIsingModel(J, E, beta, theta, tau_int, tau_ref, K_ref, **kw)
    sm.forward(6000)
    mf.forward(6000)

    m_sm = torch.stack([(sm.update(), sm.activity())[1] for _ in range(2000)]).mean()
    # the mean field's activity() is a RATE (m/dt), while the spin model's is the per-bin
    # overlap, so
    # one of the two has to be converted before they are comparable. This is the difference
    # that makes the srm numbers dt-invariant and the spin model's not.
    r_mf = torch.stack([(mf.update(), mf.activity())[1] for _ in range(2000)]).mean()
    m_mf = r_mf * dt

    # finite-N sampling noise on m_sm is ~1e-3 relative at N=2e4
    assert m_mf == pytest.approx(m_sm.item(), rel=0.02)


# The absolute refractory window
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_n_abs_bins_gives_a_dt_independent_dead_time():
    """ The off-by-one this function exists for. Firing from bin n gives an ISI of (n+1)*dt,
    so blocking round(t_abs/dt) bins -- the obvious reading -- yields a dead time of
    t_abs + dt, which moves with the solver step. """
    for dt in (0.5, 0.2, 0.1, 0.05):
        n_abs = krn.n_abs_bins(1.0, dt)
        assert (n_abs + 1) * dt == pytest.approx(1.0), dt
    assert krn.n_abs_bins(1.0, 1.0) == 0      # one bin of dead time needs no blocking
    assert krn.n_abs_bins(0.0, 0.1) == 0      # t_abs=0 is the plain exponential kernel


def test_n_abs_bins_refuses_a_step_that_cannot_represent_the_window():
    """ D4. The two guards, both against the same failure: a window that rounds moves with the
    solver step, which is the one thing an absolute refractory period must not do. """
    with pytest.raises(ValueError, match="exceeds t_abs"):
        krn.n_abs_bins(1.0, 2.0)
    with pytest.raises(ValueError, match="not an integer number"):
        krn.n_abs_bins(1.0, 0.3)
    with pytest.raises(ValueError, match="not an integer number"):
        krn.n_abs_bins(1.0, 0.07)
    # ... and the divisors of 1 ms, which are what the (dt, dt/2) Richardson pairs walk down
    for dt in (1.0, 0.5, 0.25, 0.125, 0.0625):
        assert (krn.n_abs_bins(1.0, dt) + 1) * dt == pytest.approx(1.0)


@pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
def test_the_window_is_empty_when_it_is_one_bin_wide(dt):
    """ At t_abs = dt the window blocks nothing -- a neuron that fires enters bin 0 and is
    first eligible again one step later, so the realised dead time is already dt. The kernel
    must then be the plain exponential one, bit for bit, not merely close to it. """
    P = dict(J=2., E=1., beta=4., theta=.5, tau_int=5., tau_ref=4., K_ref=3., eps=.05)
    a = RDMIsingModel(**P, dt=dt, hazard="poisson_exact", t_abs=dt)
    b = RDMIsingModel(**P, dt=dt, hazard="poisson_exact", t_abs=0.0)
    assert a.n_abs == 0
    assert torch.equal(a.R[0], b.R[0])
    a.forward(400); b.forward(400)
    assert torch.equal(a.p[0], b.p[0]) and torch.equal(a.S[0], b.S[0])


@pytest.mark.parametrize("dt", [0.5, 0.2, 0.1])
def test_the_realised_dead_time_is_t_abs_at_every_step(dt):
    """ D4, in the spin model: the shortest interspike interval actually realised is t_abs
    EXACTLY, and does not move when dt does. That is the whole content of the window under
    route B -- the exponential rate is unbounded, so without it the realised rate grows with
    the drive and with the refinement, and the model has no rate ceiling.

    This samples the bulk of the ISI distribution. It does NOT reliably catch the float32
    leak -- that is ~6e-8 per neuron-step, i.e. under one expected event at this size.
    test_blocked_neurons_cannot_fire_on_a_zero_draw covers that mechanism deterministically.
    """
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
        if idx.numel():
            gap = t - last[idx]
            prior = gap < 10**8
            if prior.any():
                worst = min(worst, gap[prior].min().item()); seen += int(prior.sum())
            last[idx] = t
    assert seen > 10_000, f"only {seen} intervals sampled; test is not exercising anything"
    assert worst * dt == pytest.approx(t_abs), f"min ISI {worst * dt} ms below t_abs {t_abs}"


@pytest.mark.slow
@pytest.mark.parametrize("hazard", ["poisson_exact", "poisson"])
@pytest.mark.parametrize("dt", [0.5, 0.2])
def test_spin_and_mean_field_agree_under_the_window(dt, hazard):
    """ The window lives in two different representations -- an R(u) kernel in the mean field,
    an age mask in the spin model, whose R is per-neuron state with no age axis. They are
    therefore free to drift apart, and nothing pinned them together before.

    Both route-B kernels, because the two representations are only pinned together through
    the kernel interface: a kernel that read dt or nu0 differently on the two sides would show
    up here and nowhere else. (The route-A grid this replaced ran four kernels at three steps;
    twelve cases took 37 s and said the same thing four times over.) The operating point is
    strongly driven, so the realised rate is set by the window rather than by the kernel --
    which is the point, since it is the window being compared. """
    t_abs, N = 1.0, 20000
    P = dict(J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0, K_ref=0.0)
    torch.manual_seed(0)
    sm = SpinIsingModel(N=N, **P, dt=dt, t_abs=t_abs, hazard=hazard, ic="silent")
    mf = RDMIsingModel(**P, dt=dt, t_abs=t_abs, hazard=hazard, Qm=[int(60 / dt)])
    assert sm.n_abs == mf.n_abs, "the two disagree on the window width itself"
    sm.forward(int(2000 / dt)); mf.forward(int(2000 / dt))
    n = int(400 / dt)
    acc = 0.0
    for _ in range(n):
        sm.update()
        acc += (sm.s == 1).float().mean().item() / dt
    # Both sides averaged over the same window. The spin rate is a time average by
    # construction, so reading ONE instant off the mean field only compares like with like
    # when the mean field has settled -- which is an assumption about the operating point, not
    # something this test is entitled to. At a strongly driven point an unbounded kernel
    # saturates Phi to 1 and the mean field enters the two-step flip (instantaneous r anywhere
    # in [0, 2/t_abs], time average 1/t_abs, which is the window's exact ceiling). The window
    # is what is being tested and it is enforced on average; the flip is the step failing to
    # resolve the hazard, which is max_step_prob's business and is tested there.
    mean_mf = float(mf.trajectory(n, pb=False)["r"][:, 0].mean())
    assert acc / n == pytest.approx(mean_mf, rel=0.02)


# The dead-time transfer function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _renewal_rate(x: float, nu0: float, t_abs: float) -> float:
    """ The closed form: lambda/(1 + lambda t_abs) with lambda = nu0 e^x, equivalently
    (1/t_abs) sigma(x + ln(nu0 t_abs)). Both spellings are returned by the same number, which
    is what the test below asserts first. """
    lam = nu0 * math.exp(x)
    return lam / (1.0 + lam * t_abs)


def _free_run(x, dt, *, nu0=1.0, t_abs=1.0, u_max=14.0, eq_ms=80.0, rec_ms=40.0):
    """ A single population with no coupling, no relative refractoriness and an instantaneous
    synaptic filter: a pure renewal process with dead time t_abs and constant hazard.

    The rate is a TIME AVERAGE. A hard window with little dispersion after it makes the age
    distribution ring with period t_abs, and the ringing decays slowly when the post-window
    hazard is strong -- so one instant of `rates()` is not the stationary rate, which is the
    same reading error the spin-vs-mean-field comparison above documents. """
    net = RDMIsingModel(J=0.0, E=x, beta=1.0, theta=0.0, tau_int=dt / 50, tau_ref=1.0,
                        K_ref=0.0, dt=dt, nu0=nu0, t_abs=t_abs, hazard="poisson_exact",
                        Qm=[int(u_max / dt)])
    net.forward(int(eq_ms / dt))
    r = float(net.trajectory(int(rec_ms / dt), pb=False)["r"][:, 0].mean())
    return r, net.truncation_error()


@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_the_stationary_rate_is_the_dead_time_transfer_function(x):
    """ D3. With K_ref = 0, tau_int -> 0, no coupling and a constant drive, the model is a
    renewal process with a hard dead time t_abs followed by a CONSTANT hazard
    lambda = nu0 e^{beta h}. Its mean interval is t_abs + 1/lambda, so

        r = lambda / (1 + lambda t_abs) = (1/t_abs) sigma(beta h + ln(nu0 t_abs)),

    a closed form rather than a measurement: the stationary rate is a logistic function of the
    drive whose SATURATION is set by the dead time rather than by the kernel.

    Checked at two steps. The exact-survival map replaces the mean waiting time 1/lambda by
    dt(1 - Phi)/Phi, which is smaller, so the discrete rate sits ABOVE the closed form by a
    relative amount bounded by lambda*dt/2 -- and that bound must halve when dt does. Both the
    sign and the halving are asserted: a scheme that happened to land inside the bound by
    cancelling two errors would pass neither. """
    nu0, t_abs, beta = 1.0, 1.0, 1.0
    lam = nu0 * math.exp(x)
    want = _renewal_rate(x, nu0, t_abs)
    assert want == pytest.approx(
        1.0 / (t_abs * (1.0 + math.exp(-(x + math.log(nu0 * t_abs))))), rel=1e-12)

    errs = []
    for dt in (0.02, 0.01):
        got, trunc = _free_run(x, dt, nu0=nu0, t_abs=t_abs)
        assert trunc < 1e-3, trunc
        err = (got - want) / want
        assert 0 < err < 1.5 * 0.5 * lam * dt, (dt, err, 0.5 * lam * dt)
        errs.append(err)
    assert errs[1] == pytest.approx(errs[0] / 2, rel=0.1), errs


@pytest.mark.slow
def test_the_dead_time_is_the_only_rate_ceiling():
    """ The saturated end of the same transfer function, and the reason t_abs is a parameter
    at all. At x = 3 the hazard is nu0 e^3 = 20/ms, twenty times the ceiling the window
    imposes, and an unbounded exponential rate with no dead time would simply follow it. With
    t_abs = 1 ms the realised rate is 0.9526/ms instead: strictly below 1/t_abs, and equal to
    the closed form to 1e-3.

    Slow because resolving a hazard of 20/ms needs dt = 2e-3, i.e. a 500-bin window. """
    x, nu0, t_abs, dt = 3.0, 1.0, 1.0, 0.002
    want = _renewal_rate(x, nu0, t_abs)
    got, trunc = _free_run(x, dt, nu0=nu0, t_abs=t_abs, u_max=4.0, eq_ms=60.0, rec_ms=20.0)
    assert trunc < 1e-6, trunc
    assert got < 1.0 / t_abs, f"the window's ceiling is 1/t_abs = {1 / t_abs}, got {got}"
    assert got == pytest.approx(want, rel=2e-3), (got, want)
    assert nu0 * math.exp(x) > 20 * want, "the hazard should be far above the realised rate"


def test_blocked_neurons_cannot_fire_on_a_zero_draw(monkeypatch):
    """ The float32 leak, hit head-on rather than waited for.

    R_abs has to stay finite -- -inf gives 0 * -inf = nan in the log-probability branches of
    the entropy production and in dPhi/dh -- so a blocked bin carries Phi ~ 1e-34 instead of 0.
    torch.rand returns exactly 0.0 with probability ~2^-24 in float32, and `0.0 < Phi` is True
    for ANY positive Phi, so a window enforced only through the field leaks. Raising R_abs
    cannot close it: the comparison bottoms out at the smallest representable draw, not at the
    size of the field.

    Forcing every draw to 0.0 turns a ~6e-8 event into a certainty, so this fails immediately
    if the mask on `fired` is removed, where the bulk ISI test above needs ~1e7 draws and a
    lucky seed. """
    dt, N = 0.1, 500
    sm = SpinIsingModel(N=N, J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0,
                        K_ref=0.0, dt=dt, t_abs=1.0, hazard="poisson_exact", ic="silent")
    assert sm.n_abs > 0, "no window to test"
    sm.forward(int(300 / dt))
    assert (sm.n < sm.n_abs).any(), "no neuron is inside the window; test proves nothing"

    # every blocked neuron has a positive-but-tiny Phi: the leak needs that to be true
    phi_blocked = sm.firing_prob()[sm.n < sm.n_abs]
    assert (phi_blocked > 0).all() and (phi_blocked < 1e-20).all(), phi_blocked.max()

    young = sm.n < sm.n_abs
    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.zeros(N, device=sm.device))
    sm.update()
    leaked = int(((sm.s == 1) & young).sum())
    assert leaked == 0, (f"{leaked} neurons fired inside the absolute refractory window on a "
                         "zero draw: the window is not enforced on the Bernoulli draw")


def test_every_factory_accepts_and_forwards_t_abs():
    """ t_abs has to reach the kernel through every public entry point, not just the two
    classes. The three batch factories and the _wilson_cowan_from_flat helper they share were
    all missed when it was first threaded through, and nothing here caught it -- the scripts
    still parsed and only failed at construction. An audit is cheap; a sweep that dies an hour
    in is not. """
    import inspect
    import rdme.rdm_single as single, rdme.rdm_batch as batch, rdme.spin_model as spin

    missing = [f"{mod.__name__}.{name}"
               for mod in (single, batch, spin)
               for name in dir(mod)
               if name.startswith(("RDM", "Spin")) and callable(getattr(mod, name))
               and (sig := _sig(getattr(mod, name))) is not None
               and "t_abs" not in sig.parameters]
    assert not missing, f"these do not accept t_abs: {missing}"


def _sig(fn):
    import inspect
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


@pytest.mark.parametrize("t_abs", [0.0, 1.0])
def test_batch_factories_forward_t_abs_to_the_kernel(t_abs):
    """ Accepting the argument is not the same as using it: the Wilson-Cowan factories route
    through a shared helper that silently dropped it. Check the kernel that comes out. """
    from rdme.rdm_batch import RDMWilsonCowanPoints, RDMWilsonCowanBatch, RDMIsingModelBatch
    T = torch.tensor
    common = dict(dt=0.1, hazard="poisson_exact", t_abs=t_abs, Qm=[60, 60])
    pts = RDMWilsonCowanPoints(0.5, T([16.667]), T([0.0311]), T([0.701]), T([10.938]),
                               0., 0., 2.5, 5., 2., 0.84, 1., 1., T([4.05]), T([3.60]),
                               T([30.]), T([30.]), **common)
    bat = RDMWilsonCowanBatch(0.5, T([16.667, 17.0]), 0.0311, 0.701, 10.938, 0., 0.,
                              2.5, 5., 2., 0.84, 1., 1., 4.05, 3.60, 30., 30., **common)
    isi = RDMIsingModelBatch(J=T([1., 2.]), E=1., beta=30., theta=1., tau_int=20.,
                             tau_ref=3., K_ref=0., dt=0.1, hazard="poisson_exact",
                             t_abs=t_abs, Qm=[60])
    want = krn.n_abs_bins(t_abs, 0.1)
    for name, m in (("points", pts), ("batch", bat), ("ising", isi)):
        assert m.t_abs == t_abs, name
        assert m.n_abs == want, (name, m.n_abs, want)
        blocked = (m.R[0][..., :want] == -m.R_abs[..., 0:1]).all() if want else True
        assert blocked, f"{name}: t_abs reached the attribute but not the kernel"
