""" Tests that the mean-field kernels agree with what the spin model realizes. """

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

    sm = SpinIsingModel(1, 0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, dt0=dt, ic="silent")
    mf = RDMIsingModel(0.0, 0.0, 1.0, 0.0, tau_int, tau_ref, K_ref, dt0=dt,
                       hazard="async")

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

    sm = SpinIsingModel(N, J, E, beta, theta, tau_int, tau_ref, K_ref, dt0=dt, ic="silent")
    mf = RDMIsingModel(J, E, beta, theta, tau_int, tau_ref, K_ref, dt0=dt,
                       hazard="async")
    sm.forward(6000)
    mf.forward(6000)

    m_sm = torch.stack([(sm.update(), sm.activity())[1] for _ in range(2000)]).mean()
    # srm's activity() is a RATE (m/dt), while the spin model's is the per-bin overlap, so
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
    assert krn.n_abs_bins(1.0, 1.0) == 0      # empty at the quantum: the whole point
    assert krn.n_abs_bins(0.0, 0.1) == 0      # t_abs=0 is the plain exponential kernel


@pytest.mark.parametrize("dt0", [0.5, 1.0, 2.0])
def test_the_window_is_empty_at_the_quantum(dt0):
    """ t_abs = dt0 is the default, and it must cost NOTHING at dt = dt0 -- otherwise it is a
    change of model rather than a refinement-consistency device. Bit-for-bit, not allclose. """
    P = dict(J=2., E=1., beta=4., theta=.5, tau_int=5., tau_ref=4., K_ref=3., eps=.05)
    a = RDMIsingModel(**P, dt0=dt0, hazard="async")                # default t_abs = dt0
    b = RDMIsingModel(**P, dt0=dt0, hazard="async", t_abs=0.0)     # pre-window behaviour
    assert a.n_abs == 0
    assert torch.equal(a.R[0], b.R[0])
    a.forward(400); b.forward(400)
    assert torch.equal(a.p[0], b.p[0]) and torch.equal(a.S[0], b.S[0])


@pytest.mark.parametrize("dt", [0.5, 0.2, 0.1])
def test_spin_model_honours_the_quantum_below_dt0(dt):
    """ The constraint the window is for: a spike train binned at dt0 cannot carry two spikes
    closer than dt0, and no hazard enforces that -- "async" caps the RATE at lam0, not the
    interval. Without the window 25-56% of intervals fall below dt0 as dt is refined.

    This samples the bulk of the ISI distribution. It does NOT reliably catch the float32
    leak -- that is ~6e-8 per neuron-step, i.e. under one expected event at this size, and it
    was measured to pass with the mask removed. test_blocked_neurons_cannot_fire_on_a_zero_draw
    covers that mechanism deterministically. """
    dt0, N = 1.0, 4000
    torch.manual_seed(0)
    sm = SpinIsingModel(N=N, J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0,
                        K_ref=0.0, dt0=dt0, dt=dt, hazard="async", ic="silent")
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
    assert worst * dt == pytest.approx(dt0), f"min ISI {worst * dt} ms below the quantum {dt0}"


@pytest.mark.parametrize("hazard", ["async", "escape", "poisson", "exp_surv"])
@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2])
def test_spin_and_mean_field_agree_below_the_quantum(dt, hazard):
    """ The window lives in two different representations -- an R(u) kernel in the mean field,
    an age mask in the spin model, whose R is per-neuron state with no age axis. They are
    therefore free to drift apart, and nothing pinned them together below dt0 before.

    Over every hazard, because the two representations are only pinned together through the
    kernel interface: a kernel that read dt or lam0 differently on the two sides would show up
    here and nowhere else. The operating point is strongly driven, so all three saturate and
    the realised rate is set by the window rather than by the kernel -- which is the point,
    since it is the window being compared. """
    dt0, N = 1.0, 20000
    P = dict(J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0, K_ref=0.0)
    torch.manual_seed(0)
    sm = SpinIsingModel(N=N, **P, dt0=dt0, dt=dt, hazard=hazard, ic="silent")
    mf = RDMIsingModel(**P, dt0=dt0, dt=dt, hazard=hazard, Qm=[int(60 / dt)])
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
    # something this test is entitled to. It fails exactly where the test is most interesting:
    # at a strongly driven point an unbounded kernel saturates Phi to 1 and the mean field
    # enters the two-step flip (measured for "exp_surv" at dt = 0.2: instantaneous r anywhere
    # in [0, 2.4], time average 1.00000, which is the window's exact ceiling of 1/dt0). The
    # window is what is being tested and it is enforced on average; the flip is the step
    # failing to resolve the hazard, which is max_step_prob's business and is tested there.
    mean_mf = float(mf.trajectory(n, pb=False)["r"][:, 0].mean())
    assert acc / n == pytest.approx(mean_mf, rel=0.02)


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
    dt0, dt, N = 1.0, 0.1, 500
    sm = SpinIsingModel(N=N, J=0.0, E=6.0, beta=6.0, theta=0.0, tau_int=5.0, tau_ref=4.0,
                        K_ref=0.0, dt0=dt0, dt=dt, hazard="async", ic="silent")
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
    common = dict(dt=0.1, dt0=0.5, hazard="async", t_abs=t_abs, Qm=[60, 60])
    pts = RDMWilsonCowanPoints(0.5, T([16.667]), T([0.0311]), T([0.701]), T([10.938]),
                               0., 0., 2.5, 5., 2., 0.84, 1., 1., T([4.05]), T([3.60]),
                               T([30.]), T([30.]), **common)
    bat = RDMWilsonCowanBatch(0.5, T([16.667, 17.0]), 0.0311, 0.701, 10.938, 0., 0.,
                              2.5, 5., 2., 0.84, 1., 1., 4.05, 3.60, 30., 30., **common)
    isi = RDMIsingModelBatch(J=T([1., 2.]), E=1., beta=30., theta=1., tau_int=20.,
                             tau_ref=3., K_ref=0., dt=0.1, dt0=0.5, hazard="async",
                             t_abs=t_abs, Qm=[60])
    want = krn.n_abs_bins(t_abs, 0.1)
    for name, m in (("points", pts), ("batch", bat), ("ising", isi)):
        assert m.t_abs == t_abs, name
        assert m.n_abs == want, (name, m.n_abs, want)
        blocked = (m.R[0][..., :want] == -m.R_abs[..., 0:1]).all() if want else True
        assert blocked, f"{name}: t_abs reached the attribute but not the kernel"
