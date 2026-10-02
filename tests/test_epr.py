""" Tests for rdme.epr: the time-reversed field, the joint distribution of forward and
reverse age, and the two paths that read the entropy production off it.

Everything here runs the synchronous kernel, so these are statements about the EPR machinery
rather than about a particular hazard -- the comparison across the three is in test_srm.py.
That also means the unvectorized references below can keep writing the hazard as a plain
sigmoid, which is what makes them independent of the implementation they check.
"""

import contextlib

import pytest
import torch

import rdme.hazards as hzd
import rdme.dynamics as dyn
import rdme.epr
import rdme.single
from rdme.batch import RDMNetworkBatch
from rdme.epr import (anchor_reverse_synaptic_term, compute_reverse_synaptic_term,
                      epr_from_joint, epr_from_parts, init_epr_state, joint_distribution,
                      make_epr_indices, step_epr)
from rdme.single import RDMIsingModel, RDMNetwork


torch.manual_seed(0)

# the synchronous kernel at its own quantum: lam0*dt = 1, where sync ignores both anyway
HZ, LAM0, DT = hzd.HAZARDS["synchronous"], 1.0, 1.0


def _phi(S, R, beta, theta):
    return HZ.phi(S + R - theta, beta, LAM0, DT)


def _epr_from_buffers(p, buff_S, buff_I, R, alpha, beta, theta):
    """ One population's EPR for one timestep, straight from its rolling buffers.

    rdme.mean_field had this as a function; srm composes it inline inside
    entropy_trajectory, so the reference the sliding path is checked against lives here. """
    S_rev = anchor_reverse_synaptic_term(buff_I, alpha)
    J = joint_distribution(p, buff_S, R, beta, theta, HZ, LAM0, DT)
    return epr_from_joint(J, _phi(buff_S[0], R, beta, theta),
                              _phi(S_rev, R, beta, theta))


def test_compute_backward_field():
    torch.manual_seed(4)
    Q = 8
    alpha = 0.4
    I = torch.randn(Q, dtype=torch.float64)

    H = compute_reverse_synaptic_term(I, alpha)

    # no field has accumulated yet at age 0
    assert H[0] == 0

    # closed form: H[j] = sum_{m=1}^{j} alpha*(1-alpha)^(m-1) * I[m], i.e. a kernel-
    # weighted cumulative sum over the buffered future inputs
    expected = torch.zeros(Q, dtype=torch.float64)
    for j in range(1, Q):
        expected[j] = sum(
            alpha * (1 - alpha) ** (m - 1) * I[m] for m in range(1, j + 1))
    # tight tolerance on purpose: the kernel used to be built against a float32 index
    # vector, which demoted the whole product to float32 (a 0-d or python-scalar alpha
    # loses the promotion) and cost ~1e-6 relative accuracy under a float64 default
    assert torch.allclose(H, expected, rtol=0, atol=1e-15)

    # alpha as a 0-d tensor -- how the models actually store it -- must be no different
    H_t = compute_reverse_synaptic_term(I, torch.tensor(alpha, dtype=torch.float64))
    assert torch.equal(H, H_t)

    # batched: a (B, K) buffer with a (B, 1) alpha is the same field, row by row
    H_b = compute_reverse_synaptic_term(I.expand(3, Q).contiguous(),
                                    torch.full((3, 1), alpha, dtype=torch.float64))
    assert torch.allclose(H_b, H.expand(3, Q), rtol=0, atol=1e-15)

    # alpha=1: the kernel is [1, 0, 0, ...], so the field is pinned at I[1]
    # for every j >= 1 rather than tracking the most recent input
    H1 = compute_reverse_synaptic_term(I, 1.0)
    assert torch.allclose(H1[1:], torch.full_like(H1[1:], I[1].item()))


def test_anchor_reverse_synaptic_term():
    """ The EPR's reverse field is h^dagger_{n^dagger, t+1}, so its synaptic half must start
    filtering at I_{t+2} -- the mirror of the forward S_{n,t}, which starts at I_{t-1}. A
    buffer whose row 0 is time t therefore has to be advanced by one row before the kernel
    sum, or the synaptic half sits at t while R stays at t+1. """
    torch.manual_seed(11)
    K = 7
    alpha = 0.45
    I = torch.randn(K, dtype=torch.float64)        # I[j] = input at time t + j

    S = anchor_reverse_synaptic_term(I, alpha)
    assert S.shape == (K,)
    assert S[0] == 0                               # no future filtered at reverse age 0

    # closed form straight off the theory: S^dagger_{t+1}(n) = sum_{tau=1..n} kappa_tau I_{t+1+tau}
    for n in range(1, K - 1):
        expected = sum(alpha * (1 - alpha) ** (tau - 1) * I[1 + tau]
                       for tau in range(1, n + 1))
        assert torch.allclose(S[n], expected, rtol=0, atol=1e-15)

    # the censoring lag would need I_{t+K}, which the buffer does not reach; it clamps to
    # the last resolvable lag, exact once S^dagger has saturated (the lumped-bin assumption)
    assert torch.equal(S[K - 1], S[K - 2])

    # and it is emphatically NOT the raw buffer's field -- that one is anchored at t
    raw = compute_reverse_synaptic_term(I, alpha)
    assert not torch.allclose(S[1:K - 1], raw[1:K - 1])

    # batched: (B, K) buffer with a (B, 1) alpha is the same field row by row
    S_b = anchor_reverse_synaptic_term(I.expand(3, K).contiguous(),
                                          torch.full((3, 1), alpha, dtype=torch.float64))
    assert torch.allclose(S_b, S.expand(3, K), rtol=0, atol=1e-15)


def test_reverse_field_anchoring_is_wired_through_the_trajectory():
    """ End-to-end guard on the anchoring: entropy_trajectory must read the reverse field off
    the buffer row AFTER the reported one. Constant input cannot see the offset (the shift is
    a no-op), so this drives the network in an oscillatory regime where I_t keeps moving, and
    pins H_rev against a hand-rolled reference that anchors explicitly. """
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        dt = 0.5
        net = RDMIsingModel(4, 1, 30, 1, 12, 3, 0, dt, hazard="synchronous",
                                eps=0.01, device="cpu")
        net.forward(300)
        out = net.entropy_trajectory(40, pb=False)

        # a shifted anchoring changes H_rev (and hence sigma) but leaves H_fwd alone --
        # so a silent regression to the t-anchored field would show up right here
        net2 = RDMIsingModel(4, 1, 30, 1, 12, 3, 0, dt, hazard="synchronous",
                                 eps=0.01, device="cpu")
        net2.forward(300)
        # The name has to be patched in the module that READS it, not in the rdme
        # package namespace: the default path is the sliding one and step_epr resolves the
        # name through rdme.epr's own globals, so patching the package re-export is a no-op
        # and the test would pass vacuously. Both modules are patched so the check holds
        # whichever path runs.
        mods = [rdme.epr, rdme.single]
        real = [m.anchor_reverse_synaptic_term for m in mods]
        for m in mods:
            m.anchor_reverse_synaptic_term = compute_reverse_synaptic_term   # the old bug
        try:
            bad = net2.entropy_trajectory(40, pb=False)
        finally:
            for m, r in zip(mods, real):
                m.anchor_reverse_synaptic_term = r

        assert torch.allclose(out["H_fwd_tot"], bad["H_fwd_tot"])    # forward half untouched
        assert not torch.allclose(out["H_rev_tot"], bad["H_rev_tot"])
        assert out["H_rev_tot"].mean() < bad["H_rev_tot"].mean()     # the bug inflates H_rev
    finally:
        torch.set_default_dtype(old)


def _naive_joint_distribution(p, buff_S, R, beta, theta):
    """ Unvectorized reference for compute_joint_distribution: loops over (n, k) explicitly
    and tracks the survival probability as a running product, rather than the closed-form
    log-space cumsum the real implementation uses. """
    Q, K = buff_S.shape[1], buff_S.shape[0]
    P_joint = torch.zeros(Q, K, dtype=p.dtype)
    for n in range(Q):
        S = 1.0
        for k in range(K):
            age = min(n + k, Q - 1)
            phi = torch.sigmoid(beta * (buff_S[k, age] + R[age] - theta))
            if k == K - 1:
                P_joint[n, k] = p[n] * S
            else:
                P_joint[n, k] = p[n] * S * phi
                S = S * (1 - phi)
    return P_joint


def test_compute_joint_distribution():
    Q = K = 6
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_S = torch.randn(K, Q, dtype=torch.float64) * 0.5
    R = -torch.rand(Q, dtype=torch.float64) * 3
    beta, theta = 2.0, 0.3

    got = joint_distribution(p, buff_S, R, beta, theta, HZ, LAM0, DT)
    expected = _naive_joint_distribution(p, buff_S, R, beta, theta)
    assert torch.allclose(got, expected, atol=1e-10)

    # each row is a first-passage-time distribution given starting age n, so it telescopes
    # back to exactly p(n) -- the whole reason no separate normalisation step is needed
    # (see the function's docstring)
    assert torch.allclose(got.sum(dim=1), p)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_distribution_survives_saturated_hazards(dtype):
    # a large beta saturates the hazard to 1 in the working dtype; the clamp before
    # log1p(-phi) has to be wide enough for that dtype or the survival goes -inf and every
    # EPR downstream is NaN (1 - 1e-12 rounds to exactly 1 in float32)
    Q = K = 6
    p = torch.full((Q,), 1.0 / Q, dtype=dtype)
    buff_S = torch.full((K, Q), 50.0, dtype=dtype)
    R = torch.zeros(Q, dtype=dtype)

    joint = joint_distribution(p, buff_S, R, 1e4, 0.0, HZ, LAM0, DT)
    assert torch.isfinite(joint).all()
    assert torch.allclose(joint.sum(dim=1), p, atol=1e-6)

    # float64 must keep the historical bound exactly, so this is not a silent numerics change
    assert dyn.hazard_clamp(torch.float64) == 1.0 - 1e-12


def _naive_epr(p_joint, S_fwd, S_rev, R, beta, theta):
    """ Unvectorized reference for compute_epr's H_fwd/H_rev sums: loops over the age/lag
    axis explicitly instead of the vectorized dim=0/dim=1 reductions, so an axis or
    transpose slip in the real implementation would not silently reproduce here too. """
    softplus = torch.nn.functional.softplus
    H_fwd = torch.zeros((), dtype=p_joint.dtype)
    for n in range(p_joint.shape[0]):
        hf = beta * (S_fwd[n] + R[n] - theta)
        H_fwd = H_fwd + p_joint[n, :].sum() * softplus(hf) - p_joint[n, 0] * hf
    H_rev = torch.zeros((), dtype=p_joint.dtype)
    for k in range(p_joint.shape[1]):
        hr = beta * (S_rev[k] + R[k] - theta)
        H_rev = H_rev + p_joint[:, k].sum() * softplus(hr) - p_joint[0, k] * hr
    return H_fwd, H_rev


def test_compute_epr():
    Q = K = 5
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_S = torch.randn(K, Q, dtype=torch.float64) * 0.5
    buff_in = torch.randn(K, dtype=torch.float64)
    R = -torch.rand(Q, dtype=torch.float64) * 2
    beta, theta, alpha = 1.8, 0.2, 0.3

    p_joint = joint_distribution(p, buff_S, R, beta, theta, HZ, LAM0, DT)
    S_fwd = buff_S[0]
    S_rev = compute_reverse_synaptic_term(buff_in, alpha)

    epr = epr_from_joint(p_joint, _phi(S_fwd, R, beta, theta),
                             _phi(S_rev, R, beta, theta))
    assert epr.shape == (3,)

    sigma, H_fwd, H_rev = epr
    assert torch.allclose(sigma, H_rev - H_fwd)  # sigma is definitionally H_rev - H_fwd

    expected_S_fwd, expected_S_rev = _naive_epr(p_joint, S_fwd, S_rev, R, beta, theta)
    assert torch.allclose(H_fwd, expected_S_fwd)
    assert torch.allclose(H_rev, expected_S_rev)


def test_epr_from_joint_matches_epr_from_parts():

    Q = K = 6
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_S = torch.randn(K, Q, dtype=torch.float64) * 0.5
    buff_in = torch.randn(K, dtype=torch.float64)
    R = -torch.rand(Q, dtype=torch.float64) * 2
    alpha, beta, theta = 0.35, 2.0, 0.25

    S_rev = anchor_reverse_synaptic_term(buff_in, alpha)
    J = joint_distribution(p, buff_S, R, beta, theta, HZ, LAM0, DT)
    phi_f, phi_r = _phi(buff_S[0], R, beta, theta), _phi(S_rev, R, beta, theta)

    # epr_from_joint is a thin wrapper that pulls four marginals out of the joint and hands
    # them to epr_from_parts; the sliding path reaches the same four by recursion, so a slip
    # in which marginal goes where would make the two paths disagree everywhere else. This
    # pins the wrapper itself.
    got = epr_from_joint(J, phi_f, phi_r)
    expected = epr_from_parts(J.sum(dim=-1), J.sum(dim=-2), J[:, 0], J[0, :],
                                  phi_f, phi_r)
    assert torch.equal(got, expected)

    # and sigma is definitionally the difference of the two halves
    assert torch.allclose(got[0], got[2] - got[1], rtol=0, atol=1e-15)


@contextlib.contextmanager
def _float64():
    """ The sliding recursions are compared against the reference at float64 tolerances,
    which needs the model's own tensors built at float64 too (see RDMNetwork.__init__). """
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def test_init_epr_state_matches_joint():
    # the state the recursions start from has to agree with the full joint it replaces
    with _float64():
        Q = K = 7
        p = torch.rand(Q); p /= p.sum()
        buff_S = torch.randn(K, Q) * 0.5
        R = -torch.rand(Q) * 3
        beta, theta = 2.0, 0.3

        st = init_epr_state(p, buff_S, R, beta, theta, HZ, LAM0, DT)
        assert torch.allclose(st.G, joint_distribution(p, buff_S, R, beta, theta, HZ, LAM0, DT).sum(dim=0))

        # V(n) is the survival of the age-n cohort over the first K-2 steps, and phi_end the
        # hazard it meets at the window edge -- both on clamped (lumped-bin) ages
        n = torch.arange(Q)
        surv = torch.ones(Q)
        for j in range(K - 2):
            a = torch.clamp(n + j, max=Q - 1)
            surv = surv * (1 - torch.sigmoid(beta * (buff_S[j, a] + R[a] - theta)))
        assert torch.allclose(st.logV.exp(), surv)

        a_end = torch.clamp(n + K - 2, max=Q - 1)
        assert torch.allclose(st.phi_end,
                              torch.sigmoid(beta * (buff_S[K - 2, a_end] + R[a_end] - theta)))


@pytest.mark.parametrize("Q", [4, 9])
def test_step_epr_matches_reference_over_long_run(Q):
    """ step_epr must reproduce compute_epr_from_buffers step for step, driven through the
    ring buffer exactly as entropy_trajectory drives it. Run long enough that drift in logV
    -- the one accumulator that is not rebuilt within a K-step window -- would show up.
    Q=4 keeps the two boundary lags (K-2, K-1) adjacent to the shifted interior. """
    with _float64():
        T = 600
        net = _small_network([Q])
        P, Y, IN = [], [], []
        for _ in range(T + Q + 1):
            I = net.total_input()
            P.append(net.p[0].clone())
            Y.append(net.S[0].clone())
            IN.append(I[0].clone())
            net.update()

        R, alpha, beta, theta = net.R[0], net.alpha_int[0], net.beta[0], net.theta[0]
        ref = torch.stack([
            _epr_from_buffers(P[t], torch.stack(Y[t:t + Q]), torch.stack(IN[t:t + Q]),
                                        R, alpha, beta, theta)
            for t in range(T)])

        idx = make_epr_indices(Q, Q, 'cpu')
        buff_S, buff_in = torch.stack(Y[:Q]).clone(), torch.stack(IN[:Q]).clone()
        state = init_epr_state(P[0], buff_S, R, beta, theta, HZ, LAM0, DT)
        p, got = P[0].clone(), []
        for t in range(T):
            epr, p, state = step_epr(p, state, buff_S, buff_in, idx, t,
                                         R, alpha, beta, theta, HZ, LAM0, DT)
            got.append(epr)
            assert torch.allclose(p, P[t + 1], atol=1e-12), f"marginal drifted at t={t}"
            # retire the row just consumed (time t); time t+Q takes over its slot
            buff_S[t % Q], buff_in[t % Q] = Y[t + Q], IN[t + Q]

        assert torch.allclose(torch.stack(got), ref, atol=1e-10)


def _small_network(Qm, seed=0):
    """ A small, fully deterministic RDMNetwork (construction draws no randomness beyond
    w, which is reseeded here) sized only for fast tests -- Qm is passed explicitly rather
    than auto-sized from tau_int/tau_ref. """
    torch.manual_seed(seed)
    M = len(Qm)
    w = torch.randn(M, M) * 0.5
    return RDMNetwork(
        M=M, w=w, N=[1000] * M,
        E=[0.3] * M, beta=[3.0] * M, theta=[0.4] * M,
        tau_int=[3.0] * M, tau_ref=[5.0] * M, K_ref=[2.0] * M,
        Qm=Qm, hazard="synchronous",
    )


def _ground_truth_epr(model, T):
    """ Reference for entropy_trajectory: store the model's entire p/S/I history
    and evaluate compute_epr_from_buffers directly against the exact window each
    population needs at each t -- no rolling buffers, no pending queue. """
    M, Qm = model.M, model.Qm
    P = [[] for _ in range(M)]
    Y = [[] for _ in range(M)]
    IN = [[] for _ in range(M)]
    for _ in range(T + max(Qm) + 1):
        I = model.total_input()
        for i in range(M):
            P[i].append(model.p[i].clone())
            Y[i].append(model.S[i].clone())
            IN[i].append(I[i].clone())
        model.update()

    out = {"sigma": torch.zeros(T, M), "H_fwd": torch.zeros(T, M), "H_rev": torch.zeros(T, M)}
    for t in range(T):
        for i in range(M):
            Q = Qm[i]
            buff_S = torch.stack(Y[i][t:t + Q])
            buff_in = torch.stack(IN[i][t:t + Q])
            epr = _epr_from_buffers(
                P[i][t], buff_S, buff_in, model.R[i], model.alpha_int[i],
                model.beta[i], model.theta[i])
            out["sigma"][t, i], out["H_fwd"][t, i], out["H_rev"][t, i] = epr
    return out


@pytest.mark.parametrize("Qm", [
    [6, 6],   # equal Qm: no pending delay line at all
    [9, 4],   # unequal Qm: exercises the delay line the historical bug lived in
])
def test_entropy_trajectory_matches_ground_truth(Qm):
    T = 30
    ref = _ground_truth_epr(_small_network(Qm), T)
    got = _small_network(Qm).entropy_trajectory(T, pb=False)
    for k in ("sigma", "H_fwd", "H_rev"):
        assert torch.allclose(ref[k], got[k], atol=1e-6), f"{k} diverged from ground truth"


def test_entropy_trajectory_rewinds_live_state():
    # entropy_trajectory buffers a Q_max-step lookahead internally and must rewind self.p/
    # self.S/self.m back to the reported time T before returning, so a caller can keep
    # stepping the model as if entropy_trajectory had never peeked ahead
    T = 20
    Qm = [9, 4]
    stepped, peeked = _small_network(Qm), _small_network(Qm)
    peeked.entropy_trajectory(T, pb=False)
    stepped.forward(T)
    for i in range(len(Qm)):
        assert torch.allclose(peeked.p[i], stepped.p[i], atol=1e-6)
        assert torch.allclose(peeked.S[i], stepped.S[i], atol=1e-6)
    assert torch.allclose(peeked.m, stepped.m, atol=1e-6)


def _small_batch(Qm, B, seed=1):
    """ A small RDMNetworkBatch whose B batch elements carry distinct parameters, so a bug
    that only shows up when elements diverge (e.g. a broadcast collapsing the batch axis)
    has something to trip on. """
    torch.manual_seed(seed)
    M = len(Qm)
    w = torch.randn(B, M, M) * 0.5
    beta  = torch.linspace(2.5, 4.0, B).unsqueeze(1).expand(B, M).contiguous()
    E     = torch.linspace(0.1, 0.5, B).unsqueeze(1).expand(B, M).contiguous()
    const = lambda v: torch.full((B, M), float(v))
    return RDMNetworkBatch(
        M=M, w=w, N=[1000] * M,
        E=E, beta=beta, theta=const(0.4),
        tau_int=const(3.0), tau_ref=const(5.0), K_ref=const(2.0),
        Qm=Qm, hazard="synchronous",
    )


def test_batch_entropy_trajectory_matches_independent_networks():
    T, Qm, B = 25, [8, 5], 3
    batch = _small_batch(Qm, B)
    batch_out = batch.entropy_trajectory(T, pb=False)

    for b in range(B):
        single = RDMNetwork(
            M=batch.M, w=batch.w[b], N=batch.N,
            E=batch.E[b].tolist(), beta=batch.beta[b].tolist(), theta=batch.theta[b].tolist(),
            tau_int=batch.tau_int[b].tolist(), tau_ref=batch.tau_ref[b].tolist(),
            K_ref=batch.K_ref[b].tolist(), Qm=Qm,
        )
        single_out = single.entropy_trajectory(T, pb=False)
        for k in ("sigma", "H_fwd", "H_rev"):
            # the batch pins its accumulators to float64 (see _out_device_kwargs) while the
            # single network follows the ambient default dtype, so compare values in a
            # common dtype rather than tripping over the mismatch
            assert torch.allclose(batch_out[k][b].double(), single_out[k].double(), atol=1e-5), \
                f"{k} mismatch at batch index {b}"


@pytest.mark.parametrize("chunk", [1, 2])
def test_batch_entropy_trajectory_chunk_matches_unchunked(chunk):
    T, Qm, B = 15, [8, 5], 4
    # chunk splits the (B, Q, K) joint, so it is the joint path that has anything to chunk;
    # the sliding path never builds that object and ignores the argument
    ref = _small_batch(Qm, B).entropy_trajectory(T, method="joint", pb=False)
    got = _small_batch(Qm, B).entropy_trajectory(T, method="joint", pb=False, chunk=chunk)
    for k in ref:
        assert torch.allclose(ref[k], got[k], atol=1e-5), f"{k} diverged at chunk={chunk}"
