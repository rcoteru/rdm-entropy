""" Tests for the entropy-production machinery in mean_field.py: the low-level
compute_joint_distribution/compute_epr/compute_epr_from_buffers formulas, and the
RDMNetwork(Batch).entropy_trajectory buffering built on top of them. """

import pytest
import torch

import rdme.mean_field as mf

torch.manual_seed(0)


# Low-level formulas
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_compute_backward_field():
    torch.manual_seed(4)
    Q = 8
    alpha = 0.4
    pop_input = torch.randn(Q, dtype=torch.float64)

    H = mf.compute_backward_field(pop_input, alpha)

    # no field has accumulated yet at age 0
    assert H[0] == 0

    # closed form: H[j] = sum_{m=1}^{j} alpha*(1-alpha)^(m-1) * pop_input[m], i.e. a kernel-
    # weighted cumulative sum over the buffered future inputs
    expected = torch.zeros(Q, dtype=torch.float64)
    for j in range(1, Q):
        expected[j] = sum(
            alpha * (1 - alpha) ** (m - 1) * pop_input[m] for m in range(1, j + 1))
    assert torch.allclose(H, expected)

    # alpha=1: the kernel is [1, 0, 0, ...], so the field is pinned at pop_input[1]
    # for every j >= 1 rather than tracking the most recent input
    H1 = mf.compute_backward_field(pop_input, 1.0)
    assert torch.allclose(H1[1:], torch.full_like(H1[1:], pop_input[1].item()))


def _naive_joint_distribution(p, buff_y, eta, beta, theta):
    """ Unvectorized reference for compute_joint_distribution: loops over (n, k) explicitly
    and tracks the survival probability as a running product, rather than the closed-form
    log-space cumsum the real implementation uses. """
    Q, K = buff_y.shape[1], buff_y.shape[0]
    P_joint = torch.zeros(Q, K, dtype=p.dtype)
    for n in range(Q):
        S = 1.0
        for k in range(K):
            age = min(n + k, Q - 1)
            phi = torch.sigmoid(beta * (buff_y[k, age] + eta[age] - theta))
            if k == K - 1:
                P_joint[n, k] = p[n] * S
            else:
                P_joint[n, k] = p[n] * S * phi
                S = S * (1 - phi)
    return P_joint


def test_compute_joint_distribution():
    Q = K = 6
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_y = torch.randn(K, Q, dtype=torch.float64) * 0.5
    eta = -torch.rand(Q, dtype=torch.float64) * 3
    beta, theta = 2.0, 0.3

    got = mf.compute_joint_distribution(p, buff_y, eta, beta, theta)
    expected = _naive_joint_distribution(p, buff_y, eta, beta, theta)
    assert torch.allclose(got, expected, atol=1e-10)

    # each row is a first-passage-time distribution given starting age n, so it telescopes
    # back to exactly p(n) -- the whole reason no separate normalisation step is needed
    # (see the function's docstring)
    assert torch.allclose(got.sum(dim=1), p)


def _naive_epr(p_joint, y_fwd, y_rev, eta, beta, theta):
    """ Unvectorized reference for compute_epr's S_fwd/S_rev sums: loops over the age/lag
    axis explicitly instead of the vectorized dim=0/dim=1 reductions, so an axis or
    transpose slip in the real implementation would not silently reproduce here too. """
    softplus = torch.nn.functional.softplus
    S_fwd = torch.zeros((), dtype=p_joint.dtype)
    for n in range(p_joint.shape[0]):
        hf = beta * (y_fwd[n] + eta[n] - theta)
        S_fwd = S_fwd + p_joint[n, :].sum() * softplus(hf) - p_joint[n, 0] * hf
    S_rev = torch.zeros((), dtype=p_joint.dtype)
    for k in range(p_joint.shape[1]):
        hr = beta * (y_rev[k] + eta[k] - theta)
        S_rev = S_rev + p_joint[:, k].sum() * softplus(hr) - p_joint[0, k] * hr
    return S_fwd, S_rev


def test_compute_epr():
    Q = K = 5
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_y = torch.randn(K, Q, dtype=torch.float64) * 0.5
    buff_in = torch.randn(K, dtype=torch.float64)
    eta = -torch.rand(Q, dtype=torch.float64) * 2
    beta, theta, alpha = 1.8, 0.2, 0.3

    p_joint = mf.compute_joint_distribution(p, buff_y, eta, beta, theta)
    y_fwd = buff_y[0]
    y_rev = mf.compute_backward_field(buff_in, alpha)

    epr = mf.compute_epr(p, p_joint, y_fwd, y_rev, eta, beta, theta)
    assert epr.shape == (3,)

    sigma, S_fwd, S_rev = epr
    assert torch.allclose(sigma, S_rev - S_fwd)  # sigma is definitionally S_bw - S_fwd

    expected_S_fwd, expected_S_rev = _naive_epr(p_joint, y_fwd, y_rev, eta, beta, theta)
    assert torch.allclose(S_fwd, expected_S_fwd)
    assert torch.allclose(S_rev, expected_S_rev)


def test_compute_epr_from_buffers():
    # regression test on the wiring alone: compute_epr_from_buffers should be exactly the
    # composition of the three functions above, nothing more
    Q = K = 6
    p = torch.rand(Q, dtype=torch.float64); p /= p.sum()
    buff_y = torch.randn(K, Q, dtype=torch.float64) * 0.5
    buff_in = torch.randn(K, dtype=torch.float64)
    eta = -torch.rand(Q, dtype=torch.float64) * 2
    alpha, beta, theta = 0.35, 2.0, 0.25

    got = mf.compute_epr_from_buffers(p, buff_y, buff_in, eta, alpha, beta, theta)

    H_rev = mf.compute_backward_field(buff_in, alpha)
    P_joint = mf.compute_joint_distribution(p, buff_y, eta, beta, theta)
    expected = mf.compute_epr(p, P_joint, buff_y[0], H_rev, eta, beta, theta)

    assert torch.equal(got, expected)


# entropy_trajectory: RDMNetwork
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _small_network(Qm, seed=0):
    """ A small, fully deterministic RDMNetwork (construction draws no randomness beyond
    w, which is reseeded here) sized only for fast tests -- Qm is passed explicitly rather
    than auto-sized from tau_int/tau_ref. """
    torch.manual_seed(seed)
    M = len(Qm)
    w = torch.randn(M, M) * 0.5
    return mf.RDMNetwork(
        M=M, w=w, N=[1000] * M,
        I=[0.3] * M, beta=[3.0] * M, theta=[0.4] * M,
        tau_int=[3.0] * M, tau_ref=[5.0] * M, K_ref=[2.0] * M,
        Qm=Qm,
    )


def _ground_truth_epr(model, T):
    """ Reference for entropy_trajectory: store the model's entire p/y/pop_input history
    and evaluate compute_epr_from_buffers directly against the exact window each
    population needs at each t -- no rolling buffers, no pending queue. """
    M, Qm = model.M, model.Qm
    P = [[] for _ in range(M)]
    Y = [[] for _ in range(M)]
    IN = [[] for _ in range(M)]
    for _ in range(T + max(Qm) + 1):
        pop_input = model.population_input()
        for i in range(M):
            P[i].append(model.p[i].clone())
            Y[i].append(model.y[i].clone())
            IN[i].append(pop_input[i].clone())
        model.update()

    out = {"sigma": torch.zeros(T, M), "S_fwd": torch.zeros(T, M), "S_rev": torch.zeros(T, M)}
    for t in range(T):
        for i in range(M):
            Q = Qm[i]
            buff_y = torch.stack(Y[i][t:t + Q])
            buff_in = torch.stack(IN[i][t:t + Q])
            epr = mf.compute_epr_from_buffers(
                P[i][t], buff_y, buff_in, model.eta[i], model.alpha_int[i],
                model.beta[i], model.theta[i])
            out["sigma"][t, i], out["S_fwd"][t, i], out["S_rev"][t, i] = epr
    return out


@pytest.mark.parametrize("Qm", [
    [6, 6],   # equal Qm: no pending delay line at all
    [9, 4],   # unequal Qm: exercises the delay line the historical bug lived in
])
def test_entropy_trajectory_matches_ground_truth(Qm):
    T = 30
    ref = _ground_truth_epr(_small_network(Qm), T)
    got = _small_network(Qm).entropy_trajectory(T)
    for k in ("sigma", "S_fwd", "S_rev"):
        assert torch.allclose(ref[k], got[k], atol=1e-6), f"{k} diverged from ground truth"


def test_entropy_trajectory_rewinds_live_state():
    # entropy_trajectory buffers a Q_max-step lookahead internally and must rewind self.p/
    # self.y/self.m back to the reported time T before returning, so a caller can keep
    # stepping the model as if entropy_trajectory had never peeked ahead
    T = 20
    Qm = [9, 4]
    stepped, peeked = _small_network(Qm), _small_network(Qm)
    peeked.entropy_trajectory(T)
    stepped.forward(T)
    for i in range(len(Qm)):
        assert torch.allclose(peeked.p[i], stepped.p[i], atol=1e-6)
        assert torch.allclose(peeked.y[i], stepped.y[i], atol=1e-6)
    assert torch.allclose(peeked.m, stepped.m, atol=1e-6)


# entropy_trajectory: RDMNetworkBatch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _small_batch(Qm, B, seed=1):
    """ A small RDMNetworkBatch whose B batch elements carry distinct parameters, so a bug
    that only shows up when elements diverge (e.g. a broadcast collapsing the batch axis)
    has something to trip on. """
    torch.manual_seed(seed)
    M = len(Qm)
    w = torch.randn(B, M, M) * 0.5
    beta  = torch.linspace(2.5, 4.0, B).unsqueeze(1).expand(B, M).contiguous()
    I     = torch.linspace(0.1, 0.5, B).unsqueeze(1).expand(B, M).contiguous()
    const = lambda v: torch.full((B, M), float(v))
    return mf.RDMNetworkBatch(
        M=M, w=w, N=[1000] * M,
        I=I, beta=beta, theta=const(0.4),
        tau_int=const(3.0), tau_ref=const(5.0), K_ref=const(2.0),
        Qm=Qm,
    )


def test_batch_entropy_trajectory_matches_independent_networks():
    T, Qm, B = 25, [8, 5], 3
    batch = _small_batch(Qm, B)
    batch_out = batch.entropy_trajectory(T, pb=False)

    for b in range(B):
        single = mf.RDMNetwork(
            M=batch.M, w=batch.w[b], N=batch.N,
            I=batch.I[b].tolist(), beta=batch.beta[b].tolist(), theta=batch.theta[b].tolist(),
            tau_int=batch.tau_int[b].tolist(), tau_ref=batch.tau_ref[b].tolist(),
            K_ref=batch.K_ref[b].tolist(), Qm=Qm,
        )
        single_out = single.entropy_trajectory(T)
        for k in ("sigma", "S_fwd", "S_rev"):
            assert torch.allclose(batch_out[k][b], single_out[k], atol=1e-5), \
                f"{k} mismatch at batch index {b}"


@pytest.mark.parametrize("chunk", [1, 2])
def test_batch_entropy_trajectory_chunk_matches_unchunked(chunk):
    T, Qm, B = 15, [8, 5], 4
    ref = _small_batch(Qm, B).entropy_trajectory(T, pb=False)
    got = _small_batch(Qm, B).entropy_trajectory(T, pb=False, chunk=chunk)
    for k in ref:
        assert torch.allclose(ref[k], got[k], atol=1e-5), f"{k} diverged at chunk={chunk}"
