from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple

import torch
import tqdm

import rdme.fixed_points as fpts
import rdme.shared as shrd

# Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_total_input(
        m: torch.Tensor, # (M,) or (B, M) overlaps
        w: torch.Tensor, # (M, M) or (B, M, M) pop connection weights
        E: torch.Tensor, # (M,) or (B, M) pop external drives
        ) -> torch.Tensor:
    """ Total input to each population: I^a = sum_b w[a, b] m^b + E^a, i.e. w @ m + E.
    Row = target, column = source, matching w^{ab} ("from population b onto a") in the theory. """
    return (w @ m.unsqueeze(-1)).squeeze(-1) + E

def update_synaptic_term(S: torch.Tensor,               # (Q,) or (B, Q) synaptic term
                         alpha: float | torch.Tensor,  # synaptic decay factor
                         I: float | torch.Tensor,      # total input to the population
                         inplace: bool = False
                         ) -> torch.Tensor:
    """ Synaptic-term update: S_new(n+1) = alpha*I + (1-alpha)*S(n), with S_new(0) = 0. """
    if inplace:
        shifted = S[..., :-1].mul(1 - alpha)
        shifted.add_(alpha * I)
        S[..., 1:] = shifted
        S[..., 0] = 0
        return S
    else:
        S_new = torch.zeros_like(S)
        S_new[..., 1:] = alpha * I + (1 - alpha) * S[..., :-1]
        return S_new

def compute_firing_prob(
        S:     torch.Tensor,        # (Q,) integrated field
        R:   torch.Tensor,          # (Q,) refractory kernel
        beta:  float | torch.Tensor,
        theta: float | torch.Tensor,
        inplace: bool = False
        ) -> torch.Tensor:            # (Q,) hazard Phi(beta*(S + R - theta))
    """ Age-resolved firing probability from an integrated field. Broadcasts, so it also
    serves batched (B, Q) fields given (B, 1)-shaped beta/theta. """
    if inplace:
        return (S+R).sub_(theta).mul_(beta).sigmoid_()
    else:
        return torch.sigmoid(beta * (S + R - theta))

def compute_firing_rate(P: torch.Tensor, fprobs: torch.Tensor) -> torch.Tensor:
    """ Firing rate for one population: sum(P * fprobs) over the age (LAST) axis. """
    if P.ndim == 1:
        return torch.dot(P, fprobs)
    # einsum is a bit slower than the fused dot for 1-D, but it scales better to (B, Q)
    return torch.einsum('...q,...q->...', P, fprobs)

def update_age_distribution(P: torch.Tensor,
                            fprobs: torch.Tensor,
                            inplace: bool = False
                            ) -> torch.Tensor:
    """ Renewal update: births into bin 0, survival-decay into interior bins, remainder
    absorbed into the last (boundary) bin to keep P normalized. """
    if inplace:
        frate = compute_firing_rate(P, fprobs)
        interior = fprobs[..., :-2].neg().add_(1)  # 1 - fprobs[..., :-2], one fresh buffer
        interior.mul_(P[..., :-2])                 # fuse in the multiply, no second buffer
        P[..., 0] = frate                          # interior is already materialised, so
        P[..., 1:-1] = interior                    # the overlapping read/write is safe
        P[..., -1] = 1 - frate - interior.sum(dim=-1)
        return P
    else:
        P_new = torch.zeros_like(P)
        P_new[..., 0]    = (P * fprobs).sum(dim=-1)
        P_new[..., 1:-1] = P[..., :-2] * (1 - fprobs[..., :-2])
        P_new[..., -1]   = 1 - P_new[..., :-1].sum(dim=-1)
        return P_new

def check_age_normalization(P: torch.Tensor, tol: float = 1e-6) -> None:
    """ Check that the population distribution P is normalized. """
    if not torch.allclose(P.sum(), torch.tensor(1.0, dtype=P.dtype, device=P.device), atol=tol):
        raise ValueError(f"Population distribution not normalized: sum={P.sum().item()}")


# Auxiliary functions for entropy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_reverse_synaptic_term(
        I: torch.Tensor,       # (Qm[i],) buffered I[j] = input at time t+j
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        ) -> torch.Tensor:             # (Qm[i],) backward field S^dagger_t(n)
    """ Age-dependent backward (time-reversed) field from a buffered trajectory of future
    population inputs, via a kernel-weighted cumulative sum:

        S^dagger(n) = sum_{tau=1..n} kappa_tau * I[tau],   kappa_tau = alpha (1-alpha)^(tau-1)

    so the field is anchored at the buffer's OWN row 0: it reads I[1] onwards and never
    touches I[0]. Callers that need the field at t+1 -- which is every EPR caller, since the
    EPR pairs h^dagger(n^dagger_{t+1}) against R(n^dagger_{t+1}) -- must hand over a buffer
    that starts at t+1, not at t. Use anchor_reverse_synaptic_term for that.

    Acts on the last axis only, so it takes a bare (K,) buffer (as it does under vmap) or a
    batched (B, K) one with a (B, 1)-shaped alpha, without a separate batched variant. """
    Q = I.shape[-1]
    n_idx = torch.arange(1, Q, device=I.device, dtype=I.dtype)
    kernel = alpha * (1.0 - alpha) ** (n_idx - 1)
    return torch.cat([I.new_zeros(I.shape[:-1] + (1,)),
                      torch.cumsum(kernel * I[..., 1:], dim=-1)], dim=-1)

def anchor_reverse_synaptic_term(
        I: torch.Tensor,               # (..., K) buffered I[j] = input at time t+j, j=0 is t
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        ) -> torch.Tensor:             # (..., K) S^dagger_{t+1}(n^dagger)
    """ The reverse synaptic term as the EPR needs it: anchored at t+1, from a buffer whose
    row 0 is time t.

    The EPR's reverse conditional is h^dagger_{n^dagger, t+1} = S^dagger_{n^dagger, t+1}
    + R(n^dagger) - theta, evaluated over the reverse age n^dagger_{t+1}, so the synaptic
    term has to start filtering at I_{t+2} -- the exact mirror of the forward field, whose
    recursion S_{n,t} = alpha*I_{t-1} + (1-alpha)*S_{n-1,t-1} starts one step behind its own
    index. Feeding the raw buffer to compute_reverse_synaptic_term instead starts it at
    I_{t+1}, which anchors the synaptic half at t while R stays at t+1 and inflates H_rev
    whenever the input is moving.

    The buffer only reaches t+K-1, so the last (censoring) lag n^dagger = K-1 would need an
    I_{t+K} that does not exist yet. It takes the K-2 value instead, which is exact under the
    same boundary assumption the lumped age bin already rests on: S^dagger has saturated by
    n^dagger = Q. """
    S = compute_reverse_synaptic_term(I[..., 1:], alpha)      # (..., K-1), anchored at t+1
    return torch.cat([S, S[..., -1:]], dim=-1)                # (..., K), censoring bin clamped

def hazard_clamp(dtype: torch.dtype) -> float:
    """ Upper bound to clamp a hazard to before taking log1p(-phi).

    1 - 1e-12 is below float32's resolution -- it rounds to exactly 1, so a saturated hazard
    turns the survival into -inf and every EPR downstream into NaN. The bound therefore
    follows the dtype's own resolution, while float64 keeps its historical 1 - 1e-12. """
    return 1.0 - max(1e-12, torch.finfo(dtype).eps)

def compute_joint_distribution(
        p:      torch.Tensor,          # (Q,)   p_t(n) age distribution at time t
        buff_S: torch.Tensor,          # (K, Q) buff_S[j, n] = S_{t+j}(n), j=0 is time t
        R:    torch.Tensor,          # (Q,)   refractory kernel
        beta:   float | torch.Tensor,
        theta:  float | torch.Tensor,
        ) -> torch.Tensor:             # (Q, K) P_joint[n, k] = p_t(n, n_dagger=k)
    """ Joint p_t(n, n_dagger=k) built from hazards, not density ratios.

    S_t(k|n) = prod_{j<k} (1 - Phi_{t+j}(n+j)) tracks a single cohort even
    after it enters the lumped bin, because it multiplies hazards instead of
    reading merged densities -- so only the marginal p_t is needed, never the
    future age distributions. Rows sum to 1 by telescoping -- no normalisation
    needed.

    This is entropy_trajectory's hot spot -- once per population per timestep, and under
    vmap every (Q, K) tensor here is really (B, Q, K) -- so it is written to keep as few of
    them alive as possible: hazards are evaluated once on buff_S's own (K, Q) layout and
    gathered onto the cohort diagonal in one pass (gathering S and R separately would
    cost one extra (Q, K) tensor each), and the survival is folded into the output buffer
    in place. Three (Q, K) tensors are live at peak. In-place ops are limited to the ones
    functorch has batching rules for -- clamp_/cumsum_ fall back to a per-sample loop
    under vmap, so those two stay out-of-place. """

    Q, device = buff_S.shape[1], buff_S.device
    K = buff_S.shape[0]                       # look-ahead horizon (time axis)
    n_idx = torch.arange(Q, device=device)
    j_idx = torch.arange(K, device=device)

    future_age  = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)
    future_time = j_idx.unsqueeze(0)

    phi = compute_firing_prob(buff_S, R, beta, theta)[future_time, future_age]   # (Q, K)

    # S[n, k] = prod_{j<k} (1 - phi[n, j]), in log space, with the k=0 empty product
    # already in place from the ones_like -- so the shift costs a copy, not a cat
    log_surv = torch.cumsum(phi.clamp(max=hazard_clamp(phi.dtype)).neg_().log1p_(), dim=1)
    surv = torch.ones_like(phi)                                            # (Q, K)
    surv[:, 1:].copy_(log_surv[:, :-1]).exp_()

    # first-passage weight S*phi, except the lumped final column, which keeps bare S
    surv[:, :-1].mul_(phi[:, :-1])
    return surv.mul_(p.unsqueeze(1))

def compute_epr(
        p_joint: torch.Tensor,         # (Q, K) P_joint[n, k] = p_t(n, n_dagger=k)
        S_fwd:   torch.Tensor,         # (Q,)   S_t(n)              forward integrated field
        S_rev:   torch.Tensor,         # (K,)   S^dagger_t(n_dagger) backward integrated field
        R:     torch.Tensor,           # (Q,)
        beta:    float | torch.Tensor,
        theta:   float | torch.Tensor,
        ) -> torch.Tensor:             # (3,) [sigma, H_fwd, H_rev]; sigma = H_rev - H_fwd
    """ sigma = H_rev - H_fwd, with

        H_fwd = sum_n p(n) softplus(hf(n))        -  sum_n P_joint[n,0] hf(n)
        H_rev  = sum_k p_dagger(k) softplus(hr(k)) -  sum_k P_joint[0,k] hr(k)

    softplus(h) = -log(1-Phi) and h = log(Phi) - log(1-Phi) are the no-spike and spike
    pieces of the per-bin Bernoulli log-likelihood log p(s) = s*h - softplus(h), for the
    logistic hazard Phi = sigmoid(h).

    The spike weights are indicators, not firing probabilities: the forward
    transition emits s(t+1), which happens iff k == 0; the reverse transition
    emits s(t), which happens iff n == 0.

    Index convention: hf[i] and hr[i] are both evaluated at age i, with
    S(0) = S_dagger(0) = 0. S_rev must be anchored at t+1. """

    hf = beta * (S_fwd + R - theta)
    hr = beta * (S_rev + R - theta)

    p_fwd = p_joint.sum(dim=1)   # p_t(n)
    p_rev = p_joint.sum(dim=0)   # p_t(n_dagger)

    H_fwd = torch.sum(p_fwd * torch.nn.functional.softplus(hf)) - torch.sum(p_joint[:, 0] * hf)
    H_rev  = torch.sum(p_rev * torch.nn.functional.softplus(hr)) - torch.sum(p_joint[0, :] * hr)

    return torch.stack([H_rev - H_fwd, H_fwd, H_rev])

def compute_epr_from_buffers(
        p:       torch.Tensor,         # (Q,)   p_t(n) age distribution at time t
        buff_S:  torch.Tensor,         # (K, Q) buff_S[j, n] = S_{t+j}(n), j=0 is time t
        buff_I: torch.Tensor,         # (K,)   buff_I[j] = population input at time t+j
        R:     torch.Tensor,         # (Q,)   refractory kernel
        alpha:   float | torch.Tensor, # integration kernel scaling factor
        beta:    float | torch.Tensor,
        theta:   float | torch.Tensor,
        ) -> torch.Tensor:             # (3,) [sigma, H_fwd, H_rev]
    """ One population's EPR for one timestep, straight from its rolling buffers.

    Keeping the joint distribution inside one function is what lets the batched caller
    vmap the whole step under a chunk_size: the (Q, K) joint never escapes, so peak memory
    follows the chunk rather than the batch, and only the (3,) result is materialised for
    every batch element. """
    S_rev   = anchor_reverse_synaptic_term(buff_I, alpha)
    P_joint = compute_joint_distribution(p, buff_S, R, beta, theta)
    return compute_epr(P_joint, buff_S[0], S_rev, R, beta, theta)


# Sliding-state EPR: the same quantity in O(Q) per step instead of O(Q^2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# compute_epr never reads the (Q, K) joint as a matrix -- it only ever needs four things
# from it, and each of those slides exactly from one timestep to the next, so nothing
# below ever materialises the joint after the first step. Writing Phi[tau, n] for the
# hazard at absolute time tau and age n (buffer row tau, clamped age), L = log1p(-Phi),
# and J_t for the joint at the reported time t:
#
#   1. row sums   J_t.sum(1) == p_t                     exactly (first-passage telescoping)
#   2. column 0   J_t[:, 0]  == p_t * Phi[t, .]         one hazard row
#   3. row 0      J_t[0, :]  is the age-0 cohort's first-passage law, driven only by the
#                 buffer diagonal Phi[t+j, min(j, Q-1)] -- one scalar per buffer row
#   4. column sums G_t = J_t.sum(0) slide, given the survival-window state
#                 V_t(n) = prod_{j<K-2} (1 - Phi[t+j, n+j]):
#                   G_{t+1}[j]   = G_t[j+1] + J_{t+1}[0, j]           for j <= K-3
#                   G_{t+1}[K-2] = sum_n p_{t+1}(n) V_{t+1}(n) * phi
#                   G_{t+1}[K-1] = sum_n p_{t+1}(n) V_{t+1}(n) * (1 - phi)
#                 with phi = Phi[t+K-1, min(n+K-2, Q-1)], and V itself sliding in log space
#                   log V_{t+1}(n+1) = log V_t(n) - L[t, n] + L[t+K-2, min(n+K-2, Q-1)]
#
# The cohort identity behind (4) is S_t(k|n) = (1 - Phi[t,n]) * S_{t+1}(k-1|n+1) together
# with p_{t+1}(n+1) = p_t(n)(1 - Phi[t,n]) -- the renewal update itself -- which is what
# turns a column sum into a shift-and-add. It survives the lumped bin because the two
# cohorts merged there share identical clamped future hazards.
#
# Every row consulted (t, t+K-2, t+K-1 and the two diagonals) lies inside the existing
# look-ahead window, so the buffering in entropy_trajectory is unchanged. Error cannot
# accumulate in G: an entry entered as a K-2 boundary term at most K steps earlier. Only
# logV is a long-lived accumulator, and it is a sliding window of exact adds/subtracts.


class EPRIndices(NamedTuple):
    """ Gather indices for the sliding EPR step, precomputed once per population.

    slots[h] is the buffer-row permutation for ring head h: logical row j (absolute time
    t+j) lives in physical slot slots[h][j], which is what lets entropy_trajectory retire
    a row by overwriting one slot instead of copying the whole (B, K, Q) buffer. """
    slots:  torch.Tensor   # (K, K) all ring rotations, slots[h, j] = (h + j) % K
    a_diag: torch.Tensor   # (K,)   ages along the age-0 cohort diagonal, min(j, Q-1)
    a_end:  torch.Tensor   # (Q,)   ages at the survival-window edge, min(n + K-2, Q-1)


class EPRState(NamedTuple):
    """ Everything the sliding step carries between timesteps, valid at reported time t.
    All fields are (..., ) leading-batch friendly: (K,)/(Q,) alone, or (B, K)/(B, Q). """
    G:       torch.Tensor   # (..., K) column sums of the joint,  G[k] = p_t(n_dagger = k)
    logV:    torch.Tensor   # (..., Q) log survival over the next K-2 steps, per age
    phi_end: torch.Tensor   # (..., Q) Phi[t+K-2, min(n+K-2, Q-1)], the window-edge hazards


def make_epr_indices(K: int, Q: int, device: str | torch.device) -> EPRIndices:
    """ Build the gather indices for a population with K look-ahead rows and Q age bins. """
    if K < 3:
        raise ValueError(f"sliding EPR needs a look-ahead of at least 3 rows, got K={K}")
    if K != Q:
        # compute_epr evaluates the reverse field's h_dagger(k) against R(k), so the lag
        # axis and the age axis have to be the same length -- a constraint the joint-based
        # path carries too, and one entropy_trajectory always satisfies (K = Qm[i] = Q)
        raise ValueError(f"sliding EPR needs K == Q (the lag and age axes share R), got K={K}, Q={Q}")
    j = torch.arange(K, device=device)
    n = torch.arange(Q, device=device)
    return EPRIndices(
        slots=(j.unsqueeze(0) + j.unsqueeze(1)) % K,
        a_diag=torch.clamp(j, max=Q - 1),
        a_end=torch.clamp(n + (K - 2), max=Q - 1),
    )


def _init_epr_core(p, buff_S, R, beta, theta):
    """ Unbatched build of the sliding state at the first reported timestep. This is the one
    O(Q*K) step in the whole trajectory: it goes through compute_joint_distribution on
    purpose, so the state the recursions start from is the reference implementation's. """
    Q, K = buff_S.shape[-1], buff_S.shape[-2]
    n_idx = torch.arange(Q, device=buff_S.device)
    j_idx = torch.arange(K, device=buff_S.device)
    ages = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)          # (Q, K)
    phi = compute_firing_prob(buff_S, R, beta, theta)[j_idx.unsqueeze(0), ages]   # (Q, K)

    G = compute_joint_distribution(p, buff_S, R, beta, theta).sum(dim=0)   # (K,)
    logV = torch.log1p(-phi[:, :K - 2].clamp(max=hazard_clamp(phi.dtype))).sum(dim=1)             # (Q,)
    return G, logV, phi[:, K - 2]


def init_epr_state(
        p:      torch.Tensor,          # (Q,) or (B, Q) age distribution at time t
        buff_S: torch.Tensor,          # (K, Q) or (B, K, Q), row j = S_{t+j}
        R:    torch.Tensor,          # (Q,) or (B, Q)
        beta:   float | torch.Tensor,  # scalar, or (B,) alongside a batched p
        theta:  float | torch.Tensor,
        chunk:  int | None = None,     # batch chunking for this one-off O(B*Q*K) build
        ) -> EPRState:
    """ Sliding state at the first reported timestep, from the full joint. Batched when p is
    2D, in which case beta/theta are (B,) and the build is vmapped under chunk_size=chunk --
    the only place in entropy_trajectory where a (chunk, Q, K) working set is allocated. """
    if p.ndim == 1:
        return EPRState(*_init_epr_core(p, buff_S, R, beta, theta))
    core = lambda *a: _init_epr_core(*a)
    return EPRState(*torch.vmap(core, chunk_size=chunk)(p, buff_S, R, beta, theta))


def step_epr(
        p:       torch.Tensor,         # (..., Q) age distribution at the reported time t
        state:   EPRState,             # sliding state at time t
        buff_S:  torch.Tensor,         # (..., K, Q) ring buffer of fields
        buff_I: torch.Tensor,         # (..., K)    ring buffer of population inputs
        idx:     EPRIndices,
        head:    int,                  # ring head: logical row j is at slot (head + j) % K
        R:     torch.Tensor,         # (..., Q)
        alpha:   float | torch.Tensor, # broadcastable against (..., Q): scalar or (B, 1)
        beta:    float | torch.Tensor,
        theta:   float | torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, EPRState]:
    """ One timestep of EPR plus the advanced state: (epr (..., 3), p_{t+1}, state_{t+1}).

    Drop-in for compute_epr_from_buffers over a whole trajectory, but O(Q) per step rather
    than O(Q^2) -- see the derivation above the class definitions. Three (..., Q) hazard
    evaluations and two (..., K) gathers replace the (Q, K) joint; nothing here allocates a
    tensor larger than the buffers already alive, so no vmap and no chunking are involved.

    p_{t+1} comes back from here rather than being stepped by the caller because it shares
    the Phi[t, .] row this function already needs. """
    K = buff_S.shape[-2]
    slots = idx.slots[head % K]

    # -- present: the hazard row at time t, shared by H_fwd, column 0 and the logV slide
    S_fwd = buff_S[..., slots[0], :]
    hf = beta * (S_fwd + R - theta)
    phi_t = torch.sigmoid(hf)  # (..., Q)
    col0 = p * phi_t                                                    # (..., Q)  J_t[:, 0]

    # -- row 0: the age-0 cohort's first-passage law, off the buffer diagonal
    d = compute_firing_prob(buff_S[..., slots, idx.a_diag], R[..., idx.a_diag],
                            beta, theta)                         # (..., K)
    log_surv = torch.cumsum(torch.log1p(-d.clamp(max=hazard_clamp(d.dtype))), dim=-1)
    surv0 = torch.cat([torch.ones_like(d[..., :1]), torch.exp(log_surv[..., :-1])], dim=-1)
    row0 = p[..., :1] * surv0
    row0 = torch.cat([row0[..., :-1] * d[..., :-1], row0[..., -1:]], dim=-1)  # last col bare

    # -- EPR itself: same formula as compute_epr, with p_fwd = p and p_rev = G
    S_rev = anchor_reverse_synaptic_term(buff_I[..., slots], alpha)          # (..., K)
    hr = beta * (S_rev + R - theta)
    Lam_f, g_f = torch.nn.functional.softplus(hf), hf
    Lam_r, g_r = torch.nn.functional.softplus(hr), hr

    H_fwd = (p * Lam_f).sum(dim=-1) - (col0 * g_f).sum(dim=-1)
    H_rev  = (state.G * Lam_r).sum(dim=-1) - (row0 * g_r).sum(dim=-1)
    epr = torch.stack([H_rev - H_fwd, H_fwd, H_rev], dim=-1)              # (..., 3)

    # -- advance the marginal, reusing the hazard row above
    p_next = update_age_distribution(p, phi_t)

    # -- slide logV: drop time t off the front of the window, add time t+K-2 at the back.
    # Index Q-1 takes the lumped-bin value rather than the shifted one -- the two agree
    # analytically (both merged cohorts see the same clamped future) and this is the one
    # that stays exact if the shift ever falls off the end.
    L_t = torch.log1p(-phi_t.clamp(max=hazard_clamp(phi_t.dtype)))
    L_e = torch.log1p(-state.phi_end.clamp(max=hazard_clamp(state.phi_end.dtype)))
    W = state.logV - L_t + L_e
    logV_next = torch.empty_like(W)
    logV_next[..., 1:]  = W[..., :-1]
    logV_next[..., -1]  = W[..., -1]

    # -- the newborn cohort's own window, and the next step's row 0, share one diagonal
    d2 = compute_firing_prob(buff_S[..., slots[1:K - 1], idx.a_diag[:K - 2]],
                             R[..., idx.a_diag[:K - 2]], beta, theta)   # (..., K-2)
    log_surv2 = torch.cumsum(torch.log1p(-d2.clamp(max=hazard_clamp(d2.dtype))), dim=-1)
    logV_next[..., 0] = log_surv2[..., -1]
    surv2 = torch.cat([torch.ones_like(d2[..., :1]), torch.exp(log_surv2[..., :-1])], dim=-1)
    row0_next = p_next[..., :1] * surv2 * d2                               # (..., K-2)

    # -- slide G: shift-and-add on the interior, V-weighted sums on the two boundary lags
    phi_end_next = compute_firing_prob(buff_S[..., slots[K - 1], idx.a_end],
                                       R[..., idx.a_end], beta, theta)  # (..., Q)
    pV = p_next * torch.exp(logV_next)
    G_next = torch.empty_like(state.G)
    G_next[..., :K - 2] = state.G[..., 1:K - 1] + row0_next
    G_next[..., K - 2]  = (pV * phi_end_next).sum(dim=-1)
    G_next[..., K - 1]  = (pV * (1 - phi_end_next)).sum(dim=-1)

    return epr, p_next, EPRState(G_next, logV_next, phi_end_next)



# Auxiliary functions for fixed points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# At a fixed point every overlap is constant, so the input I^a = sum_b w^{ab} m^b + E^a is
# constant too and the whole age-resolved state follows from it in closed form: the
# synaptic term unrolls (compute_stationary_forward_field), the hazard is then a pure
# function of age, and the stationary age distribution is its survival, normalized
# (compute_stationary_age_distribution). Composing the two gives the overlap map F, whose
# roots m = F(m) are the fixed points -- one O(sum_a Q^a) pass per evaluation, no time
# integration. The generic root finders and the stability spectrum live in
# rdme.fixed_points; this section only builds the model's own residual and the lift from a
# root back to a full (p, S) state.

def compute_stationary_forward_field(
        I: torch.Tensor,               # (...) constant total input to the population
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        Q: int,                        # number of age bins
        ) -> torch.Tensor:             # (..., Q) S_n at constant input
    """ The synaptic term at a fixed point, in closed form.

    With I frozen, S_n = alpha*I + (1-alpha)*S_{n-1} and S_0 = 0 unrolls to the geometric
    sum S_n = I*[1 - (1-alpha)^n]: the same saturating profile an isolated neuron has under
    a constant external drive, with the drive replaced by the self-consistent input.

    This is the exact fixed point of update_synaptic_term, boundary bin included -- that
    update shifts rather than lumps, so its last bin is just the recursion at n = Q-1. """
    I = torch.as_tensor(I)
    decay = torch.as_tensor(1.0 - alpha, device=I.device, dtype=I.dtype)
    n = torch.arange(Q, device=I.device, dtype=I.dtype)
    return I.unsqueeze(-1) * (1.0 - decay.unsqueeze(-1) ** n)

def compute_stationary_survival(fprobs: torch.Tensor,   # (..., Q) age-resolved hazard
                     log: bool = False,
                     ) -> torch.Tensor:      # (..., Q) survival from age 0
    """ S_0(n) = prod_{j<n} (1 - Phi_j), with the empty product S_0(0) = 1.

    Accumulated in log space for the same reason the EPR path is: at large beta the hazard
    saturates and a long product of (1-Phi) underflows to zero, which would send the
    normalization of the stationary distribution to a ratio of zeros. The hazard is clamped
    below one first, with a dtype-dependent bound (see hazard_clamp). """
    log1m = torch.log1p(-fprobs.clamp(max=hazard_clamp(fprobs.dtype)))
    cum = torch.cumsum(log1m[..., :-1], dim=-1)
    log_surv = torch.cat([torch.zeros_like(cum[..., :1]), cum], dim=-1)
    return log_surv if log else torch.exp(log_surv)

def compute_stationary_age_distribution(
        fprobs: torch.Tensor,                      # (..., Q) age-resolved hazard
        ) -> tuple[torch.Tensor, torch.Tensor]:    # (..., Q) p*, (...) m*
    """ Stationary age distribution and firing rate at a frozen, age-resolved hazard.

    Iterating the aging branch of update_age_distribution gives p*(n) = p*(0) * S_0(n) on
    the interior bins, while the lumped bin balances its own outflow against the inflow
    from n = Q-2, giving p*(Q-1) = p*(0) * S_0(Q-1) / Phi_{Q-1}: the exact resummation of
    the geometric tail the truncation cuts off, not an approximation of it. Normalization
    then fixes p*(0), which is the firing rate m* itself, since births equal spikes.

    The reset branch is not an extra condition -- sum_n p*(n) Phi_n = m* holds identically
    by telescoping -- so this single pass is the whole stationary solution. In the
    untruncated limit the boundary term drops and m* becomes the renewal firing rate
    1/sum_n S_0(n) of an isolated neuron. """
    surv = compute_stationary_survival(fprobs)
    tail = surv[..., -1] / fprobs[..., -1]                       # lumped bin, geometric tail
    m = 1.0 / (surv[..., :-1].sum(dim=-1) + tail)
    p = torch.cat([surv[..., :-1], tail.unsqueeze(-1)], dim=-1) * m.unsqueeze(-1)
    return p, m

def fp_overlap_map(m: torch.Tensor,        # (M,) or (B, M) trial overlaps
                   w: torch.Tensor,        # (M, M) or (B, M, M)
                   E: torch.Tensor,        # (M,) or (B, M)
                   alpha: torch.Tensor,    # (M,) or (B, M)
                   beta: torch.Tensor,     # (M,) or (B, M)
                   theta: torch.Tensor,    # (M,) or (B, M)
                   R: list[torch.Tensor],  # M tensors, (Qm[i],) or (B, Qm[i])
                   ) -> torch.Tensor:      # (M,) or (B, M) F(m)
    """ One evaluation of the overlap map F: trial overlaps -> stationary firing rates.

    Populations are looped over rather than stacked because Qm is ragged across them; the
    loop is over M (one or two in practice), while every batch axis stays vectorized. """
    I = compute_total_input(m, w, E)
    rates = []
    for i, R_i in enumerate(R):
        S_i = compute_stationary_forward_field(I[..., i], alpha[..., i], R_i.shape[-1])
        phi_i = compute_firing_prob(S_i, R_i, beta[..., i, None], theta[..., i, None])
        rates.append(compute_stationary_age_distribution(phi_i)[1])
    return torch.stack(rates, dim=-1)

def fp_guess_lattice(M: int,
                     n_per_axis: int = 5,
                     lo: float = 0.0,
                     hi: float = 1.0,
                     device: str | torch.device = 'cpu',
                     ) -> torch.Tensor:    # (n_per_axis**M, M) starting points
    """ A regular lattice of Newton starting points covering the overlap box [lo, hi]^M.

    The endpoints are shrunk inwards by half a cell: m = 0 and m = 1 are the edges of the
    physical range, where the hazard is most nearly degenerate, and a Newton step from
    exactly there tends to be thrown straight back out by the clamp. n_per_axis**M grows
    fast, but M is one or two for every model in this module. """
    step = (hi - lo) / (2 * n_per_axis)
    axis = torch.linspace(lo + step, hi - step, n_per_axis, device=device)
    return torch.stack(torch.meshgrid(*([axis] * M), indexing='ij'), dim=-1).reshape(-1, M)

def fp_state(m: torch.Tensor,        # (M,) or (B, M) fixed-point overlaps
             w: torch.Tensor,
             E: torch.Tensor,
             alpha: torch.Tensor,
             beta: torch.Tensor,
             theta: torch.Tensor,
             R: list[torch.Tensor],
             ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """ Lift a root of the overlap map back to the full state (p, S) it stands for.

    Returns the same (p, S) pair of per-population lists an RDMNetwork carries, so it can be
    written straight into a model as an initial condition or handed to the state map whose
    Jacobian decides stability. """
    I = compute_total_input(m, w, E)
    p_list, S_list = [], []
    for i, R_i in enumerate(R):
        S_i = compute_stationary_forward_field(I[..., i], alpha[..., i], R_i.shape[-1])
        phi_i = compute_firing_prob(S_i, R_i, beta[..., i, None], theta[..., i, None])
        p_list.append(compute_stationary_age_distribution(phi_i)[0])
        S_list.append(S_i)
    return p_list, S_list


# Class for single systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RDMNetwork:

    """ Class representing the RDM network. """

    M: int              # number of populations
    deltaT: float       # time step size, in ms
    Qm: list[int]       # (M,) number of age bins for each population
    N: list[int]        # (M,) number of neurons in each population
    w: torch.Tensor     # (M, M) synaptic weights

    p: list[torch.Tensor]     # age distribution, one (Qm[i],) tensor per population
    S: list[torch.Tensor]     # synaptic term, one (Qm[i],) tensor per population
    R: list[torch.Tensor]   # refractory kernel, one (Qm[i],) tensor per population

    E: torch.Tensor           # (M,) external input current for each population
    beta: torch.Tensor        # (M,) inverse temperature
    theta: torch.Tensor       # (M,) firing threshold
    tau_int: torch.Tensor     # (M,) integration kernel time constant (ms)
    tau_ref: torch.Tensor     # (M,) refractory kernel time constant (ms)
    K_ref: torch.Tensor       # (M,) refractory kernel strength
    alpha_int: torch.Tensor   # (M,) integration kernel scaling factor

    device: str         # device for tensors

    # precomputed quantities
    N_ratios: torch.Tensor      # (M,) population size ratios

    # Construction and initialization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, M: int,
                 w: torch.Tensor,
                 N: list[int],
                 E: torch.Tensor | list[float],
                 beta: torch.Tensor | list[float],
                 theta: torch.Tensor | list[float],
                 tau_int: torch.Tensor | list[float],
                 tau_ref: torch.Tensor | list[float],
                 K_ref: torch.Tensor | list[float],
                 deltaT: float = 1.0,
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 device: str = 'cpu') -> None:
        """ Initialize the model with given parameters and default initial conditions. """

        assert w.shape == (M, M), f"w must be ({M},{M})"
        for name, seq in [("N", N), ("E", E), ("beta", beta), ("theta", theta),
                          ("tau_int", tau_int), ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            assert len(seq) == M, f"{name} must have length M={M}"
        assert all(isinstance(n, int) for n in N), "N must be a list of integers."

        # Pin every tensor in this model to the dtype active at construction time
        # (respects torch.set_default_dtype), instead of hardcoding float32 for
        # some tensors while others silently follow the ambient default -- that
        # split made w/E/beta/... float32 and p/S/m float64 under a float64
        # default, and w @ m in total_input() doesn't auto-promote across
        # dtypes like elementwise ops do, so it crashed.
        dtype = torch.get_default_dtype()

        self.M, self.deltaT, self.device = M, deltaT, device
        self.N = N
        self.w = w.to(device=device, dtype=dtype)

        self.E       = torch.as_tensor(E,       dtype=dtype, device=device)
        self.beta    = torch.as_tensor(beta,    dtype=dtype, device=device)
        self.theta   = torch.as_tensor(theta,   dtype=dtype, device=device)
        self.tau_int = torch.as_tensor(tau_int, dtype=dtype, device=device)
        self.tau_ref = torch.as_tensor(tau_ref, dtype=dtype, device=device)
        self.K_ref   = torch.as_tensor(K_ref,   dtype=dtype, device=device)

        self.N_ratios = torch.as_tensor(N, dtype=dtype, device=device)
        self.N_ratios = self.N_ratios / self.N_ratios.sum()

        self.alpha_int = shrd.tau2alpha(self.tau_int, deltaT)

        if Qm is None:
            Qm = [shrd.q_from_tau(max(self.tau_int[i].item(), self.tau_ref[i].item()), deltaT, eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        # per-population refractory kernel, precomputed once
        self.R = [shrd.refractory_kernel(Qm[i], self.K_ref[i].item(), self.tau_ref[i].item(), dt=deltaT, device=device)
                    for i in range(M)]

        # initial conditions: all age-mass in the oldest bin, zero integrated field
        self.p = [torch.zeros(Qm[i], device=device) for i in range(M)]
        for pi in self.p: pi[-1] = 1.0
        self.S = [torch.zeros(Qm[i], device=device) for i in range(M)]

        self.m = torch.zeros(M, device=device)  # m[i] == p[i][0], starts at 0

    # Observables and derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Returns the current per-population activity vector m of shape (M,). """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ Returns the N-weighted mean network activity (scalar). """
        return (self.N_ratios * self.m).sum()

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ Computes the cross-population input current, shape (M,). """
        return compute_total_input(self.m, self.w, self.E)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ Returns the total local field S_i + R_i per population, list of (Qm[i],) tensors. """
        return [self.S[i] + self.R[i] for i in range(self.M)]

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Computes firing probabilities Phi(beta*(field - theta)) per population, list of (Qm[i],) tensors. """
        return [compute_firing_prob(self.S[i], self.R[i], self.beta[i], self.theta[i])
                for i in range(self.M)]

    # Fixed points
    # ~~~~~~~~~~~~~

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m), the residual whose roots are the fixed points. Takes and
        returns an (M,) tensor -- or a batch of them, with m of shape (..., M). """
        return m - fp_overlap_map(m, self.w, self.E, self.alpha_int,
                                  self.beta, self.theta, self.R)

    def fp_state(self, m: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ The (p, S) state a given overlap vector implies at stationarity. """
        return fp_state(m, self.w, self.E, self.alpha_int, self.beta, self.theta, self.R)

    def set_state(self, m: torch.Tensor) -> None:
        """ Place the model on the stationary state implied by m, so that a subsequent
        forward() starts exactly there. The natural way to verify a root: integrate from it
        and check nothing moves. """
        p, S = self.fp_state(m)
        self.p = [pi.detach().clone() for pi in p]
        self.S = [Si.detach().clone() for Si in S]
        self.m = torch.stack([pi[0] for pi in self.p])

    def fixed_points(self,
                     method: str = 'auto',
                     guesses: torch.Tensor | None = None,
                     n_grid: int = 201,
                     tol: float = 1e-10,
                     merge_tol: float = 1e-6,
                     **kwargs) -> torch.Tensor:     # (R, M) roots, one row per fixed point
        """ All fixed points of the mean field, as overlap vectors.

        method='bracket' scans [0,1] for sign changes of G and bisects them; it is
        exhaustive up to the grid resolution and is the default for M=1, where it reliably
        picks up the unstable middle branch inside the bistable region. method='newton'
        runs Newton from a lattice of starting points and is the default for M>1, where no
        scan is affordable; it finds whatever the basins reach, so a root missing from its
        output is not proof that none exists. 'auto' picks by M.

        Roots are returned unclassified -- pass each to fp_eigenvalues (or is_stable) to
        sort the stable branches from the unstable ones. """
        if method == 'auto':
            method = 'bracket' if self.M == 1 else 'newton'

        if method == 'bracket':
            assert self.M == 1, "bracketing is a scalar method; use method='newton' for M > 1"
            return fpts.bracket_roots(self.fp_residual, n_grid=n_grid,
                                      merge_tol=merge_tol, device=self.device, **kwargs)
        if method == 'newton':
            if guesses is None:
                guesses = fp_guess_lattice(self.M, n_per_axis=5, device=self.device)
            return fpts.newton_roots(self.fp_residual, guesses, tol=tol,
                                     merge_tol=merge_tol, **kwargs)
        raise ValueError(f"unknown method {method!r}, expected 'auto', 'bracket' or 'newton'")

    # Stability
    # ~~~~~~~~~

    def flatten_state(self, p: list[torch.Tensor], S: list[torch.Tensor]) -> torch.Tensor:
        """ Stack the per-population (p, S) lists into one flat state vector. """
        return torch.cat([t for i in range(self.M) for t in (p[i], S[i])], dim=-1)

    def unflatten_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ Inverse of flatten_state. """
        p, S, off = [], [], 0
        for Q in self.Qm:
            p.append(x[..., off:off + Q]);      off += Q
            S.append(x[..., off:off + Q]);      off += Q
        return p, S

    def state_map(self, x: torch.Tensor) -> torch.Tensor:
        """ One step of the dynamics on the flat state vector: the autograd-friendly twin
        of update().

        The state is (p, S) per population and nothing else: the overlaps are not
        independent variables, since update() reads its input from the previous step's
        firing rate, which is exactly p[0]. Keeping them out of the state is what makes the
        linearization here the true one -- an extra copy of m would contribute a spurious
        zero mode. Unlike update() this runs out of place and outside inference mode, so
        torch.func can differentiate through it. """
        p, S = self.unflatten_state(x)
        m = torch.stack([p[i][..., 0] for i in range(self.M)], dim=-1)
        I = compute_total_input(m, self.w, self.E)
        fprobs = [compute_firing_prob(S[i], self.R[i], self.beta[i], self.theta[i])
                  for i in range(self.M)]
        p_new = [update_age_distribution(p[i], fprobs[i]) for i in range(self.M)]
        S_new = [update_synaptic_term(S[i], self.alpha_int[i], I[..., i]) for i in range(self.M)]
        return self.flatten_state(p_new, S_new)

    def fp_eigenvalues(self, m: torch.Tensor) -> torch.Tensor:
        """ Spectrum of the linearized dynamics at the fixed point m, as complex
        eigenvalues of the one-step state map. Sum(Qm) of them, one per state coordinate. """
        x = self.flatten_state(*self.fp_state(m))
        return fpts.jacobian_eigvals(self.state_map, x)

    def is_stable(self, m: torch.Tensor, tol: float = 1e-8) -> bool:
        """ Whether the fixed point m is linearly stable, i.e. whether every mode of the
        linearized map decays. `tol` is the margin allowed to the unit circle: a mode within
        it is marginal, and stability is then decided by terms this linearization drops. """
        rho = fpts.spectral_radius(self.fp_eigenvalues(m))
        return bool(rho < 1.0 - tol)

    # Dynamics and trajectories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def update(self) -> None:
        """ Perform a single time-step update. """
        I = self.total_input()         # (M,) — from OLD self.m
        fprobs    = self.firing_prob()              # list of (Qm[i],) — from OLD S/R

        self.m = torch.stack([compute_firing_rate(self.p[i], fprobs[i]) for i in range(self.M)])

        for i in range(self.M):
            update_age_distribution(self.p[i], fprobs[i], inplace=True)
            update_synaptic_term(self.S[i], self.alpha_int[i], I[i], inplace=True)

        for pi in self.p:
            check_age_normalization(pi)

    @torch.inference_mode()
    def forward(self, T: int) -> None:
        for _ in range(T):
            self.update()

    @torch.inference_mode()
    def trajectory(self, T: int, m_tot: bool = False, p: bool = False, S: bool = False) -> dict:
        """ Run for T steps, returning only the requested quantities.
        Always returns "m" (T, M) per-population. Optional keys: m_tot (T,), the
        N-weighted network activity;
        p/S as lists of M tensors, each (T, Qm[i]) — ragged across populations, dense across time. """
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {"m": torch.zeros(T, self.M, device=self.device)}
        if m_tot: out["m_tot"] = torch.zeros(T, device=self.device)
        if p:   out["p"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]
        if S:   out["S"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]

        for t in tqdm.tqdm(range(T)):
            out["m"][t] = self.m
            if m_tot: out["m_tot"][t] = self.activity()
            if p:
                for i in range(self.M): out["p"][i][t] = self.p[i]
            if S:
                for i in range(self.M): out["S"][i][t] = self.S[i]
            self.update()

        def _cpu(v): return v.cpu() if torch.is_tensor(v) else [t.cpu() for t in v]
        return {k: _cpu(v) for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory(self, T: int) -> dict[str, torch.Tensor]:
        """ Online per-population EPR using one O(Qm[i]^2) rolling buffer per population.

        Only the field history needs the (Qm[i], Qm[i]) look-ahead buffer: the joint
        distribution builds each cohort's survival from hazards, so the age distribution is
        read at time t only, and a single (Qm[i],) marginal stepped forward alongside the
        buffers replaces a second square buffer.

        The joint itself is built once, by init_epr_state; from there step_epr slides the
        four quantities the EPR actually reads out of it, at O(Qm[i]) per step instead of
        O(Qm[i]^2) -- see the derivation above the EPRState definition. The buffer is a ring:
        the row retired at time t is the slot the incoming row takes, so a step costs one row

        All populations advance on ONE shared clock (self.update() has no per-population
        granularity), but each has its own Qm[i]. The fill phase therefore runs Q_max=max(Qm)
        shared ticks, writing into population i's (Qm[i], Qm[i]) buffer for only the first
        Qm[i] of them; the (Q_max-Qm[i]) leftover snapshots computed for smaller populations
        during this phase seed a per-population FIFO that the main loop then keeps pushing to
        and popping from, so that population's tail-writes stay a fixed (Q_max-Qm[i]) steps
        behind the shared clock for the whole run and no population ever loses, re-reads or
        skips ahead of a timestep.

        Returns dict of CPU tensors: m (T, M) per-population activity, m_tot (T,) its
        N-weighted network aggregate;
        sigma/H_fwd/H_rev (T, M) per-population entropy-production decomposition
        (sigma = H_rev - H_fwd, per population); sigma_tot/H_fwd_tot/H_rev_tot (T,) the
        same quantities aggregated across populations via the N_ratios weighting already
        used for activity() — i.e. sigma_tot = sum_i N_ratios[i] * sigma[:, i]. """
        M, Qm, device = self.M, self.Qm, self.device
        Q_max = max(Qm)

        out_m     = torch.zeros(T, M, device=device)
        out_m_tot = torch.zeros(T, device=device)
        out_sigma = torch.zeros(T, M, device=device)
        out_H_fwd = torch.zeros(T, M, device=device)
        out_H_rev = torch.zeros(T, M, device=device)

        p_cur   = [self.p[i].clone() for i in range(M)]   # marginal at the reported time t
        buff_S  = [torch.zeros(Qm[i], Qm[i], device=device) for i in range(M)]
        buff_I = [torch.zeros(Qm[i],        device=device) for i in range(M)]
        pending = [deque() for _ in range(M)]  # delay line, depth Q_max-Qm[i], seeded below

        for j in range(Q_max):
            I = self.total_input()
            for i in range(M):
                snap = (self.S[i].clone(), I[i].clone())
                if j < Qm[i]:
                    buff_S[i][j], buff_I[i][j] = snap
                else:
                    pending[i].append(snap)
            self.update()


        idx   = [make_epr_indices(Qm[i], Qm[i], device) for i in range(M)]
        state = [init_epr_state(p_cur[i], buff_S[i], self.R[i], self.beta[i], self.theta[i])
                 for i in range(M)]
        for t in tqdm.tqdm(range(T)):
            m_t = torch.stack([p_cur[i][0] for i in range(M)])
                # m_t must be read BEFORE step_epr advances p_cur, or the activity
                # channel reports m_{t+1} against a sigma[t] that is the t -> t+1
                # transition -- a one-step offset against the spin model, which
                # reports m_t alongside the same transition
            for i in range(M):
                # step_epr advances p_cur[i] too: it needs the Phi(S_t) row for the joint's
                # column 0 anyway, and that is exactly the renewal update self.update() applies
                epr, p_cur[i], state[i] = step_epr(
                    p_cur[i], state[i], buff_S[i], buff_I[i], idx[i], t, self.R[i],
                    self.alpha_int[i], self.beta[i], self.theta[i]
                )
                out_sigma[t, i] = epr[0]
                out_H_fwd[t, i] = epr[1]
                out_H_rev[t, i] = epr[2]

            out_m[t]     = m_t
            out_m_tot[t] = (self.N_ratios * m_t).sum()

            # read the tail from the CURRENT (pre-update) live state, THEN advance —
            # this ordering (vs. update-then-read) is what avoids a timestep skip
            I = self.total_input()
            for i in range(M):
                # push-then-pop: pending is a standing delay line of depth Q_max-Qm[i], so a
                # small population keeps reading the live state as of time t+Qm[i] instead of
                # the shared clock's t+Q_max. Depth 0 (Qm[i] == Q_max) pops what it just pushed.
                pending[i].append((self.S[i].clone(), I[i].clone()))
                S_val, I_val = pending[i].popleft()
                # ring buffer: the slot holding the row just retired (time t) is the one the
                # incoming row (time t+Qm[i]) belongs in, so nothing is copied but that row
                buff_S[i][t % Qm[i]]  = S_val
                buff_I[i][t % Qm[i]] = I_val
            self.update()

        # rewind live state to match the last reported timestep, discarding the
        # extra Q_max-step lookahead accumulated in self.p/self.S during buffering
        for i in range(M):
            self.p[i] = p_cur[i]
            # with head advanced to T, logical row 0 (time T) sits in ring slot T % Qm[i]
            self.S[i] = buff_S[i][T % Qm[i]].clone()
        self.m = torch.stack([self.p[i][0] for i in range(M)])

        return {
            "m":         out_m.cpu(),
            "m_tot":     out_m_tot.cpu(),
            "sigma":     out_sigma.cpu(),
            "H_fwd":     out_H_fwd.cpu(),
            "H_rev":     out_H_rev.cpu(),
            "sigma_tot": (out_sigma * self.N_ratios).sum(dim=1).cpu(),
            "H_fwd_tot": (out_H_fwd * self.N_ratios).sum(dim=1).cpu(),
            "H_rev_tot": (out_H_rev * self.N_ratios).sum(dim=1).cpu(),
        }

# Constructors for common models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def RDMIsingModel(
    J: float,
    E: float,
    beta: float,
    theta: float,
    tau_int: float,
    tau_ref: float,
    K_ref: float,
    dt: float = 1.0,
    eps: float = 0.01,
    device: str = 'cpu',
    ) -> RDMNetwork:
        """ Ising RDMNetwork. """
        w = torch.tensor([[J]])
        return RDMNetwork(
            M=1, w=w, N=[1000], E=[E], beta=[beta], theta=[theta],
            tau_int=[tau_int], tau_ref=[tau_ref], K_ref=[K_ref],
            deltaT=dt, eps=eps, device=device
        )

def RDMWilsonCowan(
    E_ratio: float,
    w_EE: float,
    w_EI: float,
    w_IE: float,
    w_II: float,
    E_exc: float,
    E_inh: float,
    beta_E: float,
    beta_I: float,
    theta_E: float,
    theta_I: float,
    tau_int_E: float,
    tau_int_I: float,
    tau_ref_E: float,
    tau_ref_I: float,
    K_ref_E: float,
    K_ref_I: float,
    dt: float = 1.0,
    eps: float = 0.01,
    device: str = 'cpu',
) -> RDMNetwork:
    """ Wilson-Cowan-*like* two-population (M=2, order [E, I]) RDMNetwork. """
    N_E = int(E_ratio * 1000)
    N_I = 1000 - N_E
    # w_XY is source-first (X -> Y), so the matrix stores it transposed: row = target,
    # column = source, matching w^{ab} ("from b onto a") in compute_total_input
    w = torch.tensor([[w_EE, w_IE],
                      [w_EI, w_II]], dtype=torch.float32)
    return RDMNetwork(
        M=2, w=w, N=[N_E, N_I], E=[E_exc, E_inh], beta=[beta_E, beta_I],
        theta=[theta_E, theta_I], tau_int=[tau_int_E, tau_int_I],
        tau_ref=[tau_ref_E, tau_ref_I], K_ref=[K_ref_E, K_ref_I],
        deltaT=dt, eps=eps, device=device
    )


# Class for batch of systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RDMNetworkBatch:

    """ Batch of N_param independent RDMNetworks sharing the same architecture
    (M populations, N neurons per population, Qm age bins per population) but
    each with its own parameters. """

    B: int                    # batch size (N_param)
    M: int                    # number of populations
    deltaT: float             # time step size, in ms
    Qm: list[int]             # (M,) number of age bins per population, shared across the batch
    N: list[int]              # (M,) number of neurons per population, shared across the batch
    w: torch.Tensor           # (B, M, M) synaptic weights

    p: list[torch.Tensor]     # age distribution, one (B, Qm[i]) tensor per population
    S: list[torch.Tensor]     # synaptic term, one (B, Qm[i]) tensor per population
    R: list[torch.Tensor]   # refractory kernel, one (B, Qm[i]) tensor per population

    E: torch.Tensor           # (B, M) external input current
    beta: torch.Tensor        # (B, M) inverse temperature
    theta: torch.Tensor       # (B, M) firing threshold
    tau_int: torch.Tensor     # (B, M) integration kernel time constant (ms)
    tau_ref: torch.Tensor     # (B, M) refractory kernel time constant (ms)
    K_ref: torch.Tensor       # (B, M) refractory kernel strength
    alpha_int: torch.Tensor   # (B, M) integration kernel scaling factor

    m: torch.Tensor           # (B, M) current per-population overlaps

    device: str                 # device for tensors

    # parameter-grid bookkeeping (see unflatten())
    grid_shape: tuple[int, ...]         # axis sizes whose product is B; (B,) if not a grid
    grid_axes: dict[str, torch.Tensor]  # swept 1D input vectors, in grid order

    # precomputed quantities
    N_ratios: torch.Tensor      # (M,) population size ratios, shared across the batch

    # Construction and initialization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, M: int,
                 w: torch.Tensor,
                 N: list[int],
                 E: torch.Tensor,
                 beta: torch.Tensor,
                 theta: torch.Tensor,
                 tau_int: torch.Tensor,
                 tau_ref: torch.Tensor,
                 K_ref: torch.Tensor,
                 deltaT: float = 1.0,
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 grid_shape: tuple[int, ...] | None = None,
                 grid_axes: dict[str, torch.Tensor] | None = None,
                 device: str = 'cpu') -> None:
        """ Initialize a batch of RDMNetworks with given parameters and default initial conditions.

        w has shape (B, M, M); E, beta, theta, tau_int, tau_ref, K_ref have shape (B, M), with
        B=N_param inferred from w. N stays a plain length-M list (population sizes are structural,
        not fitted parameters) and is shared across the whole batch, same as M.

        Qm, if given, is a length-M list shared across the batch. Otherwise each population's bin
        count is auto-sized from the slowest (tau_int, tau_ref) pair *anywhere in the batch* for
        that population, so a single Qm[i] safely represents every system in the batch.

        grid_shape/grid_axes describe a batch that came from a flattened parameter grid (see
        shared.flatten_param_grid, and the RDMIsingModelBatch / RDMWilsonCowanBatch constructors
        that use it): trajectory outputs are then returned with the batch axis expanded back into
        the grid axes, and grid_axes carries the swept 1D vectors for plotting. Omit both for a
        plain flat batch, which is left untouched. """

        assert w.ndim == 3 and w.shape[1:] == (M, M), f"w must be (B,{M},{M})"
        B = w.shape[0]
        for name, arr in [("E", E), ("beta", beta), ("theta", theta),
                          ("tau_int", tau_int), ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            assert arr.shape == (B, M), f"{name} must have shape (B={B},M={M})"
        assert len(N) == M and all(isinstance(n, int) for n in N), "N must be a list of M integers."

        # see RDMNetwork.__init__ for why dtype is pinned to the ambient default here
        dtype = torch.get_default_dtype()

        self.B, self.M, self.deltaT, self.device = B, M, deltaT, device
        self.N = N
        self.w = w.to(device=device, dtype=dtype)

        self.grid_shape = tuple(int(s) for s in grid_shape) if grid_shape is not None else (B,)
        n_grid = 1
        for s in self.grid_shape: n_grid *= s
        assert n_grid == B, f"grid_shape {self.grid_shape} has {n_grid} cells, expected B={B}"
        self.grid_axes = dict(grid_axes) if grid_axes is not None else {}

        self.E       = torch.as_tensor(E,       dtype=dtype, device=device)
        self.beta    = torch.as_tensor(beta,    dtype=dtype, device=device)
        self.theta   = torch.as_tensor(theta,   dtype=dtype, device=device)
        self.tau_int = torch.as_tensor(tau_int, dtype=dtype, device=device)
        self.tau_ref = torch.as_tensor(tau_ref, dtype=dtype, device=device)
        self.K_ref   = torch.as_tensor(K_ref,   dtype=dtype, device=device)

        self.N_ratios = torch.as_tensor(N, dtype=dtype, device=device)
        self.N_ratios = self.N_ratios / self.N_ratios.sum()

        self.alpha_int = shrd.tau2alpha(self.tau_int, deltaT)  # (B, M), elementwise

        if Qm is None:
            Qm = [shrd.q_from_tau(max(self.tau_int[:, i].max().item(), self.tau_ref[:, i].max().item()), deltaT, eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        # per-population, per-batch-element refractory kernel: simple_refractory_kernel takes
        # scalar K/tau_r, so vmap over the batch axis for each population separately (ragged Qm
        # across i rules out a single vmap over both axes at once)
        self.R = [
            torch.vmap(lambda K, tau, i=i: shrd.refractory_kernel(self.Qm[i], K, tau, dt=deltaT, device=device))(
                self.K_ref[:, i], self.tau_ref[:, i])
            for i in range(M)
        ]  # list of (B, Qm[i])

        # initial conditions: all age-mass in the oldest bin, zero integrated field
        self.p = [torch.zeros(B, Qm[i], device=device) for i in range(M)]
        for pi in self.p: pi[:, -1] = 1.0
        self.S = [torch.zeros(B, Qm[i], device=device) for i in range(M)]

        self.m = torch.zeros(B, M, device=device)  # m[:, i] == p[i][:, 0], starts at 0

    def select(self, b: int) -> RDMNetwork:
        """ Extract batch element b as a standalone single-system RDMNetwork.

        Built by hand rather than through RDMNetwork.__init__ so that the refractory
        kernels and bin counts are *copied* rather than recomputed -- a batch loaded from
        disk may carry a Qm that was sized from the slowest system in the batch, and
        re-deriving it per element would silently give this one a different age grid.

        The state (p, S, m) comes along, so the selected model is exactly where the batch
        is, and the single-system methods -- fixed_points, fp_eigenvalues, entropy_trajectory
        -- apply to it unchanged. """
        obj = object.__new__(RDMNetwork)
        obj.M, obj.deltaT, obj.device = self.M, self.deltaT, self.device
        obj.N, obj.Qm = list(self.N), list(self.Qm)
        obj.N_ratios = self.N_ratios
        obj.w = self.w[b]
        for key in ("E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "alpha_int"):
            setattr(obj, key, getattr(self, key)[b])
        obj.R = [R_i[b].clone() for R_i in self.R]
        obj.p = [p_i[b].clone() for p_i in self.p]
        obj.S = [S_i[b].clone() for S_i in self.S]
        obj.m = self.m[b].clone()
        return obj

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Expand a leading flat batch axis (B, ...) into the parameter-grid axes (*grid, ...).

        Pass-through when this batch didn't come from a grid, so a directly constructed
        RDMNetworkBatch keeps the plain (B, ...) output shapes. """
        if len(self.grid_shape) <= 1:
            return x
        return shrd.unflatten_batch(x, self.grid_shape)

    # Serialization
    # ~~~~~~~~~~~~~

    _SAVE_TENSORS = ("w", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref",
                     "alpha_int", "N_ratios", "m")
    _SAVE_LISTS   = ("R", "p", "S")

    def save(self, path: str | Path) -> None:
        """ Save the full batch state, parameters and grid layout to a file. """
        data = {
            "B": self.B, "M": self.M, "deltaT": self.deltaT, "Qm": self.Qm, "N": self.N,
            "grid_shape": self.grid_shape, "grid_axes": self.grid_axes, "device": self.device,
        }
        data.update({k: getattr(self, k) for k in self._SAVE_TENSORS})
        data.update({k: getattr(self, k) for k in self._SAVE_LISTS})
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> RDMNetworkBatch:
        """ Load a batch saved with save(), without rebuilding the parameter grid. """
        data = torch.load(path, map_location=device, weights_only=True)
        obj = object.__new__(cls)
        obj.B, obj.M, obj.deltaT = data["B"], data["M"], data["deltaT"]
        obj.Qm, obj.N = list(data["Qm"]), list(data["N"])
        obj.grid_shape = tuple(data["grid_shape"])
        obj.device = device if device is not None else data["device"]
        obj.grid_axes = {k: v.to(obj.device) for k, v in data["grid_axes"].items()}
        for key in cls._SAVE_TENSORS:
            setattr(obj, key, data[key].to(obj.device))
        for key in cls._SAVE_LISTS:
            setattr(obj, key, [t.to(obj.device) for t in data[key]])
        return obj

    # Observables and derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Returns the current per-population activity vector m, shape (B, M). """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ Returns the N-weighted mean network activity per batch element, shape (B,). """
        return (self.N_ratios * self.m).sum(dim=-1)

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ Computes the cross-population input current, shape (B, M). """
        return compute_total_input(self.m, self.w, self.E)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ Returns the total local field S_i + R_i per population, list of (B, Qm[i]) tensors. """
        return [self.S[i] + self.R[i] for i in range(self.M)]

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Computes firing probabilities Phi(beta*(field - theta)) per population, list of (B, Qm[i]) tensors. """
        return [compute_firing_prob(self.S[i], self.R[i], self.beta[:, i:i+1], self.theta[:, i:i+1])
                for i in range(self.M)]

    # Fixed points
    # ~~~~~~~~~~~~~

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m) for the whole batch at once: m is (B, M), out is (B, M). """
        return m - fp_overlap_map(m, self.w, self.E, self.alpha_int,
                                  self.beta, self.theta, self.R)

    def fp_state(self, m: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ The (p, S) state the batch of overlap vectors m (B, M) implies at stationarity,
        as lists of (B, Qm[i]) tensors -- the same layout self.p and self.S carry. """
        return fp_state(m, self.w, self.E, self.alpha_int, self.beta, self.theta, self.R)

    def set_state(self, m: torch.Tensor) -> None:
        """ Place every system in the batch on the stationary state implied by its row of
        m (B, M), so that a subsequent forward() starts there. """
        p, S = self.fp_state(m)
        self.p = [pi.detach().clone() for pi in p]
        self.S = [Si.detach().clone() for Si in S]
        self.m = torch.stack([pi[:, 0] for pi in self.p], dim=-1)

    def _fp_residual_one(self, m: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """ G(m) for a *single* system, with that system's parameters passed in rather than
        read off self: the form torch.vmap needs to give every Newton trajectory its own
        parameters. Argument order matches fp_sweep's params tuple. """
        w, E, alpha, beta, theta = params[:5]
        R = list(params[5:])
        return m - fp_overlap_map(m, w, E, alpha, beta, theta, R)

    def fp_sweep(self,
                 guesses: torch.Tensor | None = None,
                 n_per_axis: int = 5,
                 tol: float = 1e-10,
                 res_tol: float | None = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ Newton fixed-point search across the whole parameter batch in one call.

        Every (system, starting point) pair is one Newton trajectory and they all run
        simultaneously, so a phase map costs a single vmapped solve rather than B separate
        ones. Returns (roots, converged), shaped (B, K, M) and (B, K) with K the number of
        starting points; `converged` flags the trajectories whose residual actually reached
        res_tol, and the rows it marks False hold junk.

        Deliberately *not* deduplicated: collapsing K guesses into a variable number of
        distinct roots is a ragged, per-system operation that would force a Python loop over
        a batch that is often tens of thousands of cells. Counting distinct roots is what a
        bistability phase map wants anyway, and it is a cheap reduction over the converged
        entries; `fixed_points.deduplicate` is there for the individual systems worth
        inspecting closely. """
        if guesses is None:
            guesses = fp_guess_lattice(self.M, n_per_axis, device=self.device)
        K = guesses.shape[0]

        # one row per (system, guess): the parameters repeat_interleave, the guesses tile
        def rep(t): return t.repeat_interleave(K, dim=0)
        params = (rep(self.w), rep(self.E), rep(self.alpha_int), rep(self.beta),
                  rep(self.theta), *[rep(R_i) for R_i in self.R])
        m0 = guesses.repeat(self.B, 1)                               # (B*K, M)

        m = fpts.newton_iterate(self._fp_residual_one, m0, params, tol=tol, **kwargs)
        res = fpts.residual_norms(self._fp_residual_one, m, params)
        converged = torch.isfinite(res) & (res < (res_tol if res_tol is not None else 1e3 * tol))
        return m.reshape(self.B, K, self.M), converged.reshape(self.B, K)

    # Dynamics and trajectories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def update(self) -> None:
        """ Perform a single time-step update across the whole batch. """
        I = self.total_input()   # (B, M) — from OLD self.m
        fprobs    = self.firing_prob()         # list of (B, Qm[i]) — from OLD S/R

        self.m = torch.stack(
            [compute_firing_rate(self.p[i], fprobs[i]) for i in range(self.M)], dim=1)

        for i in range(self.M):
            self.p[i] = update_age_distribution(self.p[i], fprobs[i])
            self.S[i] = update_synaptic_term(self.S[i], self.alpha_int[:, i:i+1], I[:, i:i+1])

    @torch.inference_mode()
    def forward(self, T: int, pb: bool = True) -> None:
        for _ in tqdm.tqdm(range(T), disable=not pb):
            self.update()

    def _out_device_kwargs(self) -> dict:
        """ Allocation kwargs for trajectory-output accumulators. These grow linearly with T,
        so on long trajectories they can rival or exceed the O(B*Qm^2) per-step compute buffers
        (which stay fixed-size) — pinning them in CPU memory up front and streaming each step's
        result in via non_blocking copies keeps that growth off the GPU entirely, instead of
        accumulating on-device and paying for one big transfer at the end. No-op (plain
        device-resident allocation) when the compute device isn't CUDA. """
        if torch.device(self.device).type == 'cuda':
            return dict(device='cpu', pin_memory=True, dtype=torch.float64)
        return dict(device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def trajectory(self, T: int, 
                m_tot: bool = False, 
                p: bool = False, 
                S: bool = False, 
                pb: bool = True
                ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """ Run for T steps, returning only the requested quantities.
        Always returns "m" (*grid, T, M) per-population. Optional keys: m_tot (*grid, T),
        the N-weighted network activity;
        p/S as lists of M tensors, each (*grid, T, Qm[i]) — ragged across populations, dense
        across time. The leading grid axes are (B,) unless this batch was built from a
        parameter grid (see unflatten()). """
        out_kwargs = self._out_device_kwargs()
        out = {}
        out["m"] = torch.zeros(self.B, T, self.M, **out_kwargs)
        if m_tot: out["m_tot"] = torch.zeros(self.B, T, **out_kwargs)
        if p:   out["p"] = [torch.zeros(self.B, T, self.Qm[i], **out_kwargs) for i in range(self.M)]
        if S:   out["S"] = [torch.zeros(self.B, T, self.Qm[i], **out_kwargs) for i in range(self.M)]

        for t in tqdm.tqdm(range(T), disable=not pb):
            out["m"][:, t].copy_(self.m, non_blocking=True)
            if m_tot: out["m_tot"][:, t].copy_(self.activity(), non_blocking=True)
            if p:
                for i in range(self.M): out["p"][i][:, t].copy_(self.p[i], non_blocking=True)
            if S:
                for i in range(self.M): out["S"][i][:, t].copy_(self.S[i], non_blocking=True)
            self.update()

        def _uf(v): return self.unflatten(v) if torch.is_tensor(v) else [self.unflatten(t) for t in v]
        return {k: _uf(v) for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory(self, T: int, pb: bool = True,
                           chunk: int | None = None) -> dict[str, torch.Tensor]:
        """ Batched online per-population EPR using one O(B*Qm[i]^2) rolling buffer per population.

        Only the field history needs the (B, Qm[i], Qm[i]) look-ahead buffer; the age
        distribution is read at time t only, so a single (B, Qm[i]) marginal stepped forward
        alongside the buffers replaces a second square buffer.

        The per-step work is O(B*Qm[i]), not O(B*Qm[i]^2): the joint is materialised once by
        init_epr_state and from then on step_epr slides the four quantities the EPR reads out
        of it (see the derivation above the EPRState definition). Nothing in the loop is
        per-sample any more, so there is no vmap and no (chunk, Q, K) working set, and the

        Batched analogue of RDMNetwork.entropy_trajectory: all populations advance on ONE shared
        clock, but each keeps its own Qm[i]. The fill phase runs Q_max=max(Qm) shared ticks,
        writing into population i's (B, Qm[i], Qm[i]) buffer for only the first Qm[i] of them;
        leftover snapshots for smaller populations seed a per-population FIFO that the main
        loop keeps pushing to and popping from, holding that population's tail-writes a fixed
        (Q_max-Qm[i]) steps behind the shared clock, so no batch element or population ever
        loses, re-reads or skips ahead of a timestep.

        Returns dict of CPU tensors: m (*grid, T, M) per-population activity, m_tot
        (*grid, T) its N-weighted network aggregate;
        sigma/H_fwd/H_rev (*grid, T, M) per-population entropy-production decomposition
        (sigma = H_rev - H_fwd, per population); sigma_tot/H_fwd_tot/H_rev_tot (*grid, T) the
        same quantities aggregated across populations via the N_ratios weighting already
        used for activity(). The leading grid axes are (B,) unless this batch was built from
        a parameter grid (see unflatten()).

        chunk caps how many batch elements go through the per-step EPR at once. None (the
        default) does the whole batch in one shot; a smaller value trades speed for peak
        memory, which is dominated by the (chunk, Qm[i], Qm[i]) working set rather than by
        the buffers themselves. Results are unaffected up to float32 rounding. """

        B, M, Qm, device = self.B, self.M, self.Qm, self.device
        Q_max = max(Qm)

        out_kwargs = self._out_device_kwargs()
        out_m     = torch.zeros(B, T, M, **out_kwargs)
        out_m_tot = torch.zeros(B, T, **out_kwargs)
        out_sigma = torch.zeros(B, T, M, **out_kwargs)
        out_H_fwd = torch.zeros(B, T, M, **out_kwargs)
        out_H_rev = torch.zeros(B, T, M, **out_kwargs)

        p_cur   = [self.p[i].clone() for i in range(M)]   # marginal at the reported time t
        buff_S  = [torch.zeros(B, Qm[i], Qm[i], device=device) for i in range(M)]
        buff_I = [torch.zeros(B, Qm[i],        device=device) for i in range(M)]
        pending = [deque() for _ in range(M)]  # delay line, depth Q_max-Qm[i], seeded below

        for j in range(Q_max):
            I = self.total_input()   # (B, M)
            for i in range(M):
                snap = (self.S[i].clone(), I[:, i].clone())
                if j < Qm[i]:
                    buff_S[i][:, j], buff_I[i][:, j] = snap
                else:
                    pending[i].append(snap)
            self.update()

        idx   = [make_epr_indices(Qm[i], Qm[i], device) for i in range(M)]
        state = [init_epr_state(p_cur[i], buff_S[i], self.R[i],
                    self.beta[:, i], self.theta[:, i], chunk=chunk) for i in range(M)]
        
        for t in tqdm.tqdm(range(T), disable=not pb):
            m_t = torch.stack([p_cur[i][:, 0] for i in range(M)], dim=1)   # (B, M)
                # m_t must be read BEFORE step_epr advances p_cur, or the activity
                # channel reports m_{t+1} against a sigma[t] that is the t -> t+1
                # transition -- a one-step offset against the spin model, which
                # reports m_t alongside the same transition
            for i in range(M):
                # step_epr advances p_cur[i] too: it needs the Phi(S_t) row for the joint's
                # column 0 anyway, and that is exactly the renewal update self.update() applies
                epr, p_cur[i], state[i] = step_epr(
                    p_cur[i], state[i], buff_S[i], buff_I[i], idx[i], t, self.R[i],
                    self.alpha_int[:, i:i+1], self.beta[:, i:i+1], self.theta[:, i:i+1]
                )
                out_sigma[:, t, i].copy_(epr[:, 0], non_blocking=True)
                out_H_fwd[:, t, i].copy_(epr[:, 1], non_blocking=True)
                out_H_rev[:, t, i].copy_(epr[:, 2], non_blocking=True)

            out_m[:, t].copy_(m_t, non_blocking=True)
            out_m_tot[:, t].copy_((self.N_ratios * m_t).sum(dim=-1), non_blocking=True)

            # read the tail from the CURRENT (pre-update) live state, THEN advance —
            # this ordering (vs. update-then-read) is what avoids a timestep skip
            I = self.total_input()
            for i in range(M):
                # push-then-pop: see RDMNetwork.entropy_trajectory -- pending is a standing
                # delay line of depth Q_max-Qm[i], not a one-shot drain
                pending[i].append((self.S[i].clone(), I[:, i].clone()))
                S_val, I_val = pending[i].popleft()
                # ring buffer: see RDMNetwork.entropy_trajectory
                buff_S[i][:, t % Qm[i]]  = S_val
                buff_I[i][:, t % Qm[i]] = I_val
            self.update()

        # rewind live state to match the last reported timestep, discarding the
        # extra Q_max-step lookahead accumulated in self.p/self.S during buffering
        for i in range(M):
            self.p[i] = p_cur[i]
            self.S[i] = buff_S[i][:, T % Qm[i]].clone()
        self.m = torch.stack([self.p[i][:, 0] for i in range(M)], dim=1)

        # out_* accumulators may already be pinned CPU tensors (see _out_device_kwargs);
        # match N_ratios to whichever device they ended up on for this final reduction
        N_ratios = self.N_ratios.to(out_sigma.device)
        out = {
            "m":         out_m,
            "m_tot":     out_m_tot,
            "sigma":     out_sigma,
            "H_fwd":     out_H_fwd,
            "H_rev":     out_H_rev,
            "sigma_tot": (out_sigma * N_ratios).sum(dim=-1),
            "H_fwd_tot": (out_H_fwd * N_ratios).sum(dim=-1),
            "H_rev_tot": (out_H_rev * N_ratios).sum(dim=-1),
        }
        return {k: self.unflatten(v) for k, v in out.items()}

# Constructors for common models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def RDMIsingModelBatch(
    J: float | torch.Tensor,
    E: float | torch.Tensor,
    beta: float | torch.Tensor,
    theta: float | torch.Tensor,
    tau_int: float | torch.Tensor,
    tau_ref: float | torch.Tensor,
    K_ref: float | torch.Tensor,
    dt: float = 1.0,
    Qm: list[int] | None = None,
    eps: float = 0.01,
    device: str = 'cpu',
    ) -> RDMNetworkBatch:
        """ Batched Ising RDMNetwork (M=1). Batched analogue of RDMIsingModel: each of
        J, E, beta, theta, tau_int, tau_ref, K_ref may be a scalar or a 1D tensor, and the
        batch is the **outer product** of them all — pass two vectors and you get the full
        2D sweep, no meshgrid needed at the call site. This mirrors IsingModelBatch in the
        sibling spike-response-hopfield-model repo.

        Parameters passed as scalars contribute a singleton grid axis that is dropped again
        on output, so trajectory results come back shaped by the swept axes only, in the
        argument order above (e.g. sweeping J and E gives (n_J, n_E, T)). The swept vectors
        are kept in the returned model's grid_axes for plotting. """
        flat, grid_shape, axes = shrd.flatten_param_grid({
            "J": J, "E": E, "beta": beta, "theta": theta,
            "tau_int": tau_int, "tau_ref": tau_ref, "K_ref": K_ref,
        }, device=device)
        B = flat["J"].numel()

        return RDMNetworkBatch(
            M=1, w=flat["J"].reshape(B, 1, 1), N=[1000],
            E=flat["E"].reshape(B, 1), beta=flat["beta"].reshape(B, 1),
            theta=flat["theta"].reshape(B, 1), tau_int=flat["tau_int"].reshape(B, 1),
            tau_ref=flat["tau_ref"].reshape(B, 1), K_ref=flat["K_ref"].reshape(B, 1),
            deltaT=dt, Qm=Qm, eps=eps,
            grid_shape=grid_shape, grid_axes=axes, device=device,
        )


def RDMWilsonCowanBatch(
    E_ratio: float,
    w_EE: float | torch.Tensor,
    w_EI: float | torch.Tensor,
    w_IE: float | torch.Tensor,
    w_II: float | torch.Tensor,
    E_exc: float | torch.Tensor,
    E_inh: float | torch.Tensor,
    beta_E: float | torch.Tensor,
    beta_I: float | torch.Tensor,
    theta_E: float | torch.Tensor,
    theta_I: float | torch.Tensor,
    tau_int_E: float | torch.Tensor,
    tau_int_I: float | torch.Tensor,
    tau_ref_E: float | torch.Tensor,
    tau_ref_I: float | torch.Tensor,
    K_ref_E: float | torch.Tensor,
    K_ref_I: float | torch.Tensor,
    dt: float = 1.0,
    Qm: list[int] | None = None,
    eps: float = 0.01,
    device: str = 'cpu',
    ) -> RDMNetworkBatch:
        """ Batched Wilson-Cowan-*like* two-population (M=2, order [E, I]) RDMNetwork.
        Batched analogue of RDMWilsonCowan: each parameter may be a scalar or a 1D tensor,
        and the batch is the **outer product** of them all — pass w_EE and w_II as vectors
        and you get the full 2D sweep, no meshgrid needed at the call site, mirroring
        RDMIsingModelBatch.

        Parameters passed as scalars contribute a singleton grid axis that is dropped again
        on output, so trajectory results come back shaped by the swept axes only, in the
        argument order above. The swept vectors are kept in the returned model's grid_axes
        for plotting. E_ratio is structural (it sets the population sizes N) and stays a
        scalar — it is not a grid axis. """
        flat, grid_shape, axes = shrd.flatten_param_grid({
            "w_EE": w_EE, "w_EI": w_EI, "w_IE": w_IE, "w_II": w_II,
            "E_exc": E_exc, "E_inh": E_inh, "beta_E": beta_E, "beta_I": beta_I,
            "theta_E": theta_E, "theta_I": theta_I,
            "tau_int_E": tau_int_E, "tau_int_I": tau_int_I,
            "tau_ref_E": tau_ref_E, "tau_ref_I": tau_ref_I,
            "K_ref_E": K_ref_E, "K_ref_I": K_ref_I,
        }, device=device)

        N_E = int(E_ratio * 1000)
        N_I = 1000 - N_E
        # transposed for the same reason as RDMWilsonCowan: row = target, column = source
        w = torch.stack([
            torch.stack([flat["w_EE"], flat["w_IE"]], dim=-1),
            torch.stack([flat["w_EI"], flat["w_II"]], dim=-1),
        ], dim=1)  # (B, 2, 2)

        def stack2(name_E: str, name_I: str) -> torch.Tensor:
            return torch.stack([flat[name_E], flat[name_I]], dim=-1)  # (B, 2)

        return RDMNetworkBatch(
            M=2, w=w, N=[N_E, N_I],
            E=stack2("E_exc", "E_inh"), beta=stack2("beta_E", "beta_I"),
            theta=stack2("theta_E", "theta_I"), tau_int=stack2("tau_int_E", "tau_int_I"),
            tau_ref=stack2("tau_ref_E", "tau_ref_I"), K_ref=stack2("K_ref_E", "K_ref_I"),
            deltaT=dt, Qm=Qm, eps=eps,
            grid_shape=grid_shape, grid_axes=axes, device=device,
        )
