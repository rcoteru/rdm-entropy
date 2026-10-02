from __future__ import annotations

from typing import NamedTuple

import torch

from rdme.hazards import Hazard
from rdme.dynamics import update_age_distribution, hazard_clamp

""" Entropy production, two ways.

The joint path builds the explicit (Q, K) distribution of forward and reverse age and reads the
EPR off it, at O(Q^2) per step. The sliding path reaches the same four marginals by recursion,
at O(Q). Both call epr_from_parts, so they share one formula and can only disagree through
those four quantities -- which is what the cross-check in the tests isolates.

The EPR is a functional of the hazard alone, so one implementation serves all three kernels;
that is why it is written as the plain Bernoulli entropy in Phi rather than in the
softplus/logit form of rdme.mean_field, which is an identity of the logistic kernel only.

Everything here indexes on trailing axes, so one implementation also serves one system and a
whole batch: shapes are written (..., Q) and (..., K), with the leading axes carrying the batch
or nothing at all. rdme.mean_field has to vmap its joint build for the batched case; the gather
here consumes exactly the last two axes, so the ellipsis handles it and no vmap is needed. """


# Time-reversed field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The backward synaptic term the entropy production pairs against the forward one. Kernel-
# independent: it filters the future input and never evaluates a hazard, so it is shared by
# every member of the family.

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


# Entropy production
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def joint_distribution(p: torch.Tensor,        # (Q,)   p_t(n)
                       buff_S: torch.Tensor,   # (K, Q) buff_S[j, n] = S_{t+j}(n), j=0 is t
                       R: torch.Tensor,        # (Q,)
                       beta: float | torch.Tensor,
                       theta: float | torch.Tensor,
                       hazard: Hazard,
                       lam0: float,
                       dt: float,
                       ) -> torch.Tensor:      # (Q, K) p_t(n, n_dagger=k)
    """ Joint distribution of forward and reverse age, built from hazards.

    The hazard-parametrised port of mean_field.compute_joint_distribution; the structure is
    unchanged. A cohort's survival prod_{j<k} (1 - Phi_{t+j}(n+j)) is built by multiplying
    hazards rather than reading density ratios, which is what lets it track a cohort through
    the lumped final bin and means only the marginal p_t is ever needed. Rows sum to 1 by
    telescoping, so no normalization. """
    Q, K = buff_S.shape[-1], buff_S.shape[-2]
    # buff_S carries an extra lag axis that R, beta and theta do not: they are (..., Q) and
    # (..., 1) as everywhere else in the package, so each is lifted one axis here rather than
    # making every caller remember a different convention for these two functions. A 0-dim
    # scalar becomes (1,), which broadcasts just as well.
    lift = lambda x: x.unsqueeze(-1) if torch.is_tensor(x) else x
    R, beta, theta = R.unsqueeze(-2), lift(beta), lift(theta)
    n_idx = torch.arange(Q, device=buff_S.device)
    j_idx = torch.arange(K, device=buff_S.device)
    future_age  = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)   # (Q, K)
    future_time = j_idx.unsqueeze(0)                                                # (1, K)

    phi_all = hazard.phi(buff_S + R - theta, beta, lam0, dt)        # (..., K, Q)
    # the two index tensors consume the last two axes, so the ellipsis carries any batch
    phi = phi_all[..., future_time, future_age]                     # (..., Q, K)

    log_surv = torch.cumsum(torch.log1p(-phi.clamp(max=hazard_clamp(phi.dtype))), dim=-1)
    surv = torch.ones_like(phi)
    surv[..., 1:] = torch.exp(log_surv[..., :-1])
    surv[..., :-1] = surv[..., :-1] * phi[..., :-1]  # first passage; last column keeps bare S
    return surv * p.unsqueeze(-1)


def epr_from_parts(p_fwd: torch.Tensor,    # (..., Q) row sums of the joint == p_t
                   p_rev: torch.Tensor,    # (..., K) column sums   == p_t(n_dagger)
                   spike_f: torch.Tensor,  # (..., Q) J[:, 0], the forward spike weight
                   spike_r: torch.Tensor,  # (..., K) J[0, :], the reverse spike weight
                   phi_fwd: torch.Tensor,  # (..., Q) forward hazard at time t, by age
                   phi_rev: torch.Tensor,  # (..., K) reverse hazard at time t+1, by reverse age
                   ) -> torch.Tensor:      # (..., 3) [sigma, H_fwd, H_rev]
    """ sigma = H_rev - H_fwd, written in terms of the hazard itself.

    The EPR never needs the joint as a matrix -- only these four marginals of it -- which is
    what lets the sliding path compute them by recursion and reuse this function unchanged.
    Both paths therefore share one EPR formula, and a disagreement between them can only come
    from the four inputs.

    mean_field.compute_epr writes the same quantity as H_fwd = sum p softplus(hf) - sum J[:,0] hf,
    using softplus(h) = -log(1 - Phi) and h = log Phi - log(1 - Phi). Both are identities of the
    LOGISTIC kernel only. The generic form is the plain Bernoulli entropy,
        H = -[ sum (spike weight) log Phi + sum (no-spike weight) log(1 - Phi) ],
    which reduces to theirs for the sigmoid and holds for any hazard.

    The spike weights are indicators, not probabilities: the forward transition emits s(t+1),
    which happens iff k == 0; the reverse emits s(t), iff n == 0. Both hazards are clamped away
    from 0 and 1 before the logs -- where Phi is 0 the weight is 0 too, and 0 * -inf is a NaN
    rather than the 0 it should be. """
    lo = lambda phi: torch.log(phi.clamp(min=torch.finfo(phi.dtype).tiny,
                                         max=hazard_clamp(phi.dtype)))
    l1 = lambda phi: torch.log1p(-phi.clamp(max=hazard_clamp(phi.dtype)))

    sm = lambda x: x.sum(dim=-1)
    H_fwd = -(sm(spike_f * lo(phi_fwd)) + sm((p_fwd - spike_f) * l1(phi_fwd)))
    H_rev = -(sm(spike_r * lo(phi_rev)) + sm((p_rev - spike_r) * l1(phi_rev)))
    return torch.stack([H_rev - H_fwd, H_fwd, H_rev], dim=-1)


def epr_from_joint(p_joint: torch.Tensor,   # (Q, K)
                   phi_fwd: torch.Tensor,   # (Q,)
                   phi_rev: torch.Tensor,   # (K,)
                   ) -> torch.Tensor:       # (3,) [sigma, H_fwd, H_rev]
    """ The EPR read off the explicit joint: the reference path, and the oracle the sliding
    recursions are checked against. """
    return epr_from_parts(p_joint.sum(dim=-1), p_joint.sum(dim=-2),
                          p_joint[..., :, 0], p_joint[..., 0, :], phi_fwd, phi_rev)


# Sliding-state EPR: the same quantity in O(Q) per step instead of O(Q^2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Port of the recursions in rdme.mean_field, with the hazard as a parameter. epr_from_parts
# needs only four marginals of the joint, and each slides exactly from one timestep to the
# next, so the (Q, K) joint is built once and never again. Writing Phi[tau, n] for the hazard
# at absolute time tau and age n (buffer row tau, clamped age), L = log1p(-Phi), and J_t for
# the joint at the reported time t:
#
#   1. row sums   J_t.sum(1) == p_t                     exactly (first-passage telescoping)
#   2. column 0   J_t[:, 0]  == p_t * Phi[t, .]         one hazard row
#   3. row 0      J_t[0, :]  is the age-0 cohort's first-passage law, driven only by the
#                 buffer diagonal Phi[t+j, min(j, Q-1)]
#   4. column sums G_t = J_t.sum(0) slide, given the survival window
#                 V_t(n) = prod_{j<K-2} (1 - Phi[t+j, n+j]):
#                   G_{t+1}[j]   = G_t[j+1] + J_{t+1}[0, j]           for j <= K-3
#                   G_{t+1}[K-2] = sum_n p_{t+1}(n) V_{t+1}(n) * phi
#                   G_{t+1}[K-1] = sum_n p_{t+1}(n) V_{t+1}(n) * (1 - phi)
#                 with phi = Phi[t+K-1, min(n+K-2, Q-1)], and V sliding in log space
#                   log V_{t+1}(n+1) = log V_t(n) - L[t, n] + L[t+K-2, min(n+K-2, Q-1)]
#
# The cohort identity behind (4) is S_t(k|n) = (1 - Phi[t,n]) S_{t+1}(k-1|n+1) together with
# p_{t+1}(n+1) = p_t(n)(1 - Phi[t,n]) -- the renewal update itself. It survives the lumped bin
# because the two cohorts merged there share identical clamped future hazards.


class EPRIndices(NamedTuple):
    """ Gather indices for the sliding step, precomputed once per population.

    slots[h] is the buffer-row permutation for ring head h: logical row j (absolute time t+j)
    lives in physical slot slots[h][j]. The ring is what keeps a step O(Q): retiring a row
    overwrites one slot instead of copying the whole (K, Q) buffer, which would be O(Q^2) and
    would cost exactly what the recursions save. """
    slots:  torch.Tensor   # (K, K) all rotations, slots[h, j] = (h + j) % K
    a_diag: torch.Tensor   # (K,)   ages along the age-0 cohort diagonal, min(j, Q-1)
    a_end:  torch.Tensor   # (Q,)   ages at the survival-window edge, min(n + K-2, Q-1)


class EPRState(NamedTuple):
    """ What the sliding step carries between timesteps, valid at the reported time t. """
    G:       torch.Tensor   # (K,) column sums of the joint, G[k] = p_t(n_dagger = k)
    logV:    torch.Tensor   # (Q,) log survival over the next K-2 steps, per age
    phi_end: torch.Tensor   # (Q,) Phi[t+K-2, min(n+K-2, Q-1)], the window-edge hazards


def make_epr_indices(K: int, Q: int, device: str | torch.device) -> EPRIndices:
    if K < 3:
        raise ValueError(f"sliding EPR needs a look-ahead of at least 3 rows, got K={K}")
    if K != Q:
        # the reverse field's h(k) is evaluated against R(k), so the lag axis and the age axis
        # share a length -- a constraint the joint path carries too, and entropy_trajectory
        # always satisfies it (K = Qm[i] = Q)
        raise ValueError(f"sliding EPR needs K == Q, got K={K}, Q={Q}")
    j = torch.arange(K, device=device)
    n = torch.arange(Q, device=device)
    return EPRIndices(slots=(j.unsqueeze(0) + j.unsqueeze(1)) % K,
                      a_diag=torch.clamp(j, max=Q - 1),
                      a_end=torch.clamp(n + (K - 2), max=Q - 1))


def init_epr_state(p, buff_S, R, beta, theta, hazard, lam0, dt) -> EPRState:
    """ The sliding state at the first reported step, built from the full joint.

    The one O(Q*K) step in a trajectory, and deliberately routed through the reference path so
    the recursions start from exactly the state that path would produce. buff_S must already be
    in logical row order (row j = time t+j). """
    Q, K = buff_S.shape[-1], buff_S.shape[-2]
    G = joint_distribution(p, buff_S, R, beta, theta, hazard, lam0, dt).sum(dim=-2)

    # buff_S carries an extra lag axis that R, beta and theta do not: they are (..., Q) and
    # (..., 1) as everywhere else in the package, so each is lifted one axis here rather than
    # making every caller remember a different convention for these two functions. A 0-dim
    # scalar becomes (1,), which broadcasts just as well.
    lift = lambda x: x.unsqueeze(-1) if torch.is_tensor(x) else x
    R, beta, theta = R.unsqueeze(-2), lift(beta), lift(theta)
    n_idx = torch.arange(Q, device=buff_S.device)
    j_idx = torch.arange(K, device=buff_S.device)
    ages = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)      # (Q, K)
    phi_all = hazard.phi(buff_S + R - theta, beta, lam0, dt)                    # (..., K, Q)
    phi = phi_all[..., j_idx.unsqueeze(0), ages]                                # (..., Q, K)

    logV = torch.log1p(-phi[..., :K - 2].clamp(max=hazard_clamp(phi.dtype))).sum(dim=-1)
    return EPRState(G, logV, phi[..., K - 2])


def step_epr(p, state, buff_S, buff_I, idx, head, R, alpha, beta, theta, hazard, lam0, dt):
    """ One timestep of EPR plus the advanced state: (epr (3,), p_{t+1}, state_{t+1}).

    Drop-in for the reference path over a whole trajectory, at O(Q) per step. Three (Q,) hazard
    evaluations and two (K,) gathers replace the (Q, K) joint.

    p_{t+1} is returned rather than stepped by the caller because it shares the Phi[t, .] row
    this function already needs. """
    K = buff_S.shape[-2]
    slots = idx.slots[head % K]
    phi = lambda h: hazard.phi(h, beta, lam0, dt)

    # -- present: the hazard row at time t, shared by H_fwd, column 0 and the logV slide
    S_fwd = buff_S[..., slots[0], :]
    phi_t = phi(S_fwd + R - theta)                                   # (Q,)
    col0 = p * phi_t                                                 # (Q,)  J_t[:, 0]

    # -- row 0: the age-0 cohort's first-passage law, off the buffer diagonal
    d = phi(buff_S[..., slots, idx.a_diag] + R[..., idx.a_diag] - theta)       # (..., K)
    log_surv = torch.cumsum(torch.log1p(-d.clamp(max=hazard_clamp(d.dtype))), dim=-1)
    surv0 = torch.cat([torch.ones_like(d[..., :1]), torch.exp(log_surv[..., :-1])], dim=-1)
    row0 = p[..., :1] * surv0
    row0 = torch.cat([row0[..., :-1] * d[..., :-1], row0[..., -1:]], dim=-1)   # last col bare

    S_rev = anchor_reverse_synaptic_term(buff_I[..., slots], alpha)            # (..., K)
    epr = epr_from_parts(p, state.G, col0, row0, phi_t, phi(S_rev + R - theta))

    p_next = update_age_distribution(p, phi_t)

    # -- slide logV: drop time t off the front of the window, add t+K-2 at the back. Index Q-1
    # takes the lumped-bin value rather than the shifted one; the two agree analytically (both
    # merged cohorts see the same clamped future) and this stays exact if the shift falls off.
    L_t = torch.log1p(-phi_t.clamp(max=hazard_clamp(phi_t.dtype)))
    L_e = torch.log1p(-state.phi_end.clamp(max=hazard_clamp(state.phi_end.dtype)))
    W = state.logV - L_t + L_e
    logV_next = torch.empty_like(W)
    logV_next[..., 1:] = W[..., :-1]
    logV_next[..., -1] = W[..., -1]

    # -- the newborn cohort's own window, and the next step's row 0, share one diagonal
    d2 = phi(buff_S[..., slots[1:K - 1], idx.a_diag[:K - 2]]
             + R[..., idx.a_diag[:K - 2]] - theta)
    log_surv2 = torch.cumsum(torch.log1p(-d2.clamp(max=hazard_clamp(d2.dtype))), dim=-1)
    logV_next[..., 0] = log_surv2[..., -1]
    surv2 = torch.cat([torch.ones_like(d2[..., :1]), torch.exp(log_surv2[..., :-1])], dim=-1)
    row0_next = p_next[..., :1] * surv2 * d2                         # (..., K-2)

    # -- slide G: shift-and-add on the interior, V-weighted sums on the two boundary lags
    phi_end_next = phi(buff_S[..., slots[K - 1], idx.a_end] + R[..., idx.a_end] - theta)
    pV = p_next * torch.exp(logV_next)
    G_next = torch.empty_like(state.G)
    G_next[..., :K - 2] = state.G[..., 1:K - 1] + row0_next
    G_next[..., K - 2]  = (pV * phi_end_next).sum(dim=-1)
    G_next[..., K - 1]  = (pV * (1 - phi_end_next)).sum(dim=-1)

    return epr, p_next, EPRState(G_next, logV_next, phi_end_next)
