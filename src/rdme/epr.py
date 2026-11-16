from __future__ import annotations

from typing import NamedTuple

import math

import torch

from rdme.hazards import Hazard
import rdme.rdm as rdm

""" Entropy production, by the O(Q) observer.

The EPR needs only four marginals of the joint distribution of forward and reverse age, and
each slides exactly from one timestep to the next, so epr_step reaches them by recursion at
O(Q) per step and O(Q) memory, and epr_from_parts reads the EPR off them. The explicit (Q, K)
joint -- the definition, at O(Q^2) per step -- is not in the library: it lives in
tests/epr_reference.py as the oracle the observer is checked against, and calls the same
epr_from_parts, so the two can only disagree through the four marginals.

The EPR is a functional of the hazard alone, so one implementation serves every kernel; that
is why it is written as the plain Bernoulli entropy in Phi rather than in the softplus/logit
form of the pre-port implementation, which is an identity of the logistic kernel only.

Everything here indexes on trailing axes, so one implementation also serves one system and a
whole batch: shapes are written (..., Q) and (..., K), with the leading axes carrying the batch
or nothing at all. """


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
    w = kernel * I[..., 1:]
    del kernel
    w.cumsum_(dim=-1)              # in the product's own buffer: two (..., Q) live, not three
    out = w.new_empty(w.shape[:-1] + (Q,))
    out[..., 0] = 0
    out[..., 1:] = w
    return out


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
#
# Every function below takes the `inplace` flag of rdme.rdm and rdme.hazards, with the
# same contract: the out-of-place branch is the definition and the oracle, the in-place one is
# the same arithmetic in the same order with the temporaries folded away, so the two agree bit
# for bit. Nothing a caller owns is touched -- p, the observer state, the live S and R
# are read-only on both branches, and the state a step returns is freshly allocated -- the flag
# only licenses reuse of the buffers these functions allocate themselves, starting with the
# field h = S + R - theta that every hazard call throws away.


def hazard_field(S: torch.Tensor, R: torch.Tensor, theta, inplace: bool) -> torch.Tensor:
    """ h = S + R - theta in one buffer instead of two. S is never written to: the sub_ lands
    on the sum, which is already a fresh tensor of the broadcast shape. """
    return (S + R).sub_(theta) if inplace else S + R - theta


def _log1m(phi: torch.Tensor, inplace: bool) -> torch.Tensor:
    """ log1p(-Phi) with the clamp every survival product needs, in one buffer instead of
    three. phi itself is never written to -- clamp() already copies, and the rest rides on that
    copy -- so this is safe on a hazard the caller still needs, which is every caller here. """
    c = rdm.hazard_clamp(phi.dtype)
    return phi.clamp(max=c).neg_().log1p_() if inplace else torch.log1p(-phi.clamp(max=c))


def _cumlog1m(phi: torch.Tensor, inplace: bool) -> torch.Tensor:
    """ The running log-survival sum_{j<=k} log1p(-Phi_j), the cumsum folded into _log1m's own
    buffer. Same reassurance as _log1m: phi is read, not written. """
    L = _log1m(phi, inplace)
    return L.cumsum_(-1) if inplace else torch.cumsum(L, dim=-1)


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
    # each chain runs in the buffer its clamp() copies into: the same operations in the same
    # order as the out-of-place forms, on a temporary nothing else holds
    lo = lambda phi: phi.clamp(min=torch.finfo(phi.dtype).tiny,
                               max=rdm.hazard_clamp(phi.dtype)).log_()
    l1 = _log1m

    sm = lambda x: x.sum(dim=-1)
    H_fwd = -(sm(lo(phi_fwd).mul_(spike_f)) + sm(l1(phi_fwd, True).mul_(p_fwd - spike_f)))
    H_rev = -(sm(lo(phi_rev).mul_(spike_r)) + sm(l1(phi_rev, True).mul_(p_rev - spike_r)))
    return torch.stack([H_rev - H_fwd, H_fwd, H_rev], dim=-1)


def poisson_reference_entropy(m: torch.Tensor,    # (...) per-bin firing probability
                              nu0: float, dt: float) -> torch.Tensor:
    """ The per-step entropy of the POISSON REFERENCE at the same spike weights:

        H_ref = -[ m ln c + (1 - m) ln(1 - c) ],      c = nu0*dt

    and the quantity to subtract from H_fwd and H_rev to report them relative to the reference
    (decided 2026-11-07). Under route B the Poisson process of rate nu0 is the model's own
    reference measure -- the kernel is a MaxCal tilt of it -- so this is the prior, not a trick
    to cancel a divergence.

    Why it is needed. H_fwd and H_rev are entropies of a binned process against a COUNTING
    measure, so each diverges as rate*ln(1/dt): a spike that costs -ln Phi ~ -ln(rate*dt) nats
    costs more the finer the bin, without limit. H_ref carries exactly that divergence,
    -m ln(nu0*dt), so

        H_rel = H - H_ref  ->  finite as dt -> 0,

    the spike term going to -<beta*h> and the silent term to O(dt). The RAW values stay the
    control CLAUDE.md asks for (increment per halving -> r ln 2); the referenced ones are what
    is comparable across resolutions.

    sigma is untouched by the choice, and that is the check on it: H_fwd and H_rev are built
    from the same total spike weight m and the same total mass 1, so they subtract the SAME
    H_ref and sigma = H_rev - H_fwd = H_rel_rev - H_rel_fwd exactly, in float as well as on
    paper. Reporting the pair relative to the reference therefore adds information about the
    two half-terms without moving the number the paper quotes.

    m is the per-bin firing probability (rates() * dt), not the rate: this is a per-STEP
    entropy, like H_fwd and H_rev, and is divided by dt for nats/ms alongside them. """
    c = nu0 * dt
    if not 0.0 < c < 1.0:
        raise ValueError(f"the Poisson reference needs 0 < c = nu0*dt < 1, got {c:.6g}")
    m = torch.as_tensor(m)
    return -(m * math.log(c) + (1.0 - m) * math.log1p(-c))


class EPRParams(NamedTuple):
    """ One population's parameters, as every observer step and init takes them. """
    R:      torch.Tensor                 # (..., Q) refractory kernel
    alpha:  float | torch.Tensor         # (..., 1) integration kernel scaling factor
    beta:   float | torch.Tensor         # (..., 1)
    theta:  float | torch.Tensor         # (..., 1)
    hazard: Hazard
    nu0:    float
    dt:     float


# Sliding-state EPR: the same quantity in O(Q) per step and O(Q) memory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Port of the recursions in the pre-port implementation, with the hazard as a parameter.
# epr_from_parts needs only four marginals of the joint, and each slides exactly from one
# timestep to the next, so the (Q, K) joint is never built. Writing Phi[tau, n] for the hazard
# at absolute time tau and age n (clamped to Q-1), L = log1p(-Phi), and J_t for the joint at
# the reported time t:
#
#   1. row sums   J_t.sum(1) == p_t                     exactly (first-passage telescoping)
#   2. column 0   J_t[:, 0]  == p_t * Phi[t, .]         one hazard row
#   3. row 0      J_t[0, :]  is the age-0 cohort's first-passage law, driven only by the
#                 diagonal Phi[t+j, j]
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
#
# What the recursions read is three things, none of them the (K, Q) look-ahead block S_{t+j}(n)
# the pre-rebuild implementation kept as a ring:
#
#   the row S_t, for Phi[t, .]: a copy of the model's S lagged K-1 steps, advanced by the
#     model's own rdm.update_synaptic_term on the stored input, so it is the model's S bit for
#     bit;
#   the row S_{t+K-1}, for the window-edge hazards: the model's LIVE S, passed in by the caller
#     at the step it is current, so the observer keeps no copy of it;
#   the two diagonals S_{t+j}(j) and S_{t+1+j}(j): the field along the cohorts born at t and
#     t+1, which depend on nothing but the inputs since their birth, and are rebuilt from the
#     input window every step by synaptic_diagonal.
#
# State per population is therefore O(Q): three (Q,) rows, G (K,), and the input window (K-1,).


def synaptic_diagonal(I: torch.Tensor,                 # (..., n) inputs I_t .. I_{t+n-1}
                      alpha: float | torch.Tensor,
                      ) -> torch.Tensor:               # (..., n+1)
    """ The synaptic term along the cohort born at t, at ages 0..n:

        D_j = S_{t+j}(j) = sum_{i<j} alpha (1-alpha)^(j-1-i) I_{t+i},    D_0 = 0,

    which is update_synaptic_term unrolled along the diagonal (a newborn's S is 0). Written as
    a kernel times a cumulative sum, the form compute_reverse_synaptic_term uses for the
    reverse field, with the forward kernel's growth folded into the summand:

        D_j = alpha (1-alpha)^(j-1) * cumsum_i [ I_{t+i} / (1-alpha)^i ].

    Chosen over the exponential running sum X_{t+j} - (1-alpha)^j X_t, which needs one more
    (K,) window of state and carries a cancellation of eps/alpha relative at small j (measured
    2e-13 at dt = 1/128, against 6e-15 here; the old recursion's own float64 diagonal is at
    9e-15). The divisor reaches exp(-U_max/tau_int) at the far end of the window, far from
    underflow for any grid the package sizes. """
    n = I.shape[-1]
    # exp(i log(1-alpha)) rather than (1-alpha)**i: the same accuracy, at an eighth of the cost
    log_keep = torch.log1p(-torch.as_tensor(alpha, dtype=I.dtype, device=I.device))
    decay = torch.exp(log_keep * torch.arange(n, device=I.device, dtype=I.dtype))
    return torch.cat([I.new_zeros(I.shape[:-1] + (1,)),
                      alpha * decay * torch.cumsum(I / decay, dim=-1)], dim=-1)


def _first_passage(d: torch.Tensor, p0: torch.Tensor, inplace: bool) -> torch.Tensor:
    """ The age-0 cohort's first-passage law along a diagonal of hazards d:
    p0 * prod_{i<j}(1 - d_i) * d_j, with the last column the bare survival. """
    log_surv = _cumlog1m(d, inplace)
    surv = torch.cat([torch.ones_like(d[..., :1]), torch.exp(log_surv[..., :-1])], dim=-1)
    del log_surv
    if inplace:
        surv.mul_(p0)[..., :-1].mul_(d[..., :-1])
        return surv
    surv = p0 * surv
    return torch.cat([surv[..., :-1] * d[..., :-1], surv[..., -1:]], dim=-1)


class EPRState(NamedTuple):
    """ What the sliding observer carries between timesteps, valid at the reported time t.
    Every field is (..., Q) or (..., K) with K == Q: O(Q) per population. Nothing here is the
    model's own state: p and S lag it by K-1 steps, the rest are sums over the window. """
    p:       torch.Tensor   # (..., Q)   p_t, the reported marginal
    S:       torch.Tensor   # (..., Q)   S_t, the model's S lagged K-1 steps
    I:       torch.Tensor   # (..., K-1) I_t .. I_{t+K-2}
    G:       torch.Tensor   # (..., K)   column sums of the joint, G[k] = p_t(n_dagger = k)
    logV:    torch.Tensor   # (..., Q)   log survival over the next K-2 steps, per age
    phi_end: torch.Tensor   # (..., Q)   Phi[t+K-2, min(n+K-2, Q-1)], the window-edge hazards


def _window_edge_ages(Q: int, device) -> torch.Tensor:
    return torch.clamp(torch.arange(Q, device=device) + (Q - 2), max=Q - 1)


def init_epr_state(p: torch.Tensor,       # (..., Q) p_t
                   S: torch.Tensor,       # (..., Q) S_t
                   I: torch.Tensor,       # (..., K-1) inputs I_t .. I_{t+K-2}
                   prm: EPRParams,
                   inplace: bool = False) -> EPRState:
    """ The sliding state at the first reported step, from S_t and the inputs that follow it.

    G, logV and phi_end are sums over the joint's lag axis, so they are accumulated one lag at
    a time, walking S forward along the window with the model's own update: row k of the
    joint's hazard is Phi[t+k, min(n+k, Q-1)], one (..., Q) gather of S_{t+k}, and the survival
    to lag k is a running log sum. The (Q, K) joint is never materialised -- the one O(Q*K)
    step of a trajectory is O(Q*K) in time and O(Q) in memory. The running sum is the same
    sequence of additions as the joint path's cumsum; G and logV are reduced in a different
    order from it, so they agree with the joint to rounding, not bit for bit. """
    R, alpha, beta, theta, hazard, nu0, dt = prm
    Q = S.shape[-1]
    K = I.shape[-1] + 1
    if K < 3:
        raise ValueError(f"sliding EPR needs a look-ahead of at least 3 rows, got K={K}")
    if K != Q:
        # the reverse field's h(k) is evaluated against R(k), so the lag axis and the age axis
        # share a length -- a constraint the joint path carries too, and entropy_trajectory
        # always satisfies it (K = Qm[i] = Q)
        raise ValueError(f"sliding EPR needs K == Q, got K={K}, Q={Q}")
    n_idx = torch.arange(Q, device=S.device)
    G = p.new_empty(p.shape[:-1] + (K,))
    S_k, acc, logV, phi_end = S, None, None, None
    for k in range(K - 1):
        if k > 0:
            # in place only on a row this function owns, never on the caller's S
            S_k = rdm.update_synaptic_term(S_k, alpha, I[..., k - 1:k], inplace=inplace and k > 1)
        # the field on the whole row, then gathered: the same elementwise sums as gathering S
        # and R first, with one (..., Q) temporary instead of three
        a = torch.clamp(n_idx + k, max=Q - 1)
        phi = hazard.phi(hazard_field(S_k, R, theta, inplace)[..., a], beta, nu0, dt,
                         inplace=inplace)
        if acc is None:
            G[..., k] = (phi * p).sum(dim=-1)                # first passage at lag k
        elif inplace:
            G[..., k] = torch.exp(acc).mul_(phi).mul_(p).sum(dim=-1)
        else:
            G[..., k] = (torch.exp(acc) * phi * p).sum(dim=-1)
        L = _log1m(phi, inplace)
        if k == K - 2:
            phi_end = phi
        del phi
        acc = L if acc is None else acc.add_(L) if inplace else acc + L
        del L
        if k == K - 3:
            logV = acc.clone()
    # the last lag is censoring, the bare survival through K-1 rows: it needs no hazard of its
    # own, so the window's last row S_{t+K-1} is never formed here -- the live S carries it
    G[..., K - 1] = (torch.exp(acc).mul_(p) if inplace else torch.exp(acc) * p).sum(dim=-1)
    return EPRState(p, S, I, G, logV, phi_end)


def _edge_field(S: torch.Tensor, R: torch.Tensor, theta, inplace: bool) -> torch.Tensor:
    """ hazard_field(S[a_end], R[a_end], theta) for the window-edge ages a_end = min(n+Q-2, Q-1),
    which are only two: Q-2 for n = 0 and Q-1 for every other n. The field is formed on those
    two and gathered after, the same elementwise sums on the same operands as gathering S and
    R first, without the two (..., Q) gathers. """
    Q = S.shape[-1]
    a = _window_edge_ages(Q, S.device) - (Q - 2)
    return hazard_field(S[..., Q - 2:], R[..., Q - 2:], theta, inplace)[..., a]


def epr_step(state: EPRState,
             S_live: torch.Tensor,        # (..., Q) the model's S at t+K-1, the window's edge
             I_new: torch.Tensor,         # (...) the model's input at t+K-1
             prm: EPRParams,
             inplace: bool = False):
    """ One timestep of the observer: (state_{t+1}, sigma, H_fwd, H_rev) at the reported t.

    The observer lags the model by K-1 steps -- the reverse field needs the input that far into
    the future -- and is fed the model's live S and live input, both read at time t+K-1 BEFORE
    the model advances. It never sees the model otherwise: S_t is advanced here, by
    rdm.update_synaptic_term on the stored inputs, and p_t by rdm.update_age_distribution on
    the Phi[t, .] row this step already needs. Three (Q,) hazard evaluations and two diagonal
    rebuilds replace the (Q, K) joint.

    The state and S_live passed in are read-only on both branches, so the returned state is
    fresh either way. Every temporary is dropped as soon as it has been read, so the step's
    peak is the new state plus a couple of (..., Q) buffers; inplace reuses the buffers this
    function allocates itself, in the same arithmetic order, and lowers it further. """
    p, S, I, G, logV, phi_end = state
    R, alpha, beta, theta, hazard, nu0, dt = prm
    K = I.shape[-1] + 1
    phi = lambda h: hazard.phi(h, beta, nu0, dt, inplace=inplace)
    fld = lambda S, Rv: hazard_field(S, Rv, theta, inplace)

    # -- present: the hazard row at time t, shared by H_fwd, column 0 and the logV slide
    phi_t = phi(fld(S, R))                                           # (Q,)
    col0 = p * phi_t                                                 # (Q,)  J_t[:, 0]

    # -- row 0: the age-0 cohort's first-passage law, off the diagonal Phi[t+j, j] (K == Q, so
    # the diagonal's ages are R's own and need no gather)
    d = phi(fld(synaptic_diagonal(I, alpha), R))                     # (..., K)
    row0 = _first_passage(d, p[..., :1], inplace)                    # (..., K), last col bare
    del d

    I_full = torch.cat([I, I_new.unsqueeze(-1)], dim=-1)             # I_t .. I_{t+K-1}
    phi_r = phi(fld(anchor_reverse_synaptic_term(I_full, alpha), R))
    del I_full              # rebuilt as the next window at the end, rather than held through
    epr = epr_from_parts(p, G, col0, row0, phi_t, phi_r)
    del col0, row0, phi_r

    p_next = rdm.update_age_distribution(p, phi_t)

    # -- slide logV: drop time t off the front of the window, add t+K-2 at the back. Index Q-1
    # takes the lumped-bin value rather than the shifted one; the two agree analytically (both
    # merged cohorts see the same clamped future) and this stays exact if the shift falls off.
    L_t = _log1m(phi_t, inplace)
    del phi_t
    L_e = _log1m(phi_end, inplace)
    W = L_t.neg_().add_(logV).add_(L_e) if inplace else logV - L_t + L_e
    del L_t, L_e
    logV_next = torch.empty_like(W)
    logV_next[..., 1:] = W[..., :-1]
    logV_next[..., -1] = W[..., -1]
    del W

    # -- the newborn cohort's own window, and the next step's row 0, share one diagonal: the
    # cohort born at t+1, at ages 0..K-3
    d2 = phi(fld(synaptic_diagonal(I[..., 1:K - 2], alpha), R[..., :K - 2]))
    log_surv2 = _cumlog1m(d2, inplace)
    logV_next[..., 0] = log_surv2[..., -1]
    surv2 = torch.cat([torch.ones_like(d2[..., :1]), torch.exp(log_surv2[..., :-1])], dim=-1)
    del log_surv2
    row0_next = surv2.mul_(p_next[..., :1]).mul_(d2) if inplace else \
                p_next[..., :1] * surv2 * d2                         # (..., K-2)
    del d2, surv2

    # -- slide G: shift-and-add on the interior, V-weighted sums on the two boundary lags
    G_next = torch.empty_like(G)
    G_next[..., :K - 2] = row0_next.add_(G[..., 1:K - 1]) if inplace else \
                          G[..., 1:K - 1] + row0_next
    del row0_next
    phi_end_next = phi(_edge_field(S_live, R, theta, inplace))       # Phi[t+K-1, .]
    if inplace:
        pV = torch.exp(logV_next).mul_(p_next)
        w = torch.mul(pV, phi_end_next)
        G_next[..., K - 2] = w.sum(dim=-1)
        G_next[..., K - 1] = w.copy_(phi_end_next).neg_().add_(1).mul_(pV).sum(dim=-1)
        del pV, w
    else:
        pV = p_next * torch.exp(logV_next)
        G_next[..., K - 2] = (pV * phi_end_next).sum(dim=-1)
        G_next[..., K - 1] = (pV * (1 - phi_end_next)).sum(dim=-1)
        del pV

    S_next = rdm.update_synaptic_term(S.clone(), alpha, I[..., :1], inplace=True) if inplace \
        else rdm.update_synaptic_term(S, alpha, I[..., :1])
    I_next = torch.cat([I[..., 1:], I_new.unsqueeze(-1)], dim=-1)   # I_{t+1} .. I_{t+K-1}
    nxt = EPRState(p_next, S_next, I_next, G_next, logV_next, phi_end_next)
    return nxt, epr[..., 0], epr[..., 1], epr[..., 2]
