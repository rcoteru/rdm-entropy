""" Entropy production, by the O(Q) observer.

The EPR needs only four marginals of the joint distribution of forward and reverse age, and
each slides exactly from one timestep to the next, so epr_step reaches them by recursion at
O(Q) per step and O(Q) memory, and epr_from_parts reads the EPR off them. The explicit (Q, K)
joint -- the definition, at O(Q^2) per step -- is kept at the end of this module as the oracle
the observer is checked against, and calls the same epr_from_parts, so the two can only
disagree through the four marginals.

The EPR is a functional of the hazard alone, so one implementation serves every kernel; that
is why it is written as the plain Bernoulli entropy in Phi rather than in the softplus/logit
form of the pre-port implementation, which is an identity of the logistic kernel only.

Everything here indexes on trailing axes, so one implementation also serves one system and a
whole batch: shapes are written (..., Q) and (..., K), with the leading axes carrying the batch
or nothing at all. """

from __future__ import annotations

import math

import torch

import rdme.equations as eq
from rdme.hazards import Hazard

# Time-reversed field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The backward synaptic term the entropy production pairs against the forward one. Kernel-
# independent: it filters the future input and never evaluates a hazard, so it is shared by
# every member of the family.

def compute_reverse_synaptic_term(
        I: torch.Tensor,       # (..., Q) buffered I[j] = input at time t+j
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        ) -> torch.Tensor:             # (..., Q) backward field S^dagger_t(n)
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
# Every function below takes the `inplace` flag of rdme.equations and rdme.hazards, with the
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


class _Kernel:
    """ One population's hazard with its parameters bound: Phi and the exact ln Phi and
    ln(1 - Phi) branches of rdme.hazards, as functions of the field alone.

    Every log the EPR takes comes from the kernel's own closed form, never from Phi: at strong
    drive Phi rounds to one, and ln(1 - Phi) read off it is -inf, or, clamped, a floor set by
    the clamp rather than by the model -- which in the saturated regime is what H_rev is made
    of. The logs are evaluated out of place, so a caller takes them FIRST and hands the field
    to phi last, which with inplace may consume it.

    Every method takes the absolute window's mask `blk`, gathered at the same (time, age)
    indices as h, and returns the window's exact values there: Phi = 0, ln(1 - Phi) = 0 and
    ln Phi = -inf. The -inf is deliberate rather than a finite stand-in: its weight is
    exactly 0 (see epr_from_parts), and a non-zero one would surface as an infinite H instead
    of being hidden. """

    def __init__(self, hazard: Hazard, beta, nu0: float, dt: float, inplace: bool):
        self.hazard, self.beta, self.nu0, self.dt, self.inplace = hazard, beta, nu0, dt, inplace

    def _mask(self, x, blk, fill: float):
        return x.masked_fill_(blk, fill) if self.inplace else x.masked_fill(blk, fill)

    def phi(self, h, blk):
        return self._mask(self.hazard.phi(h, self.beta, self.nu0, self.dt, inplace=self.inplace),
                          blk, 0.0)

    def logphi(self, h, blk):
        return self._mask(self.hazard.logphi(h, self.beta, self.nu0, self.dt), blk, -math.inf)

    def log1m(self, h, blk):
        return self._mask(self.hazard.log1m_phi(h, self.beta, self.nu0, self.dt), blk, 0.0)


def _cumsum(L: torch.Tensor, inplace: bool) -> torch.Tensor:
    """ The running log-survival sum_{j<=k} ln(1 - Phi_j), in L's own buffer with inplace. """
    return L.cumsum_(-1) if inplace else torch.cumsum(L, dim=-1)


def epr_from_parts(p_fwd: torch.Tensor,    # (..., Q) row sums of the joint == p_t
                   p_rev: torch.Tensor,    # (..., K) column sums   == p_t(n_dagger)
                   spike_f: torch.Tensor,  # (..., Q) J[:, 0], the forward spike weight
                   spike_r: torch.Tensor,  # (..., K) J[0, :], the reverse spike weight
                   lp_fwd: torch.Tensor,   # (..., Q) ln Phi of the forward hazard at t, by age
                   l1_fwd: torch.Tensor,   # (..., Q) ln(1 - Phi), same
                   lp_rev: torch.Tensor,   # (..., K) ln Phi of the reverse hazard at t+1
                   l1_rev: torch.Tensor,   # (..., K) ln(1 - Phi), same
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
    which happens iff k == 0; the reverse emits s(t), iff n == 0. The logs are the kernel's
    exact branches (see _Kernel): finite wherever the field is, except ln Phi = -inf on the
    absolute window's blocked bins. Every product is taken with 0 * l = 0 (_wlog), the 0 ln 0
    = 0 convention, which is exact there: a blocked bin's spike weight is exactly 0, forward
    (p * Phi with Phi = 0) and reverse (a first passage through Phi = 0). A non-zero weight
    against -inf is not masked and gives an infinite H. The logs are read, not written: the
    sliding path still needs the forward ln(1 - Phi) after this. """
    sm = lambda x: x.sum(dim=-1)
    H_fwd = -(sm(_wlog(spike_f, lp_fwd)) + sm(_wlog(p_fwd - spike_f, l1_fwd)))
    H_rev = -(sm(_wlog(spike_r, lp_rev)) + sm(_wlog(p_rev - spike_r, l1_rev)))
    return torch.stack([H_rev - H_fwd, H_fwd, H_rev], dim=-1)


def _wlog(w: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """ w * l, with w = 0 giving exactly 0 whatever l is, -inf included (xlogy's convention,
    on a log that is already taken). The only product of a weight and a log in the EPR. """
    return torch.where(w == 0, 0.0, w * l)


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
    control (increment per halving -> r ln 2); the referenced ones are what
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


# Sliding-state EPR: the same quantity in O(Q) per step and O(Q) memory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Port of the recursions in the pre-port implementation, with the hazard as a parameter.
# epr_from_parts needs only four marginals of the joint, and each slides exactly from one
# timestep to the next, so the (Q, K) joint is never built. Writing Phi[tau, n] for the hazard
# at absolute time tau and age n (clamped to Q-1), L = ln(1 - Phi) (the kernel's exact branch,
# see _Kernel), and J_t for the joint at the reported time t:
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
#     model's own eq.update_synaptic_term on the stored input, so it is the model's S bit for
#     bit;
#   the row S_{t+K-1}, for the window-edge hazards: the model's LIVE S, passed in by the caller
#     at the step it is current, so the observer keeps no copy of it;
#   the two diagonals S_{t+j}(j) and S_{t+1+j}(j): the field along the cohorts born at t and
#     t+1, which depend on nothing but the inputs since their birth, and are rebuilt from the
#     input window every step by synaptic_diagonal.
#
# State per population is therefore O(Q): three (Q,) rows, G (K,), and the input window (K-1,).
# It is carried as a plain tuple, valid at the reported time t:
#
#   (p, S, I, G, logV, L_end)
#     p        (..., Q)    p_t, the reported marginal, lagging the model by K-1 steps
#     S        (..., Q)    S_t, the model's S lagged K-1 steps
#     I        (..., K-1)  I_t .. I_{t+K-2}
#     G        (..., K)    column sums of the joint, G[k] = p_t(n_dagger = k)
#     logV     (..., Q)    log survival over the next K-2 steps, per age
#     L_end    (..., Q)    ln(1 - Phi[t+K-2, min(n+K-2, Q-1)]), at the window-edge hazards
#
# Every init and step takes one population's parameters as the trailing positional arguments
# R (..., Q), blocked (..., Q) (the absolute window, bool), alpha, beta, theta (each (..., 1) or
# a float), hazard, nu0, dt. Wherever the field is gathered at some (time, age), the mask is
# gathered at the same ages and handed to the _Kernel with it.


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


def _first_passage(d: torch.Tensor, L: torch.Tensor, p0: torch.Tensor,
                   inplace: bool) -> torch.Tensor:
    """ The age-0 cohort's first-passage law along a diagonal of hazards d, with L = ln(1 - d):
    p0 * prod_{i<j}(1 - d_i) * d_j, with the last column the bare survival. With inplace, L's
    buffer is reused. """
    log_surv = _cumsum(L, inplace)
    surv = torch.cat([torch.ones_like(d[..., :1]), torch.exp(log_surv[..., :-1])], dim=-1)
    del log_surv
    if inplace:
        surv.mul_(p0)[..., :-1].mul_(d[..., :-1])
        return surv
    surv = p0 * surv
    return torch.cat([surv[..., :-1] * d[..., :-1], surv[..., -1:]], dim=-1)


def _window_edge_ages(Q: int, device) -> torch.Tensor:
    return torch.clamp(torch.arange(Q, device=device) + (Q - 2), max=Q - 1)


def init_epr_state(p: torch.Tensor,       # (..., Q) p_t
                   S: torch.Tensor,       # (..., Q) S_t
                   I: torch.Tensor,       # (..., K-1) inputs I_t .. I_{t+K-2}
                   R: torch.Tensor,
                   blocked: torch.Tensor,
                   alpha: float | torch.Tensor,
                   beta: float | torch.Tensor,
                   theta: float | torch.Tensor,
                   hazard: Hazard,
                   nu0: float,
                   dt: float,
                   inplace: bool = False) -> tuple:
    """ The sliding state (p, S, I, G, logV, L_end) at the first reported step, from S_t and
    the inputs that follow it.

    G, logV and L_end are sums over the joint's lag axis, so they are accumulated one lag at
    a time, walking S forward along the window with the model's own update: row k of the
    joint's hazard is Phi[t+k, min(n+k, Q-1)], one (..., Q) gather of S_{t+k}, and the survival
    to lag k is a running log sum. The (Q, K) joint is never materialised -- the one O(Q*K)
    step of a trajectory is O(Q*K) in time and O(Q) in memory. The running sum is the same
    sequence of additions as the joint path's cumsum; G and logV are reduced in a different
    order from it, so they agree with the joint to rounding, not bit for bit. """
    Q = S.shape[-1]
    K = I.shape[-1] + 1
    if K < 3:
        raise ValueError(f"sliding EPR needs a look-ahead of at least 3 rows, got K={K}")
    if K != Q:
        # the reverse field's h(k) is evaluated against R(k), so the lag axis and the age axis
        # share a length -- a constraint the joint path carries too, and entropy_trajectory
        # always satisfies it (K = Q)
        raise ValueError(f"sliding EPR needs K == Q, got K={K}, Q={Q}")
    n_idx = torch.arange(Q, device=S.device)
    kern = _Kernel(hazard, beta, nu0, dt, inplace)
    G = p.new_empty(p.shape[:-1] + (K,))
    S_k, acc, logV, L_end = S, None, None, None
    for k in range(K - 1):
        if k > 0:
            # in place only on a row this function owns, never on the caller's S
            S_k = eq.update_synaptic_term(S_k, alpha, I[..., k - 1:k], inplace=inplace and k > 1)
        # the field on the whole row, then gathered: the same elementwise sums as gathering S
        # and R first, with one (..., Q) temporary instead of three
        a = torch.clamp(n_idx + k, max=Q - 1)
        h, blk = hazard_field(S_k, R, theta, inplace)[..., a], blocked[..., a]
        L = kern.log1m(h, blk)
        phi = kern.phi(h, blk)
        del h
        if acc is None:
            G[..., k] = (phi * p).sum(dim=-1)                # first passage at lag k
        elif inplace:
            G[..., k] = torch.exp(acc).mul_(phi).mul_(p).sum(dim=-1)
        else:
            G[..., k] = (torch.exp(acc) * phi * p).sum(dim=-1)
        if k == K - 2:
            # k >= 1 here, so acc is not L's buffer and the add_ below leaves L alone
            L_end = L
        del phi
        acc = L if acc is None else acc.add_(L) if inplace else acc + L
        del L
        if k == K - 3:
            logV = acc.clone()
    # the last lag is censoring, the bare survival through K-1 rows: it needs no hazard of its
    # own, so the window's last row S_{t+K-1} is never formed here -- the live S carries it
    assert acc is not None                                 # K >= 3: set from k = 1 on
    G[..., K - 1] = (torch.exp(acc).mul_(p) if inplace else torch.exp(acc) * p).sum(dim=-1)
    return p, S, I, G, logV, L_end


def _edge_field(S: torch.Tensor, R: torch.Tensor, blocked: torch.Tensor, theta,
                inplace: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """ hazard_field(S[a_end], R[a_end], theta) for the window-edge ages a_end = min(n+Q-2, Q-1),
    which are only two: Q-2 for n = 0 and Q-1 for every other n. The field is formed on those
    two and gathered after, the same elementwise sums on the same operands as gathering S and
    R first, without the two (..., Q) gathers. Returns it with the mask at the same ages. """
    Q = S.shape[-1]
    a = _window_edge_ages(Q, S.device) - (Q - 2)
    return (hazard_field(S[..., Q - 2:], R[..., Q - 2:], theta, inplace)[..., a],
            blocked[..., Q - 2:][..., a])


def _epr_now(p, G, I, I_new, phi_t, lp_t, L_t, R, blocked, alpha, theta, kern: _Kernel,
             inplace: bool) -> torch.Tensor:
    """ (..., 3) [sigma, H_fwd, H_rev] at the reported t, from the four marginals of J_t: the
    row sums p and column sums G the state carries, column 0 off the hazard row phi_t, and
    row 0 off the age-0 cohort's diagonal. lp_t and L_t are ln phi_t and ln(1 - phi_t). """
    col0 = p * phi_t                                                 # (Q,)  J_t[:, 0]

    # row 0: the age-0 cohort's first-passage law, off the diagonal Phi[t+j, j] (K == Q, so
    # the diagonal's ages are R's own and need no gather)
    h = hazard_field(synaptic_diagonal(I, alpha), R, theta, inplace)
    L = kern.log1m(h, blocked)
    d = kern.phi(h, blocked)                                         # (..., K)
    del h
    row0 = _first_passage(d, L, p[..., :1], inplace)                 # (..., K), last col bare
    del d, L

    I_full = torch.cat([I, I_new.unsqueeze(-1)], dim=-1)             # I_t .. I_{t+K-1}
    h_r = hazard_field(anchor_reverse_synaptic_term(I_full, alpha), R, theta, inplace)
    del I_full
    lp_r, L_r = kern.logphi(h_r, blocked), kern.log1m(h_r, blocked)  # the reverse needs no Phi
    del h_r
    return epr_from_parts(p, G, col0, row0, lp_t, L_t, lp_r, L_r)


def _newborn_cohort(I, p0_next, R, blocked, alpha, theta, kern: _Kernel, inplace: bool):
    """ The cohort born at t+1, at ages 0..K-3, whose one diagonal of hazards gives both its
    own survival window logV_{t+1}(0) and the next step's row 0 up to lag K-3. Returns
    (logV0 (...), row0_next (..., K-2)). """
    K = I.shape[-1] + 1
    h = hazard_field(synaptic_diagonal(I[..., 1:K - 2], alpha), R[..., :K - 2], theta, inplace)
    blk = blocked[..., :K - 2]
    L2 = kern.log1m(h, blk)
    d2 = kern.phi(h, blk)
    del h
    log_surv2 = _cumsum(L2, inplace)
    del L2
    logV0 = log_surv2[..., -1]
    surv2 = torch.cat([torch.ones_like(d2[..., :1]), torch.exp(log_surv2[..., :-1])], dim=-1)
    del log_surv2
    row0_next = surv2.mul_(p0_next).mul_(d2) if inplace else p0_next * surv2 * d2
    return logV0, row0_next


def _slide_logV(logV, L_t, L_e, logV0, inplace: bool) -> torch.Tensor:
    """ logV_{t+1}: drop time t off the front of the window, add t+K-2 at the back, and shift
    one age up, with the newborn's logV0 at age 0. L_t and L_e are ln(1 - Phi) at the hazard
    row t and the window edge t+K-2; with inplace, L_t's buffer is reused. Index Q-1 takes the
    lumped-bin value rather than the shifted one; the two agree analytically (both merged
    cohorts see the same clamped future) and this stays exact if the shift falls off. """
    W = L_t.neg_().add_(logV).add_(L_e) if inplace else logV - L_t + L_e
    del L_t
    logV_next = torch.empty_like(W)
    logV_next[..., 1:] = W[..., :-1]
    logV_next[..., -1] = W[..., -1]
    logV_next[..., 0] = logV0
    return logV_next


def _slide_G(G, row0_next, logV_next, p_next, phi_end_next, inplace: bool) -> torch.Tensor:
    """ G_{t+1}: shift-and-add on the interior lags, V-weighted sums on the two boundary ones
    (the first passage at K-2 and the censoring bin at K-1). """
    K = G.shape[-1]
    G_next = torch.empty_like(G)
    G_next[..., :K - 2] = row0_next.add_(G[..., 1:K - 1]) if inplace else \
                          G[..., 1:K - 1] + row0_next
    if inplace:
        pV = torch.exp(logV_next).mul_(p_next)
        w = torch.mul(pV, phi_end_next)
        G_next[..., K - 2] = w.sum(dim=-1)
        G_next[..., K - 1] = w.copy_(phi_end_next).neg_().add_(1).mul_(pV).sum(dim=-1)
    else:
        pV = p_next * torch.exp(logV_next)
        G_next[..., K - 2] = (pV * phi_end_next).sum(dim=-1)
        G_next[..., K - 1] = (pV * (1 - phi_end_next)).sum(dim=-1)
    return G_next


def epr_step(state: tuple,                # (p, S, I, G, logV, L_end) at the reported t
             S_live: torch.Tensor,        # (..., Q) the model's S at t+K-1, the window's edge
             I_new: torch.Tensor,         # (...) the model's input at t+K-1
             R: torch.Tensor,
             blocked: torch.Tensor,
             alpha: float | torch.Tensor,
             beta: float | torch.Tensor,
             theta: float | torch.Tensor,
             hazard: Hazard,
             nu0: float,
             dt: float,
             inplace: bool = False):
    """ One timestep of the observer: (state_{t+1}, sigma, H_fwd, H_rev) at the reported t.
    The state is (p, S, I, G, logV, L_end); see the layout above.

    The observer lags the model by K-1 steps -- the reverse field needs the input that far into
    the future -- and is fed the model's live S and live input, both read at time t+K-1 BEFORE
    the model advances. It never sees the model otherwise: S_t is advanced here, by
    eq.update_synaptic_term on the stored inputs, and p_t by eq.update_age_distribution on
    the Phi[t, .] row this step already needs. Three (Q,) hazard evaluations and two diagonal
    rebuilds replace the (Q, K) joint.

    The state and S_live passed in are read-only on both branches, so the returned state is
    fresh either way. Every temporary is dropped as soon as it has been read, so the step's
    peak is the new state plus a couple of (..., Q) buffers; inplace reuses the buffers this
    function allocates itself, in the same arithmetic order, and lowers it further. """
    p, S, I, G, logV, L_end = state
    kern = _Kernel(hazard, beta, nu0, dt, inplace)

    # the hazard row at time t, shared by H_fwd, column 0, the p update and the logV slide
    h = hazard_field(S, R, theta, inplace)
    lp_t, L_t = kern.logphi(h, blocked), kern.log1m(h, blocked)
    phi_t = kern.phi(h, blocked)                                     # (Q,)
    del h
    epr = _epr_now(p, G, I, I_new, phi_t, lp_t, L_t, R, blocked, alpha, theta, kern, inplace)
    del lp_t
    p_next = eq.update_age_distribution(p, phi_t)
    del phi_t

    logV0, row0_next = _newborn_cohort(I, p_next[..., :1], R, blocked, alpha, theta, kern,
                                       inplace)
    logV_next = _slide_logV(logV, L_t, L_end, logV0, inplace)
    del L_t, logV0
    h, blk = _edge_field(S_live, R, blocked, theta, inplace)
    L_end_next = kern.log1m(h, blk)
    phi_end_next = kern.phi(h, blk)                                  # Phi[t+K-1, .]
    del h, blk
    G_next = _slide_G(G, row0_next, logV_next, p_next, phi_end_next, inplace)
    del row0_next

    S_next = eq.update_synaptic_term(S.clone(), alpha, I[..., :1], inplace=True) if inplace \
        else eq.update_synaptic_term(S, alpha, I[..., :1])
    I_next = torch.cat([I[..., 1:], I_new.unsqueeze(-1)], dim=-1)   # I_{t+1} .. I_{t+K-1}
    nxt = (p_next, S_next, I_next, G_next, logV_next, L_end_next)
    return nxt, epr[..., 0], epr[..., 1], epr[..., 2]


# The joint-distribution reference
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The EPR read off the explicit (Q, K) joint of forward and reverse age: the definition, at
# O(Q^2) per step. It is the oracle the sliding observer above is checked against, selected
# with RDMNetwork.entropy_trajectory(method="joint"). It shares epr_from_parts with the
# observer, so the two can only disagree through the four marginals they hand it.

def joint_distribution(p: torch.Tensor,        # (Q,)   p_t(n)
                       buff_S: torch.Tensor,   # (K, Q) buff_S[j, n] = S_{t+j}(n), j=0 is t
                       R: torch.Tensor,        # (Q,)
                       blocked: torch.Tensor,  # (Q,) bool, the absolute window
                       beta: float | torch.Tensor,
                       theta: float | torch.Tensor,
                       hazard: Hazard,
                       nu0: float,
                       dt: float,
                       inplace: bool = False,
                       ) -> torch.Tensor:      # (Q, K) p_t(n, n_dagger=k)
    """ Joint distribution of forward and reverse age, built from hazards.

    The hazard-parametrised port of mean_field.compute_joint_distribution; the structure is
    unchanged. A cohort's survival prod_{j<k} (1 - Phi_{t+j}(n+j)) is built by multiplying
    hazards rather than reading density ratios, which is what lets it track a cohort through
    the lumped final bin and means only the marginal p_t is ever needed. Rows sum to 1 by
    telescoping, so no normalization.

    inplace folds the chain of (Q, K) temporaries into three live buffers, in the same order so
    the two branches agree bit for bit; buff_S, R and p are read-only either way, only the
    intermediates this function allocates itself are reused. It is the expensive flag in the
    package: this is the one (B, Q, K) object in an EPR trajectory, so each temporary dropped
    is the whole joint again. """
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

    kern = _Kernel(hazard, beta, nu0, dt, inplace)
    h_all = hazard_field(buff_S, R, theta, inplace)
    # the two index tensors consume the last two axes, so the ellipsis carries any batch
    h = h_all[..., future_time, future_age]                         # (..., Q, K)
    del h_all               # the gather already copied; holding the source doubles the peak
    blk = blocked[..., future_age]                                  # (..., Q, K), same ages
    L = kern.log1m(h, blk)
    phi = kern.phi(h, blk)
    del h

    if inplace:
        log_surv = _cumsum(L, True)
        surv = torch.empty_like(phi)
        surv[..., 0] = 1.0
        torch.exp(log_surv[..., :-1], out=surv[..., 1:])
        del log_surv
        surv[..., :-1].mul_(phi[..., :-1])   # first passage; last column keeps bare S
        return surv.mul_(p.unsqueeze(-1))

    log_surv = _cumsum(L, False)
    surv = torch.ones_like(phi)
    surv[..., 1:] = torch.exp(log_surv[..., :-1])
    surv[..., :-1] = surv[..., :-1] * phi[..., :-1]  # first passage; last column keeps bare S
    return surv * p.unsqueeze(-1)



def epr_from_joint(p_joint: torch.Tensor,   # (Q, K)
                   lp_fwd: torch.Tensor,    # (Q,) ln Phi, forward
                   l1_fwd: torch.Tensor,    # (Q,) ln(1 - Phi), forward
                   lp_rev: torch.Tensor,    # (K,) ln Phi, reverse
                   l1_rev: torch.Tensor,    # (K,) ln(1 - Phi), reverse
                   ) -> torch.Tensor:       # (3,) [sigma, H_fwd, H_rev]
    """ The EPR read off the explicit joint: the reference path, and the oracle the sliding
    recursions are checked against. """
    return epr_from_parts(p_joint.sum(dim=-1), p_joint.sum(dim=-2),
                          p_joint[..., :, 0], p_joint[..., 0, :], lp_fwd, l1_fwd, lp_rev, l1_rev)



def synaptic_window(S: torch.Tensor,      # (..., Q) S_t
                    I: torch.Tensor,      # (..., K-1) inputs I_t .. I_{t+K-2}
                    alpha: float | torch.Tensor,
                    ) -> torch.Tensor:    # (..., K, Q) row j = S_{t+j}
    """ The look-ahead block the joint path reads, replayed from S_t and the input window by
    eq.update_synaptic_term -- the model's own update on the model's own inputs, so every row
    is bit for bit the S the model held at that time. O(K*Q): the oracle's cost, not the
    observer's. """
    out = S.new_empty(S.shape[:-1] + (I.shape[-1] + 1, S.shape[-1]))
    out[..., 0, :] = S
    for j in range(I.shape[-1]):
        out[..., j + 1, :] = eq.update_synaptic_term(out[..., j, :], alpha, I[..., j:j + 1])
    return out


def epr_step_joint(state: tuple,       # (p, S, I) at the reported t
                   S_live, I_new, R, blocked, alpha, beta, theta, hazard: Hazard, nu0: float,
                   dt: float, inplace: bool = False):
    """ The reference path as an observer: same feed and same return as epr_step, but the EPR
    is read off the explicit (Q, K) joint, rebuilt every step from the lagged S and the input
    window. O(Q^2) per step; it is the oracle the sliding observer is checked against. Its
    state is only (p_t, S_t, I_t .. I_{t+K-2}); it replays the window's last row itself, so
    S_live is taken for the shared interface only. """
    p, S, I = state
    I_full = torch.cat([I, I_new.unsqueeze(-1)], dim=-1)                # I_t .. I_{t+K-1}
    kern = _Kernel(hazard, beta, nu0, dt, inplace)
    J = joint_distribution(p, synaptic_window(S, I, alpha), R, blocked, beta, theta,
                           hazard, nu0, dt, inplace=inplace)
    h_f = hazard_field(S, R, theta, inplace)
    lp_f, l1_f = kern.logphi(h_f, blocked), kern.log1m(h_f, blocked)
    phi_f = kern.phi(h_f, blocked)
    h_r = hazard_field(anchor_reverse_synaptic_term(I_full, alpha), R, theta, inplace)
    e = epr_from_joint(J, lp_f, l1_f, kern.logphi(h_r, blocked), kern.log1m(h_r, blocked))
    nxt = (eq.update_age_distribution(p, phi_f),
           eq.update_synaptic_term(S, alpha, I[..., :1]),
           I_full[..., 1:])
    return nxt, e[..., 0], e[..., 1], e[..., 2]


def init_joint_state(p, S, I, R, blocked, alpha, beta, theta, hazard: Hazard, nu0: float,
                     dt: float,
                     inplace: bool = False) -> tuple:
    """ The joint path's state (p, S, I) at the first reported step: it carries the inputs and
    rebuilds everything else every step. Same signature as init_epr_state. """
    return p, S, I
