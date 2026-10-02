from __future__ import annotations

from typing import Callable, NamedTuple

import torch

""" Lyapunov exponents by the Benettin algorithm, for discrete-time maps.

Organised by topic rather than by how model-specific each piece is, so everything about the
tangent dynamics is in one place. The boundary still matters and is marked by the section
headers: the tangent map below evaluates a hazard, while the driver above it knows nothing
about populations or age distributions -- it takes an `advance` callable that steps a state
together with a basis, and is tested against maps whose spectra are known in closed form.

Basis convention: a list of blocks, each (B, k, Q_b). B indexes independent systems, k indexes
tangent directions, Q_b is the b-th block of a ragged state vector. Inner products reduce over
the block axis and the last axis, so a state split across several differently-sized blocks
never has to be concatenated into one contiguous vector.

Everything here is for a *map*, not a flow. An attracting periodic orbit of a map has all
exponents strictly negative; a zero exponent means an invariant circle. Exponents are returned
per unit time (divided by dt), which is the only form comparable across discretizations. """


# Tangent dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The model-specific half: the derivative of one step of the dynamics, and the basis it acts on.

def update_tangent(blocks: list[torch.Tensor],   # 2M blocks [dp_0, dS_0, ...], each (k, Q_i)
                   p: list[torch.Tensor],        # M age masses, each (Q_i,)
                   fprobs: list[torch.Tensor],   # M hazards at the CURRENT state
                   dphi: list[torch.Tensor],     # M hazard DERIVATIVES dPhi/dh, same shapes
                   w: torch.Tensor,              # (M, M), unscaled
                   alpha: list[torch.Tensor],
                   dt: float,
                   ) -> list[torch.Tensor]:
    """ One step of the linearized dynamics.

    Same algebra as mean_field.update_tangent with two changes. The hazard derivative is passed
    in rather than computed as beta*Phi*(1-Phi), which is the sigmoid's and only the sigmoid's;
    and dI is built from the rate, dI = w @ (dp[..., 0]/dt), matching this module's unscaled
    weights.

    Kept separate from mean_field's rather than generalising that one, so the reference
    implementation stays untouched. Its correctness is checked against
    jacfwd(RDMNetwork.state_map) in the tests, independently of mean_field. """
    M = len(p)
    dm = torch.stack([blocks[2 * i][..., 0] for i in range(M)], dim=-1)   # (k, M)
    dI = (w.unsqueeze(-3) @ (dm / dt).unsqueeze(-1)).squeeze(-1)          # (k, M)

    out: list[torch.Tensor] = []
    for i in range(M):
        p_i, fp_i = p[i].unsqueeze(-2), fprobs[i].unsqueeze(-2)   # (1, Q), broadcast over k
        dp, dS = blocks[2 * i], blocks[2 * i + 1]
        dfp = dphi[i].unsqueeze(-2) * dS

        dp_new = torch.zeros_like(dp)
        dp_new[..., 0]    = (dp * fp_i + p_i * dfp).sum(dim=-1)
        dp_new[..., 1:-1] = dp[..., :-2] * (1 - fp_i[..., :-2]) - p_i[..., :-2] * dfp[..., :-2]
        dp_new[..., -1]   = -dp_new[..., :-1].sum(dim=-1)

        dS_new = torch.zeros_like(dS)
        dS_new[..., 1:] = alpha[i] * dI[..., i].unsqueeze(-1) + (1 - alpha[i]) * dS[..., :-1]

        out.append(dp_new)
        out.append(dS_new)
    return out


def init_tangent_blocks(Qm: list[int],
                        k: int,
                        prefix: tuple[int, ...] = (),   # leading axes, e.g. (B,) for a batch
                        device: str | torch.device = 'cpu',
                        generator: torch.Generator | None = None,
                        ) -> list[torch.Tensor]:        # 2M blocks, each (*prefix, k, Q_i)
    """ A random orthonormal tangent basis, laid out as the state is: per population a block
    for delta-p and one for delta-S, interleaved [dp_0, dS_0, dp_1, dS_1, ...] to match
    RDMNetwork.flatten_state.

    The dp blocks are centred so sum(dp_i) = 0. Perturbations that change an age distribution's
    total mass leave the physical manifold; update_tangent annihilates them in one step whatever
    comes in, but starting inside the subspace means the first accumulation block does not log
    that annihilation as if it were contraction. """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    dtype = torch.get_default_dtype()
    blocks = []
    for Q in Qm:
        dp = torch.randn(*prefix, k, Q, device=device, dtype=dtype, generator=generator)
        blocks.append(dp - dp.mean(dim=-1, keepdim=True))            # onto sum(dp) = 0
        blocks.append(torch.randn(*prefix, k, Q, device=device, dtype=dtype, generator=generator))
    orthonormalize(blocks)
    return blocks


# Orthonormalization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def orthonormalize(blocks: list[torch.Tensor],   # each (..., k, Q_b), modified IN PLACE
                   eps: float = 1e-300,
                   ) -> torch.Tensor:            # (..., k) log of each column's norm
    """ Modified Gram-Schmidt over a ragged block list, in place, returning log|R_ii|.

    Modified rather than classical: classical GS loses orthogonality catastrophically once the
    columns have been squeezed toward the dominant direction between renormalizations, which is
    exactly the regime a Lyapunov run lives in. MGS subtracts each projection as it is computed,
    so later columns are orthogonalized against the already-corrected earlier ones.

    Hand-written rather than torch.linalg.qr or a Cholesky-QR on the Gram matrix, which is the
    obvious review comment and is slower here for two reasons. Both alternatives want one
    contiguous (B, k, sum(Q_b)) matrix, so every call pays a cat of the blocks; and both are
    matmul-bound, while consumer Ada runs float64 matmul at about 1/64 of its float32 rate.
    Measured at B=2197, k=6, sum(Q_b)=2306: MGS 12.0 ms, Cholesky-QR 31.1 ms. MGS also gives
    log|R_ii| for free as the normalization it performs anyway.

    Returns the logs rather than the norms because that is what the accumulator needs and
    because the product over many intervals would otherwise underflow. """
    lead, k = blocks[0].shape[:-2], blocks[0].shape[-2]
    logs = torch.empty(*lead, k, device=blocks[0].device, dtype=blocks[0].dtype)

    for j in range(k):
        for i in range(j):                    # subtract projections onto the corrected columns < j
            # <v_j, v_i> reduced over every block and its age axis -> (..., 1)
            dot = sum((b[..., j, :] * b[..., i, :]).sum(dim=-1) for b in blocks).unsqueeze(-1)
            for b in blocks:
                b[..., j, :] -= dot * b[..., i, :]
        nrm = torch.sqrt(sum((b[..., j, :] ** 2).sum(dim=-1) for b in blocks))      # (...,)
        logs[..., j] = torch.log(nrm.clamp_min(eps))
        inv = 1.0 / nrm.clamp_min(eps)
        for b in blocks:
            b[..., j, :] *= inv.unsqueeze(-1)

    return logs


# Benettin driver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Spectrum(NamedTuple):
    """ Output of `benettin`. Leading axes are whatever the caller's basis carried: none for a
    single system, (B,) for a batch. """
    lam:        torch.Tensor   # (..., k) exponents per unit time, descending
    se:         torch.Tensor   # (..., k) standard error of lam, from the block means
    partial:    torch.Tensor   # (..., k) partial sums, cumsum of lam -- see `degenerate`
    drift:      torch.Tensor   # (..., k) second-half minus first-half estimate (a bias check)
    degenerate: torch.Tensor   # (..., k) bool: |lam_i - lam_{i+1}| < se, so the pair is unresolved
    valid:      torch.Tensor   # (..., k) bool: |lam_i| * dt < 0.1, so lam/dt is a rate not an artifact
    n_blocks:   int
    T:          float          # measurement window in time units, not steps


def benettin(advance: Callable[[list[torch.Tensor]], None],
             blocks: list[torch.Tensor],   # (..., k, Q_b) orthonormal basis, modified IN PLACE
             n_steps: int,
             dt: float = 1.0,
             cadence: int = 100,
             n_blocks: int = 20,
             warmup: int = 0,
             progress: Callable[[int], None] | None = None,
             ) -> Spectrum:
    """ Benettin/QR Lyapunov spectrum of a map, for a whole batch at once.

    `advance(blocks)` steps the state and the tangent basis together by one step, in place. It
    is one callable rather than a (step, tangent_step) pair on purpose: the tangent is the
    linearization *at the current state*, so the two must share one evaluation of the model's
    nonlinearity. Two separate callables would let them desync, and the resulting exponents
    would be wrong in a way nothing downstream could detect.

    `cadence` is how often the basis is reorthonormalized. The usual worry is that columns
    collapse onto the dominant direction between renormalizations and the subdominant exponents
    are lost to roundoff; the safe interval is about -log(eps_machine) / ((lam_1 - lam_k) * dt).
    For this model the measured gap is ~1.5e-3 per step, giving ~24000 steps, so the default of
    100 is three orders of magnitude inside the limit and cadences of 1, 20 and 200 agree to
    five decimals. It is a parameter because that gap is a property of the attractor, not of
    the algorithm, and a different parameter regime could make it much larger.

    Error bars come from `n_blocks` block means rather than from a split in half. Split-half
    answers "is the running estimate still drifting", which is a bias question; the variance of
    a time average over a serially correlated signal needs blocks long compared to the
    correlation time. The drift check is still reported, separately, as `drift`.

    `warmup` steps are run, with renormalization, before any logs are accumulated. A random
    initial basis is not aligned with the dominant invariant subspace, and the misalignment is
    worked off geometrically at rate (lam_i - lam_{i+1}); logging it biases the estimate low by
    however much of that transient lands in the first blocks. Measured on a linear map with an
    exactly known spectrum, warmup=0 is off by ~8e-3 while a warmup comparable to the alignment
    time is exact to machine precision. It is separate from the state equilibration the caller
    does first: that settles the trajectory, this settles the basis.

    The caller is expected to have equilibrated the state already and to have applied one
    tangent step to the initial basis -- see `init_tangent`. Propagating tangents through
    equilibration costs several times a plain state step and buys nothing. """
    if n_steps < n_blocks:
        raise ValueError(f"n_steps={n_steps} must be at least n_blocks={n_blocks}")
    if cadence < 1:
        raise ValueError(f"cadence must be >= 1, got {cadence}")

    lead, k = blocks[0].shape[:-2], blocks[0].shape[-2]
    dev, dtype = blocks[0].device, blocks[0].dtype
    per_block = torch.zeros(n_blocks, *lead, k, device=dev, dtype=dtype)
    counts    = torch.zeros(n_blocks, device=dev, dtype=dtype)

    # Block boundaries in steps, so every step lands in exactly one block even when n_steps is
    # not a multiple of n_blocks. edges[j+1] is where block j closes.
    edges = [round(j * n_steps / n_blocks) for j in range(n_blocks + 1)]

    since = 0
    for t in range(warmup):
        advance(blocks)
        since += 1
        if since == cadence:
            orthonormalize(blocks)          # discarded: this is the alignment transient
            since = 0
    if warmup:
        orthonormalize(blocks)              # start the first block from an orthonormal basis

    acc = torch.zeros(*lead, k, device=dev, dtype=dtype)
    since, blk = 0, 0
    for t in range(n_steps):
        advance(blocks)
        since += 1
        # Renormalize on cadence, and always at a block edge: a block's logs must cover exactly
        # the steps attributed to it, so an interval may not straddle the boundary.
        if since == cadence or t + 1 == edges[blk + 1]:
            acc += orthonormalize(blocks)
            since = 0
        if t + 1 == edges[blk + 1]:
            per_block[blk] = acc
            counts[blk] = edges[blk + 1] - edges[blk]
            acc = torch.zeros_like(acc)
            blk += 1
        if progress is not None:
            progress(t)

    # per-block exponent estimates, then mean and standard error over blocks
    rates = per_block / counts.reshape(-1, *([1] * (len(lead) + 1))) / dt   # (n_blocks, ..., k)

    # Sort descending. QR orders the exponents only asymptotically; inside a degenerate cluster
    # a finite window routinely returns them out of order, and the exponents are *defined* as an
    # ordered sequence. Sorting the whole (n_blocks, B, k) stack by the final estimate keeps the
    # block statistics attached to the exponent they belong to. Without this a positive exponent
    # can land in column 1 and be missed by a classifier that reads column 0.
    order = rates.mean(dim=0).argsort(dim=-1, descending=True)   # (..., k)
    rates = torch.gather(rates, -1, order.unsqueeze(0).expand_as(rates))

    lam   = rates.mean(dim=0)
    se    = rates.std(dim=0) / (n_blocks ** 0.5)

    h = n_blocks // 2
    drift = rates[h:].mean(dim=0) - rates[:h].mean(dim=0)

    gaps = (lam[..., :-1] - lam[..., 1:]).abs()
    degenerate = torch.zeros_like(lam, dtype=torch.bool)
    degenerate[..., :-1] = gaps < se[..., :-1]
    degenerate[..., 1:] |= gaps < se[..., 1:]

    return Spectrum(lam=lam, se=se, partial=lam.cumsum(dim=-1), drift=drift,
                    degenerate=degenerate, valid=(lam.abs() * dt) < 0.1,
                    n_blocks=n_blocks, T=n_steps * dt)


# Classification
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FIXED, LOCKED, CIRCLE, TORUS, CHAOS = 0, 1, 2, 3, 4
CLASS_NAMES = ["fixed point", "locked cycle", "invariant circle", "2-torus", "chaos"]


def classify_spectrum(lam: torch.Tensor,            # (..., k) exponents per unit time
                      tol: float,                   # floor on the half-width of the zero band
                      moving: torch.Tensor,         # (...) bool: the state varies in time
                      se: torch.Tensor | None = None,   # (..., k) standard error of lam
                      n_sigma: float = 3.0,         # how many SE a nonzero exponent must clear
                      ) -> torch.Tensor:            # (...) long, one of the class codes above
    """ Attractor class from the leading exponents, for a MAP.

    The flow table does not apply and getting this wrong is the easiest mistake here. In a flow
    a limit cycle has one zero exponent, the phase direction. In a map an attracting periodic
    orbit is a fixed point of the P-th iterate and every exponent is strictly negative; a zero
    exponent means an invariant circle, i.e. quasiperiodicity. So lam_1 < 0 does not separate a
    dead cell from an oscillating one, and `moving` -- in practice a variance or
    coefficient-of-variation test from a cheaper screening pass -- has to supply that bit.

    Needs only two exponents. lam_1 decides chaos and the fixed/cycle split; lam_2 separates an
    invariant circle (lam_1 ~ 0 > lam_2) from a 2-torus (both ~ 0) and catches hyperchaos. A
    larger k is not read here.

    The zero band has two half-widths and takes the larger, per cell:

      `tol` is a floor, covering anything that biases a genuinely marginal exponent by a fixed
      amount -- for an age-structured map the lattice pins the phase direction and gives it a
      small negative bias of order dt. It is a property of the model and the discretization.

      `n_sigma * se` is the per-cell statistical half-width. This is the one that actually
      binds: a finite window leaves each exponent with its own error bar, and cells differ by
      an order of magnitude in how well converged they are. Classifying against a scalar
      threshold alone counts every cell whose noise happens to land positive as chaotic. On a
      28561-cell sweep that was the difference between 138 "chaotic" cells and 3.

    Pass `se` from the Benettin block statistics. Omitting it falls back to the scalar `tol`,
    which is only defensible if the windows are long enough that se << tol everywhere, and
    that is worth checking rather than assuming. """
    lam1 = lam[..., 0]
    band = torch.full_like(lam1, tol) if se is None else torch.clamp(n_sigma * se[..., 0], min=tol)
    out = torch.full(lam1.shape, FIXED, dtype=torch.long, device=lam.device)

    near0_1 = lam1.abs() <= band
    out[~near0_1 & (lam1 < 0) & moving] = LOCKED
    out[near0_1] = CIRCLE

    if lam.shape[-1] > 1:
        # two independent neutral directions, not one conjugate pair sitting near zero
        band2 = torch.full_like(lam1, tol) if se is None else torch.clamp(n_sigma * se[..., 1], min=tol)
        out[near0_1 & (lam[..., 1].abs() <= band2)] = TORUS

    out[lam1 > band] = CHAOS
    return out


def kaplan_yorke(lam: torch.Tensor,   # (..., k) exponents per unit time, descending
                 ) -> torch.Tensor:   # (...) Lyapunov dimension, nan where k is too small
    """ Lyapunov (Kaplan-Yorke) dimension: j + (sum_{i<=j} lam_i) / |lam_{j+1}|, with j the
    largest index whose partial sum is still non-negative.

    Returns nan when the partial sum never turns negative within the k supplied, which is the
    honest answer -- the dimension is then only bounded below by k, and reporting k itself
    would look like a measurement. With k = 2 this is informative only for D < 2, so a run that
    finds chaos should recompute the few chaotic cells with a larger k. """
    csum = lam.cumsum(dim=-1)
    j = (csum >= 0).sum(dim=-1)                     # number of leading non-negative partial sums
    k = lam.shape[-1]

    ok = (j >= 1) & (j < k)
    jj = j.clamp(1, k - 1)
    num = torch.gather(csum, -1, (jj - 1).unsqueeze(-1)).squeeze(-1)
    den = torch.gather(lam,  -1, jj.unsqueeze(-1)).squeeze(-1).abs()

    out = torch.full_like(lam[..., 0], float('nan'))
    out[ok] = (jj + num / den.clamp_min(1e-300))[ok]
    out[j == 0] = 0.0
    return out
