""" Stability of the mean field: its fixed points, their linear stability, their bifurcations
along a 1-D cut of parameter space, and the attractor a run reaches.

Each section answers one question:

  Fixed points            where are they?  Newton (any M) and scalar bracketing (M = 1), generic
                          over a residual G(m) = m - F(m); F itself is equations.fp_overlap_map.
  Linear stability        is a fixed point stable, and how does it lose stability?
                          leading_modes (ARPACK) for the leading eigenvalues and their
                          frequencies; leading_growth (power iteration) for the growth rate where
                          the leading modes are too degenerate for ARPACK; jacobian_eigvals (dense)
                          as the oracle the other two are tested against.
  Bifurcations on a cut   where along a 1-D cut do the fixed points change, and how? scan_cut
                          for the fixed points and growth rates on a grid, find_bifurcations
                          for the saddle-nodes (bisected on the count) and the crossings of the
                          unit circle -- Hopf, period-doubling, branch point -- (bisected with
                          ARPACK).
  Attractors              what does a run actually reach? classify_attractor and
                          compare_starts, on trajectories from batched runs. Not a bifurcation
                          question: a cycle growing into bursts, or one coexisting with a
                          stable fixed point, is invisible to the fixed points.

Devices (measured 2026-11-19, E-I state D = 8924, RTX 4070 SUPER): batched work belongs on the
GPU, single systems on the CPU.
  - Newton over a batch: 81 systems x 36 starts took 78 s on the CPU; 27 x 36 took 0.4 s on the
    GPU. Its memory grows with the number of Newton trajectories (~0.7 MB each at Q ~ 2100),
    so fp_sweep runs it in chunks that halve on out-of-memory (run_chunked).
  - Power iteration over a batch: 0.43 ms per step for 81 systems on the GPU, 2.2 ms on the CPU.
  - ARPACK is sequential in single-system JVPs, 0.22 ms on the CPU against 0.44 ms on the GPU.
Every function runs on the device of the tensors (or the model) it is given; the scan helpers
take the device as an argument. """

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

import rdme.analysis as an
import rdme.helpers as hlp

# Memory
# ~~~~~~

def run_chunked(fn: Callable[[slice], torch.Tensor], n: int,
                chunk: int | None = None) -> tuple[torch.Tensor, int]:
    """ fn over the rows [0, n) in chunks, concatenated along the first axis, and the chunk size
    that worked. Starts at `chunk` (default: all n at once) and halves it on
    torch.OutOfMemoryError, emptying the CUDA cache before retrying the failed slice; a chunk
    of one row that still does not fit re-raises. """
    chunk = n if chunk is None else max(1, min(chunk, n))
    out, i = [], 0
    while i < n:
        sl = slice(i, min(i + chunk, n))
        try:
            out.append(fn(sl))
        except torch.OutOfMemoryError:
            if chunk == 1:
                raise
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            continue
        i = sl.stop
    return torch.cat(out), chunk


# Fixed points
# ~~~~~~~~~~~~
#
# Model-agnostic: everything here takes a residual G(m) = m - F(m), (M,) -> (M,), and never asks
# what produced it. The overlaps live in [0, 1], and the solvers clamp to it.

def fp_guess_lattice(M: int,
                     n_per_axis: int = 5,
                     lo: float = 0.0,
                     hi: float = 1.0,
                     device: str | torch.device = 'cpu',
                     dtype: torch.dtype = torch.float64,
                     ) -> torch.Tensor:    # (n_per_axis**M, M) starting points
    """ A regular lattice of Newton starting points covering the overlap box [lo, hi]^M.

    The endpoints are shrunk inwards by half a cell: m = 0 and m = 1 are the edges of the
    physical range, where the hazard is most nearly degenerate, and a Newton step from
    exactly there tends to be thrown straight back out by the clamp. n_per_axis**M grows
    fast, but M is one or two for every model here. """
    step = (hi - lo) / (2 * n_per_axis)
    axis = torch.linspace(lo + step, hi - step, n_per_axis, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(*([axis] * M), indexing='ij'), dim=-1).reshape(-1, M)


def overlap_ceiling(mf) -> float:
    """ The largest overlap any population of `mf` can sustain: dt/t_abs, the dead-time
    ceiling in overlaps, or 1 with no dead time. The box Newton starts should cover. """
    t = float(mf.t_abs.min())
    return 1.0 if t == 0.0 else min(1.0, mf.dt / t)


def default_guesses(mf, n_per_axis: int | None = None) -> torch.Tensor:
    """ The Newton starts RDMNetwork.fixed_points uses for one system with M > 1: a lattice
    over the physical box [0, overlap_ceiling]^M (12 per axis for M = 2), plus the
    reduced_guesses where they apply. """
    n = n_per_axis if n_per_axis is not None else (12 if mf.M == 2 else 5)
    lat = fp_guess_lattice(mf.M, n, 0.0, overlap_ceiling(mf), device=mf.device, dtype=mf.dtype)
    return torch.cat([lat, reduced_guesses(mf)])


def reduced_guesses(mf, n_grid: int = 4001) -> torch.Tensor:
    """ For one M = 2 system with w(I→I) = 0: a Newton start inside every sign change of
    G_E(m_E, F_I(m_E)) on a fine m_E grid, (R, 2).

    With no I→I coupling I's input reads E's rate alone, so r_I = F_I(r_E) and the fixed-point
    problem is one-dimensional: these starts reach every fixed point up to the grid's
    resolution, as bracketing does for M = 1. A Newton lattice alone missed the saddle and the
    unstable focus of the w(I→I) = 0 E-I family (2026-11-19). Empty when it does not apply. """
    if mf.M != 2 or float(mf.w[..., 1, 1].abs().max()) != 0.0:
        return torch.empty(0, mf.M, device=mf.device, dtype=mf.dtype)
    mE = torch.linspace(0.0, overlap_ceiling(mf), n_grid, device=mf.device, dtype=mf.dtype)
    z  = torch.stack([mE, torch.zeros_like(mE)], -1)
    mI = (z - mf.fp_residual(z))[:, 1]             # F_I, which reads m_E only
    g  = mf.fp_residual(torch.stack([mE, mI], -1))[:, 0]
    i  = ((g[:-1] * g[1:]) <= 0).nonzero().flatten()
    return torch.stack([0.5 * (mE[i] + mE[i + 1]), 0.5 * (mI[i] + mI[i + 1])], -1)


def newton_step(residual: Callable[..., torch.Tensor],
                m: torch.Tensor,             # (M,) current iterate
                params: tuple = (),          # extra positional args for the residual
                lo: float = 0.0,
                hi: float = 1.0,
                damping: float = 1.0,
                ls_max: int = 0,
                ls_beta: float = 0.5,
                ridge: float = 1e-12,
                ) -> torch.Tensor:        # (M,) next iterate
    """ One damped Newton step on a single point, with optional backtracking.

    The step is clamped back into the box [lo, hi] -- the overlaps are firing rates and
    F maps [0,1]^M into (0,1]^M, so an iterate outside the box carries no information
    and only risks overflowing the hazard.

    Backtracking (ls_max > 0) halves the step while the residual norm fails to improve on
    the undamped one. It is written with torch.where over a *fixed* number of trials rather
    than a while loop, so the whole step stays vmap-safe at the cost of ls_max extra
    residual evaluations; the cost is why it is off by default.

    `ridge` is added to the Jacobian diagonal so that a singular Jacobian yields a large
    but finite step instead of raising -- an exception inside vmap would take down the
    whole sweep, while a large step is caught by the non-finite guard in `newton_iterate`
    and by the clamp. """
    res = (lambda x: residual(x, *params)) if params else residual
    f = res(m)
    jac = torch.func.jacfwd(res)(m)
    eye = torch.eye(m.shape[-1], device=m.device, dtype=m.dtype)
    delta = torch.linalg.solve(jac + ridge * eye, f)

    f_norm = torch.linalg.vector_norm(f)
    best = torch.clamp(m - damping * delta, lo, hi)
    best_norm = torch.linalg.vector_norm(res(best))

    alpha = damping
    for _ in range(ls_max):
        alpha = alpha * ls_beta
        trial = torch.clamp(m - alpha * delta, lo, hi)
        trial_norm = torch.linalg.vector_norm(res(trial))
        # only backtrack while the best step so far is still worse than where we started
        take = (best_norm >= f_norm) & (trial_norm < best_norm)
        best = torch.where(take, trial, best)
        best_norm = torch.where(take, trial_norm, best_norm)

    return best


def newton_iterate(residual: Callable[..., torch.Tensor],
                   m0: torch.Tensor,       # (K, M) batch of initial guesses
                   params: tuple = (),     # extra residual args, each with a leading K axis
                   max_iter: int = 200,
                   tol: float = 1e-10,
                   lo: float = 0.0,
                   hi: float = 1.0,
                   damping: float = 1.0,
                   ls_max: int = 0,
                   ls_beta: float = 0.5,
                   ) -> torch.Tensor:      # (K, M) iterates, converged or not
    """ Run Newton from K starting points at once, without filtering.

    The K trajectories are vmapped, while the stopping test sits outside the vmap: the
    iteration halts once *every* trajectory has stopped moving, which costs a few wasted
    steps on the fast ones and keeps the whole loop free of data-dependent control flow.
    Trajectories that leave the finite range are frozen at their last good iterate and
    filtered out downstream by their residual.

    Every entry of `params` is mapped over its leading axis together with m0's, so K is
    really "number of Newton trajectories" -- a sweep of B parameter sets from K0 starting
    points each flattens into K = B*K0 rows, solved simultaneously. """
    assert m0.dim() == 2, "m0 must be (K, M); use m0.unsqueeze(-1) for scalar problems"
    m = torch.clamp(m0.clone(), lo, hi)

    step = torch.vmap(lambda mi, *pi: newton_step(residual, mi, pi, lo, hi, damping, ls_max, ls_beta))

    for _ in range(max_iter):
        m_next = step(m, *params)
        stuck = ~torch.isfinite(m_next).all(dim=-1, keepdim=True)
        m_next = torch.where(stuck, m, m_next)
        if torch.linalg.vector_norm(m_next - m, dim=-1).max() < tol:
            m = m_next
            break
        m = m_next

    return m


def newton_roots(residual: Callable[..., torch.Tensor],
                 m0: torch.Tensor,       # (K, M) batch of initial guesses
                 params: tuple = (),     # extra residual args, each with a leading K axis
                 max_iter: int = 200,
                 tol: float = 1e-10,
                 res_tol: float | None = None,
                 merge_tol: float = 1e-6,
                 lo: float = 0.0,
                 hi: float = 1.0,
                 damping: float = 1.0,
                 ls_max: int = 0,
                 ls_beta: float = 0.5,
                 ) -> torch.Tensor:      # (R, M) deduplicated roots
    """ Newton from K starting points, keeping only the converged, distinct roots.

    A trajectory counts as a root when |G(m)| < res_tol (defaulting to a loose multiple of
    `tol`: the step size and the residual are different scales, and demanding the residual
    itself reach `tol` discards perfectly good roots on flat branches). Returns an empty
    (0, M) tensor when nothing converged -- bistable sweeps hit that legitimately at some
    parameter values, so it is a return value and not an exception. """
    res_tol = res_tol if res_tol is not None else 1e3 * tol

    m = newton_iterate(residual, m0, params, max_iter, tol, lo, hi, damping, ls_max, ls_beta)
    res = residual_norms(residual, m, params)
    keep = torch.isfinite(res) & (res < res_tol)
    return deduplicate(m[keep], merge_tol)


def residual_norms(residual: Callable[..., torch.Tensor],
                   m: torch.Tensor,        # (K, M) points
                   params: tuple = (),     # extra residual args, each with a leading K axis
                   ) -> torch.Tensor:      # (K,) |G(m)|
    """ Residual norm at each of K points, evaluated in one vmapped pass.

    This is the convergence test a sweep needs: `newton_roots` filters and deduplicates for
    you, but a sweep that keeps its roots laid out per parameter set has to do its own
    filtering, and this is the quantity to filter on. """
    return torch.vmap(lambda mi, *pi: torch.linalg.vector_norm(residual(mi, *pi)))(m, *params)


def bracket_roots(residual: Callable[[torch.Tensor], torch.Tensor],  # single, unparametrized
                  lo: float = 0.0,
                  hi: float = 1.0,
                  n_grid: int = 201,
                  max_iter: int = 60,
                  merge_tol: float = 1e-6,
                  device: str | torch.device = 'cpu',
                  dtype: torch.dtype = torch.float64,
                  ) -> torch.Tensor:     # (R, 1) roots, sorted
    """ All sign changes of a scalar residual on [lo, hi], refined by bisection.

    For M=1 this is the method of choice: the grid scan is exhaustive up to its own
    resolution, so it finds every root of odd multiplicity -- including the unstable middle
    branch of a bistable region, which Newton only reaches from a narrow basin -- and it
    cannot diverge. The brackets are bisected simultaneously, so the cost is one residual
    evaluation on (n_grid,) plus max_iter on (R,).

    `residual` follows the same (M,) -> (M,) convention as the Newton path, with M=1;
    it is evaluated on a whole grid at once via vmap. """
    grid = torch.linspace(lo, hi, n_grid, device=device, dtype=dtype).unsqueeze(-1)  # (n_grid, 1)
    vals = torch.vmap(residual)(grid).squeeze(-1)                        # (n_grid,)
    grid = grid.squeeze(-1)

    # exact hits on the grid are roots already, and would otherwise be missed by the
    # strict sign-change test below
    exact = grid[vals == 0]

    a, b = grid[:-1], grid[1:]
    fa, fb = vals[:-1], vals[1:]
    brack = (fa * fb) < 0
    a, b, fa = a[brack], b[brack], fa[brack]

    # No bracket to bisect is a real case, not an empty edge case: a residual that lands
    # exactly on a grid point makes both of its neighbouring products zero rather than
    # negative, so a lone exact root leaves nothing to refine. It happens whenever the
    # hazard stops depending on the field -- beta = 0 gives Phi = 1/2 everywhere, hence
    # G(m) = m - 1/2, whose root is on the grid for any odd n_grid. Bisecting an empty
    # bracket sends vmap a zero-size batch, which raises rather than returning nothing.
    if a.numel() == 0:
        return deduplicate(exact.unsqueeze(-1), merge_tol) if exact.numel() else \
               torch.empty(0, 1, device=device, dtype=grid.dtype)

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = torch.vmap(residual)(mid.unsqueeze(-1)).squeeze(-1)
        same = (fm * fa) > 0            # root lies in the right half
        a = torch.where(same, mid, a)
        fa = torch.where(same, fm, fa)
        b = torch.where(same, b, mid)

    roots = torch.cat([exact, 0.5 * (a + b)]).unsqueeze(-1)
    return deduplicate(roots, merge_tol)


def deduplicate(roots: torch.Tensor,       # (R, M)
                merge_tol: float = 1e-6,
                ) -> torch.Tensor:         # (R', M) representatives, sorted
    """ Collapse roots that are within merge_tol of each other into one.

    Greedy single-pass clustering over points sorted by their first coordinate: each root
    joins the first kept representative it is close to, so a chain of points spaced just
    under merge_tol collapses to its head rather than fragmenting. R is the number of
    starting points, which is small (tens), so the quadratic scan never matters. """
    if roots.numel() == 0:
        return roots
    order = torch.argsort(roots[:, 0])
    kept: list[torch.Tensor] = []
    for r in roots[order]:
        if all(torch.linalg.vector_norm(r - k) > merge_tol for k in kept):
            kept.append(r)
    return torch.stack(kept)


# Linear stability
# ~~~~~~~~~~~~~~~~
#
# All three act on the linearization of the one-step map on the FULL state (age distributions and
# synaptic terms, RDMNetwork.state_map), not of the overlap map F, whose Jacobian says nothing
# about age-resolved perturbations. A growth rate mu = ln|lambda|/dt < 0 is stable.
#
# The spectrum, measured on the E-I Fig. 5 cut (2026-11-19): a few isolated leading modes, then a
# band of ~Q near-degenerate eigenvalues (the roots of the discrete renewal polynomial). ARPACK
# converges on the isolated modes at k <= 4 (k = 6 reaches into the band: 4.3k matvecs against
# 240k). Deep on the stable side the leading modes themselves become degenerate in modulus (two
# pairs within ~1e-4 /ms, more just below) and ARPACK fails at any k, ncv or tol; the growth rate
# still converges there under power iteration, which is what leading_growth is for. The real fix
# for such spectra is root-finding on the characteristic equation, mode by mode; not done.
#
# Tried and dropped: ARPACK on the propagator J^p with Rayleigh-Ritz recovery, to widen the gap to
# the band -- slower at every p tried (p = 20 not converged after 9 min against 1.1 s at p = 1).

def jacobian_eigvals(state_map: Callable[[torch.Tensor], torch.Tensor],
                     x: torch.Tensor,     # (D,) state vector at the fixed point
                     ) -> torch.Tensor:   # (D,) complex eigenvalues
    """ Every eigenvalue of the linearization of a discrete-time map at a point: the dense
    oracle. O(D^3) -- 113 s at dt = 0.2 and 613 s at dt = 0.1 for the E-I state -- so for tests,
    spot checks and single-population states, not for scans. The age distributions are normalized, so the map is confined
    to an affine subspace and the spectrum carries eigenvalues of the directions that leave it;
    they are harmless but should not be mistaken for physical modes. """
    jac = torch.func.jacfwd(state_map)(x)
    return torch.linalg.eigvals(jac)


def leading_modes(state_map: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                  dt: float, k: int = 4, ncv: int = 40, tol: float = 1e-8,
                  maxiter: int = 20000, v0: torch.Tensor | None = None) -> dict:
    """ The k leading modes of the linearization of the one-step map `state_map` at `x` (D,),
    for ONE system: ARPACK on Jacobian-vector products, the eigenvalues of largest modulus
    only. `tol` is ARPACK's relative accuracy on the eigenvalues; its default (machine
    precision) is what made it stall (322k matvecs against 7k at 1e-6, mu unchanged to 7
    digits). If it does not converge on k, retries with k = 2 (one leading pair, or two real
    modes) before giving up with ArpackNoConvergence -- which says the modes were too crowded
    to resolve, NOT that the point is stable: leading_growth gets the rate there.

    `v0` (D,) starts the iteration, e.g. the `v` of a neighbouring point in a scan. CPU: the
    JVPs are sequential and single-system, 0.22 ms each against 0.44 ms on the GPU.

    Returns
      mu        (k,) growth rates ln|lambda| / dt, in 1/ms, decreasing; stable iff all < 0
      omega     (k,) angular frequencies arg(lambda) / dt, in rad/ms; a conjugate pair carries
                both signs, a real mode 0 (or pi/dt if negative)
      v         (D,) real part of the leading eigenvector, a warm start for the next solve
      n_matvec  Jacobian-vector products used, over every attempt """
    _, jvp = torch.func.linearize(state_map, x)
    D, n = x.numel(), [0]

    def mv(v):
        n[0] += 1
        return jvp(torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64))
                   .to(x)).cpu().numpy()

    A  = LinearOperator((D, D), matvec=mv, dtype=np.float64)
    v0 = None if v0 is None else v0.detach().cpu().double().numpy()
    for kk in dict.fromkeys((k, 2)):
        try:
            w, V = eigs(A, k=kk, ncv=min(ncv, D - 1), which="LM", tol=tol, maxiter=maxiter,
                        v0=v0)
        except ArpackNoConvergence:
            continue
        o = np.argsort(-np.abs(w))
        w, V = torch.from_numpy(w[o]), torch.from_numpy(V[:, o])
        return dict(mu=w.abs().log() / dt, omega=w.angle() / dt, v=V[:, 0].real.clone(),
                    n_matvec=n[0])
    raise ArpackNoConvergence(f"no convergence at k = {k} or 2 ({n[0]} matvecs)", [], [])


def leading_growth(state_map: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                   dt: float, t_ms: float = 1000.0, seed: int = 0) -> torch.Tensor:
    """ Leading growth rate mu (1/ms) of the linearization of `state_map` at `x`, by power
    iteration, for one system, x (D,) -> (), or for a batch, x (*b, D) -> (*b,), each row
    iterated on its own (state_map must then be a batched model's).

    A renormalized tangent vector is pushed for t_ms of burn-in, then for t_ms more, and mu is
    the least-squares slope of its cumulative log norm against time -- the slope, not
    (end - start)/T: a single vector's norm oscillates with a leading complex pair, and the
    endpoints would carry that phase. The rate converges where the leading modes are
    degenerate and ARPACK cannot; no frequency comes out of it.

    Measured on the E-I Fig. 5 cut at dt = 0.1, t_ms = 1000 (2026-11-19): within 2e-5 /ms of
    ARPACK where ARPACK converges, within 5e-4 of the dense spectrum at a degenerate point,
    closing slowly with t_ms. Batches belong on the GPU: 0.43 ms per step for 81 systems there,
    2.2 ms on the CPU. """
    _, jvp = torch.func.linearize(state_map, x)
    g = torch.Generator().manual_seed(seed)
    # one start vector for every row, so a batch's rows match their systems run alone
    v = torch.randn(x.shape[-1], dtype=x.dtype, generator=g).to(x).expand_as(x)
    v = v / v.norm(dim=-1, keepdim=True)
    n = hlp.n_steps(t_ms, dt, name="t_ms")
    logn = torch.zeros((n,) + x.shape[:-1], dtype=torch.float64, device=x.device)
    acc = torch.zeros(x.shape[:-1], dtype=torch.float64, device=x.device)
    for t in range(2 * n):
        v = jvp(v)
        s = v.norm(dim=-1, keepdim=True)
        v = v / s
        if t >= n:
            acc = acc + s.squeeze(-1).log()
            logn[t - n] = acc
    tt = torch.arange(n, dtype=torch.float64, device=x.device) * dt
    tt = (tt - tt.mean()).reshape((n,) + (1,) * (x.dim() - 1))
    return ((tt * (logn - logn.mean(0))).sum(0) / (tt * tt).sum(0)).cpu()


# Bifurcations along a cut
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# A cut is a 1-D family of systems, built by `build(values, device)`: a batched RDMNetwork, one
# system per value, for scan_cut on the GPU; and one system per value on the CPU for the
# bisections.

def scan_cut(build: Callable[[torch.Tensor, str], object], values: Sequence[float],
             device: str = "cuda" if torch.cuda.is_available() else "cpu",
             n_per_axis: int = 12, t_ms: float = 1000.0,
             chunk: int | None = None) -> list[dict]:
    """ Every fixed point and its leading growth rate at each value of a cut, as a list of
    {"value", "m" (R, M), "r" (R, M) Hz, "mu" (R,)}, roots sorted by the first population.

    The fixed points come from a batched Newton (fp_sweep, chunked against out-of-memory) from
    a lattice over the physical box, plus the reduced starts of an M = 2, w(I→I) = 0 system;
    the growth rates from a batched leading_growth, one round per root index. """
    vals = torch.as_tensor(values, dtype=torch.float64)
    mf = build(vals, device)
    roots, ok = mf.fp_sweep(n_per_axis=n_per_axis, chunk=chunk)        # (B, K, M), (B, K)
    roots, ok = roots.cpu(), ok.cpu()
    per = [deduplicate(roots[b][ok[b]]) for b in range(mf.B)]
    if mf.M == 2 and float(mf.w[..., 1, 1].abs().max()) == 0.0:
        # the reduced starts are per system (their number varies), so on the CPU, one by one
        for b in range(mf.B):
            sys = _single(build, float(vals[b]))
            g = reduced_guesses(sys)
            if len(g):
                per[b] = deduplicate(torch.cat([per[b], sys.fixed_points(guesses=g)]))
    n_max = max((len(p) for p in per), default=0)
    mu = [torch.full((len(p),), float("nan"), dtype=torch.float64) for p in per]
    for j in range(n_max):
        # round j: each system's j-th root, or its first as a placeholder whose result is dropped
        m = torch.stack([p[j] if len(p) > j else (p[0] if len(p) else
                         torch.zeros(mf.M, dtype=torch.float64)) for p in per]).to(device)
        g = leading_growth(mf.state_map, mf.flatten_state(*mf.fp_state(m)), mf.dt, t_ms)
        for b, p in enumerate(per):
            if len(p) > j:
                mu[b][j] = g[b]
    return [dict(value=float(vals[b]), m=per[b], r=per[b] / mf.dt * 1e3, mu=mu[b])
            for b in range(mf.B)]


def branch_scan(mf,                                   # a batched RDMNetwork, M = 1
                axis: torch.Tensor,                   # (B,) the swept parameter
                idx: torch.Tensor | None = None,      # which batch elements, default all
                n_grid: int = 401,
                n_lead: int = 4) -> dict[str, torch.Tensor]:
    """ Every fixed point of every selected batch element, with its leading eigenvalues: a
    branch diagram for an M = 1 sweep (plots.sweep_summary's `fp`).

    Returns flat tensors, one entry per (element, root) pair rather than per element: the
    number of roots varies along the axis, which is the whole point of a branch diagram, so
    the axis value is carried alongside as "x" rather than implied by position. "lead" holds
    the n_lead leading eigenvalues of the one-step map (leading_modes) and "rho" the largest
    modulus; where ARPACK cannot separate the leading modes, "lead" is nan and "rho" comes from
    leading_growth.

    M = 1 only. The bracketing scan over [0, 1] is then exhaustive up to its resolution, so it
    picks up unstable branches that Newton would only reach from a narrow basin -- and it is
    the unstable branch that says where the attractor went. """
    if idx is None:
        idx = torch.arange(mf.B)

    x, m_star, rho, lead = [], [], [], []
    for b in idx:
        sys = mf.select(int(b))
        for r in sys.fixed_points(method='bracket', n_grid=n_grid):
            try:
                o = sys.leading_modes(r, k=n_lead)
                ev = torch.polar(torch.exp(o["mu"] * sys.dt), o["omega"] * sys.dt)
                rh = ev.abs().max()
            except ArpackNoConvergence:
                ev = torch.full((n_lead,), complex("nan"), dtype=torch.complex128)
                rh = torch.exp(sys.leading_growth(r) * sys.dt)
            x.append(axis[b])
            m_star.append(r[0])
            rho.append(rh)
            lead.append(ev)

    return {"x": torch.stack(x), "m_star": torch.stack(m_star),
            "rho": torch.stack(rho), "lead": torch.stack(lead)}


def _single(build, value: float):
    """ The single system at `value` of a cut, on the CPU. """
    return build(torch.tensor([value], dtype=torch.float64), "cpu").select(0)


def _fixed_points_at(build, value: float, extra: torch.Tensor | None = None) -> torch.Tensor:
    """ Every fixed point of the single system at `value` (CPU Newton from its default starts,
    plus `extra`). """
    sys = _single(build, value)
    g = default_guesses(sys) if extra is None else torch.cat([default_guesses(sys),
                                                              extra.cpu().reshape(-1, sys.M)])
    return sys.fixed_points(guesses=g)


def _mode_at(build, value: float, m_near: torch.Tensor) -> dict:
    """ The fixed point of the single system at `value` nearest m_near, with its leading mode
    from ARPACK (CPU). """
    sys = _single(build, value)
    m = min(_fixed_points_at(build, value, m_near), key=lambda r: float((r - m_near.cpu()).norm()))
    o = leading_modes(sys.state_map, sys.flatten_state(*sys.fp_state(m)), sys.dt, k=2)
    return dict(value=value, m=m, r=m / sys.dt * 1e3, mu=float(o["mu"][0]),
                omega=float(o["omega"][0]), dt=sys.dt)


def _crossing_kind(omega: float, dt: float) -> str:
    """ The bifurcation a fixed point's critical eigenvalue lambda = exp((mu + i omega) dt)
    marks as it crosses the unit circle with the fixed-point count unchanged. """
    if abs(omega) * dt < 1e-9:
        return "branch-point"         # real +1, no count change: transcritical or pitchfork
    if abs(abs(omega) * dt - math.pi) < 1e-9:
        return "period-doubling"      # real -1
    return "hopf"                     # complex pair


BIFURCATIONS = ("saddle-node", "hopf", "period-doubling", "branch-point")


def find_bifurcations(scan: list[dict], build, tol: float = 0.01) -> list[dict]:
    """ The local bifurcations of the fixed points along a scan_cut result, each bisected to
    `tol` in the cut's value. One of BIFURCATIONS:

      saddle-node      the fixed-point count changes between neighbours (two fixed points
                       merge and vanish); bisected on the count. Its real +1 eigenvalue is not
                       reported again as a crossing.
      hopf             a complex pair crosses the unit circle: a cycle is born or dies.
      period-doubling  a real eigenvalue crosses -1.
      branch-point     a real eigenvalue crosses +1 with the count unchanged (transcritical,
                       or a pitchfork the grid did not resolve).

    The last three are crossings: a fixed point's growth rate changes sign between neighbours
    with equal counts (the fixed point followed by nearest neighbour), bisected with ARPACK on
    single CPU systems, where the critical mode is isolated; the critical eigenvalue gives the
    kind and the frequency.

    Each is a dict with "kind", "value", and the bracket "lo"/"hi" from the scan; a saddle-node
    adds "counts" (before, after); a crossing adds "side" (where the fixed point is unstable,
    "above"/"below" the value), "freq_hz", and "r" (Hz) and "mu" (1/ms) at the bisected
    value. """
    out = []
    for a, b in zip(scan[:-1], scan[1:]):
        if len(a["m"]) != len(b["m"]):
            lo, hi = a["value"], b["value"]
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                n = len(_fixed_points_at(build, mid))
                lo, hi = (mid, hi) if n == len(a["m"]) else (lo, mid)
            out.append(dict(kind="saddle-node", value=0.5 * (lo + hi), lo=a["value"],
                            hi=b["value"], counts=(len(a["m"]), len(b["m"]))))
            continue
        for i, m in enumerate(a["m"]):
            j = int(((b["m"] - m).norm(dim=-1)).argmin())
            if (a["mu"][i] < 0) == (b["mu"][j] < 0):
                continue
            flo, fhi = _mode_at(build, a["value"], m), _mode_at(build, b["value"], b["m"][j])
            while fhi["value"] - flo["value"] > tol:
                f = _mode_at(build, 0.5 * (flo["value"] + fhi["value"]), flo["m"])
                if (f["mu"] < 0) == (flo["mu"] < 0):
                    flo = f
                else:
                    fhi = f
            crit = flo if abs(flo["mu"]) < abs(fhi["mu"]) else fhi
            out.append(dict(kind=_crossing_kind(crit["omega"], crit["dt"]),
                            value=0.5 * (flo["value"] + fhi["value"]),
                            lo=a["value"], hi=b["value"],
                            side="above" if fhi["mu"] > 0 else "below",
                            freq_hz=abs(crit["omega"]) / (2 * math.pi) * 1e3,
                            r=crit["r"], mu=crit["mu"]))
    return out


# Attractors
# ~~~~~~~~~~
#
# What a run reaches, read off its recorded rates. For batched runs from two starts (silent, and
# near a fixed point): linear stability says nothing about which attractor a start lands on, nor
# about a cycle coexisting with a stable fixed point -- the reason the first E-I presets failed.

ATTRACTORS = ("async", "cycle", "burst", "settling")


def classify_attractor(r: torch.Tensor, ceiling: float, pop: int = 0,
                       amp_tol: float = 0.05, ratio_tol: float = 0.05) -> list[str]:
    """ One of ATTRACTORS per system, from recorded rates r (*b, T, M) in Hz:

      async     the peak-to-peak of population `pop` is below amp_tol of its mean: settled.
      burst     oscillating, and some population peaks above `ceiling` (e.g. 1/t_abs).
      cycle     oscillating below the ceiling, the last and first cycles' amplitudes within
                ratio_tol (analysis.cycle_amplitude_ratio).
      settling  oscillating below the ceiling, the amplitude still moving: ringing onto a fixed
                point or growing towards a cycle -- record longer.

    Returns a flat list over the leading axes, in their row-major order. """
    flat = r.reshape((-1,) + r.shape[-2:])
    out = []
    for x in flat:
        y = x[:, pop]
        if float(y.max() - y.min()) < amp_tol * float(y.mean().abs().clamp_min(1e-12)):
            out.append("async")
        elif float(x.max()) > ceiling:
            out.append("burst")
        else:
            ratio = an.cycle_amplitude_ratio(y)
            out.append("cycle" if abs(ratio - 1) <= ratio_tol else "settling")
    return out


def compare_starts(r_a: torch.Tensor, r_b: torch.Tensor, ceiling: float,
                   rel_tol: float = 0.05, **kw) -> torch.Tensor:
    """ True per system where two runs of it, r_a and r_b (*b, T, M) Hz from different starts,
    reached different attractors: different classify_attractor classes, or mean rates of some
    population apart by more than rel_tol. A sign of bistability; only as good as the two
    starts. """
    ca = classify_attractor(r_a, ceiling, **kw)
    cb = classify_attractor(r_b, ceiling, **kw)
    ma, mb = r_a.mean(-2), r_b.mean(-2)                                   # (*b, M)
    far = ((ma - mb).abs() > rel_tol * ma.abs() + 1e-2).any(-1).reshape(-1)
    same = torch.tensor([x == y for x, y in zip(ca, cb)])
    return (far | ~same).reshape(r_a.shape[:-2])
