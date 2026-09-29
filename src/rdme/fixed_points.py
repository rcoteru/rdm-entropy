from __future__ import annotations

from typing import Callable

import torch

""" Model-agnostic machinery for locating and classifying fixed points.

Nothing here knows about the RDM model: every entry point takes a residual
callable G and works on whatever vector space G is defined on. The model-side
half -- how to build G from the mean-field equations, and how to lift a root
back to a full (p, S) state -- lives in rdme.mean_field.

Two solvers are provided, mirroring App. "Root finding": `bracket_roots` for
scalar problems (M=1), where a sign scan over an interval is exhaustive up to
the grid resolution, and `newton_roots` for the general case (M>1), where roots
are collected from many starting points. Both return deduplicated roots, and
`jacobian_eigvals` / `spectral_radius` classify them from a state-space map.

Residual convention: G(m) = m - F(m), with G taking and returning a (M,) tensor
(a length-1 vector for M=1, not a bare scalar). It may take further positional
arguments, which the Newton path carries alongside m with a matching leading
axis: that is how a whole parameter sweep is solved in one call, with one Newton
trajectory per (parameter set, starting point) pair. Differentiation and root
finding are always with respect to the first argument only.
"""

# Root finding: Newton
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


# Root finding: scalar bracketing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bracket_roots(residual: Callable[[torch.Tensor], torch.Tensor],  # single, unparametrized
                  lo: float = 0.0,
                  hi: float = 1.0,
                  n_grid: int = 201,
                  max_iter: int = 60,
                  merge_tol: float = 1e-6,
                  device: str | torch.device = 'cpu',
                  ) -> torch.Tensor:     # (R, 1) roots, sorted
    """ All sign changes of a scalar residual on [lo, hi], refined by bisection.

    For M=1 this is the method of choice: the grid scan is exhaustive up to its own
    resolution, so it finds every root of odd multiplicity -- including the unstable middle
    branch of a bistable region, which Newton only reaches from a narrow basin -- and it
    cannot diverge. The brackets are bisected simultaneously, so the cost is one residual
    evaluation on (n_grid,) plus max_iter on (R,).

    `residual` follows the same (M,) -> (M,) convention as the Newton path, with M=1;
    it is evaluated on a whole grid at once via vmap. """
    grid = torch.linspace(lo, hi, n_grid, device=device).unsqueeze(-1)   # (n_grid, 1)
    vals = torch.vmap(residual)(grid).squeeze(-1)                        # (n_grid,)
    grid = grid.squeeze(-1)

    # exact hits on the grid are roots already, and would otherwise be missed by the
    # strict sign-change test below
    exact = grid[vals == 0]

    a, b = grid[:-1], grid[1:]
    fa, fb = vals[:-1], vals[1:]
    brack = (fa * fb) < 0
    a, b, fa = a[brack], b[brack], fa[brack]

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = torch.vmap(residual)(mid.unsqueeze(-1)).squeeze(-1)
        same = (fm * fa) > 0            # root lies in the right half
        a = torch.where(same, mid, a)
        fa = torch.where(same, fm, fa)
        b = torch.where(same, b, mid)

    roots = torch.cat([exact, 0.5 * (a + b)]).unsqueeze(-1)
    return deduplicate(roots, merge_tol)


# Stability
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def jacobian_eigvals(state_map: Callable[[torch.Tensor], torch.Tensor],
                     x: torch.Tensor,     # (D,) state vector at the fixed point
                     ) -> torch.Tensor:   # (D,) complex eigenvalues
    """ Eigenvalues of the linearization of a discrete-time map at a point.

    `state_map` is the one-step update on the *full* state -- for the mean field, the age
    distributions and synaptic terms of every population stacked into one vector -- not the
    overlap map F, whose Jacobian says nothing about the age-resolved perturbations. The
    fixed point is linearly stable iff every eigenvalue lies strictly inside the unit
    circle. Note that the age distributions are normalized, so the map is confined to an
    affine subspace and the spectrum carries eigenvalues belonging to the directions that
    leave it; they are harmless but should not be mistaken for physical modes. """
    jac = torch.func.jacfwd(state_map)(x)
    return torch.linalg.eigvals(jac)


def spectral_radius(eigvals: torch.Tensor) -> torch.Tensor:
    """ Largest eigenvalue modulus, over the last axis. Stable iff < 1. """
    return eigvals.abs().max(dim=-1).values


# Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
