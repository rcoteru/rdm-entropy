from __future__ import annotations

from typing import Callable

import torch

from rdme.hazards import Hazard
from rdme.dynamics import compute_total_input, hazard_clamp

""" Fixed points: the closed forms at stationarity, the model's overlap map, and the generic
root finders that locate its roots.

Organised by topic rather than by how model-specific each piece is, so everything about a fixed
point is in one place. The boundary still matters and is marked by the section headers below:
the solvers know nothing about this model -- they take a residual callable and a tensor, and
are tested against cubics and closed-form roots -- while the maps above them evaluate a hazard.

The closed forms are themselves kernel-independent and take the hazard as data: the field never
sees it, and the survival and the age distribution take fprobs as an argument. Only the two maps
take a Hazard. """


# Closed forms at stationarity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# At a fixed point the input is constant, so the whole age-resolved state follows from it: the
# synaptic term unrolls to a geometric sum, the hazard is then a pure function of age, and the
# stationary age distribution is its normalized survival.

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


# The model's maps
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Composing the two closed forms above gives the overlap map F, whose roots m = F(m) are the
# fixed points -- one O(sum_a Q^a) pass per evaluation, no time integration.

def fp_overlap_map(m: torch.Tensor,        # (M,) trial overlaps
                   w: torch.Tensor,        # (M, M)
                   E: torch.Tensor,        # (M,)
                   alpha: torch.Tensor,    # (M,)
                   beta: torch.Tensor,     # (M,)
                   theta: torch.Tensor,    # (M,)
                   R: list[torch.Tensor],  # M tensors, (Qm[i],)
                   hazard: Hazard,
                   lam0: float,
                   dt: float,
                   ) -> torch.Tensor:      # (M,) F(m)
    """ One evaluation of the overlap map F: trial overlaps -> stationary overlaps.

    The hazard-parametrised counterpart of mean_field.fp_overlap_map. Only the hazard step
    differs: compute_stationary_forward_field and compute_stationary_age_distribution are
    reused from there unchanged, because both already take the hazard as data -- the first
    never sees it, and the second takes fprobs as an argument.

    Note this works in overlaps m, not rates, so that the residual lives on the same scale as
    the generic Newton and bracketing routines below (which clamp to
    [0, 1]) applies unchanged. Rates are m/dt throughout the class API. """
    I = compute_total_input(m / dt, w, E)       # w is unscaled here, so feed it the rate
    rates = []
    for i, R_i in enumerate(R):
        S_i = compute_stationary_forward_field(I[..., i], alpha[..., i], R_i.shape[-1])
        h_i = S_i + R_i - theta[..., i, None]
        phi_i = hazard.phi(h_i, beta[..., i, None], lam0, dt)
        rates.append(compute_stationary_age_distribution(phi_i)[1])
    return torch.stack(rates, dim=-1)


def fp_state(m: torch.Tensor,
             w: torch.Tensor,
             E: torch.Tensor,
             alpha: torch.Tensor,
             beta: torch.Tensor,
             theta: torch.Tensor,
             R: list[torch.Tensor],
             hazard: Hazard,
             lam0: float,
             dt: float,
             ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """ The full stationary state (p, S) implied by a set of overlaps. """
    I = compute_total_input(m / dt, w, E)
    p_out, S_out = [], []
    for i, R_i in enumerate(R):
        S_i = compute_stationary_forward_field(I[..., i], alpha[..., i], R_i.shape[-1])
        h_i = S_i + R_i - theta[..., i, None]
        phi_i = hazard.phi(h_i, beta[..., i, None], lam0, dt)
        p_out.append(compute_stationary_age_distribution(phi_i)[0])
        S_out.append(S_i)
    return p_out, S_out


# Root finding: Newton
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Everything from here down is model-agnostic: it takes a residual G(m) and never asks what
# produced it. Residual convention: G(m) = m - F(m), with G taking and returning an (M,) tensor.

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


# Auxiliary
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
