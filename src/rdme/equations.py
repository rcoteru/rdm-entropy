from __future__ import annotations

from typing import Callable

import torch

import rdme.rdm as rdm
import rdme.fixed_points as fpts
import rdme.lyapunov as lyap
from rdme.hazards import Hazard

""" The mean field's equations, as pure functions on per-population lists.

A module of its own rather than rdm.py because it composes fixed_points and lyapunov, which
both import rdm.py: putting these there would make the base layer import its own dependents.

Every function takes the state and parameters as arguments and reads nothing else. Populations
are lists (one tensor per population, its own Q_i on the last axis); every tensor carries
arbitrary LEADING axes, so the same call serves one system, (Q_i,), a batch, (B, Q_i), or a
grid. Per-population parameters are (..., M) and are indexed as x[..., i:i+1] against an
age axis. RDMNetwork and RDMNetworkBatch hold state and call these; neither writes an
equation of its own.

The arithmetic is the batch class's (decided 2026-11-13): out-of-place updates, and the firing
rate as a (1, Q) @ (Q, 1) product, which reproduces the batch's einsum bit for bit on every
shape including 1-D (see rdm.compute_firing_rate). """


# Observables
# ~~~~~~~~~~~

def rates(m: torch.Tensor, dt: float) -> torch.Tensor:
    """ r = m/dt, in 1/ms, from the per-bin overlaps m (..., M). """
    return m / dt


def activity(m: torch.Tensor, N_ratios: torch.Tensor, dt: float) -> torch.Tensor:
    """ N-weighted mean network rate, (...), in 1/ms. """
    return (N_ratios * rates(m, dt)).sum(dim=-1)


def age_density(p: list[torch.Tensor], dt: float) -> list[torch.Tensor]:
    """ q = p/dt, the age density of the transport equations. """
    return [p_i / dt for p_i in p]


def total_input(m: torch.Tensor, w: torch.Tensor, E: torch.Tensor, dt: float) -> torch.Tensor:
    """ I^a = sum_b w^{ab} r^b + E^a from the overlaps m (..., M); see rdm.compute_total_input
    for the weight convention. """
    return rdm.compute_total_input(rates(m, dt), w, E)


def field(S: list[torch.Tensor], R: list[torch.Tensor],
          theta: torch.Tensor) -> list[torch.Tensor]:
    """ h^a(u) = S^a(u) + R^a(u) - theta^a, one (..., Q_i) per population. """
    return [S_i + R_i - theta[..., i:i+1] for i, (S_i, R_i) in enumerate(zip(S, R))]


def per_population(fn: Callable[..., torch.Tensor], h: list[torch.Tensor],
                   beta: torch.Tensor, nu0: float, dt: float) -> list[torch.Tensor]:
    """ A hazard function -- Hazard.phi (per-bin firing probability), .dphi (its h-derivative)
    or .rate (the rate it represents, 1/ms) -- applied to each population's field. """
    return [fn(h_i, beta[..., i:i+1], nu0, dt) for i, h_i in enumerate(h)]


def max_step_prob(fprobs: list[torch.Tensor]) -> torch.Tensor:
    """ max over bins and populations of Phi, (...). """
    return torch.stack([f.amax(dim=-1) for f in fprobs], dim=0).amax(dim=0)


def truncation_error(p: list[torch.Tensor], h: list[torch.Tensor], I: torch.Tensor,
                     m: torch.Tensor, beta: torch.Tensor, theta: torch.Tensor,
                     phi: Callable[..., torch.Tensor], nu0: float, dt: float) -> torch.Tensor:
    """ p[-1] * |Phi(h_inf) - Phi(h[-1])| / m, max over populations, (...). The bound and why
    it is one: RDMNetworkBatch.truncation_error. """
    out = []
    for i in range(len(p)):
        h_inf = (I[..., i] - theta[..., i]).unsqueeze(-1)
        b = beta[..., i:i+1]
        gap = (phi(h_inf, b, nu0, dt).squeeze(-1)
               - phi(h[i][..., -1:], b, nu0, dt).squeeze(-1)).abs()
        out.append(p[i][..., -1] * gap / m[..., i].clamp_min(1e-300))
    return torch.stack(out, dim=0).amax(dim=0)


def stationary_rate(fprobs: list[torch.Tensor], dt: float) -> torch.Tensor:
    """ r* = [int_0^inf Survival(u) du]^{-1} at the given hazards, (..., M); the absorbing last
    bin contributes its exact geometric tail. See RDMNetwork.stationary_rate. """
    out = []
    for f in fprobs:
        # survival to the START of each bin: surv[0] = 1, surv[n] = prod_{k<n}(1 - Phi_k)
        keep = torch.cumprod(1 - f, dim=-1)
        surv = torch.cat([torch.ones_like(keep[..., :1]), keep[..., :-1]], dim=-1)
        tail = surv[..., -1] / f[..., -1].clamp_min(1e-300)     # absorbing bin, geometric
        out.append(1.0 / ((surv[..., :-1].sum(dim=-1) + tail) * dt))
    return torch.stack(out, dim=-1)


# Dynamics
# ~~~~~~~~

def advance(p: list[torch.Tensor], S: list[torch.Tensor], I: torch.Tensor,
            fprobs: list[torch.Tensor], alpha: torch.Tensor,
            ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """ One step along the characteristics, given the input I (..., M) and hazards fprobs of
    the CURRENT state. Returns (m, p, S) of the next one; nothing is modified in place. """
    m = torch.stack([rdm.compute_firing_rate(p_i, f_i) for p_i, f_i in zip(p, fprobs)], dim=-1)
    p_new = [rdm.update_age_distribution(p_i, f_i) for p_i, f_i in zip(p, fprobs)]
    S_new = [rdm.update_synaptic_term(S_i, alpha[..., i:i+1], I[..., i:i+1])
             for i, S_i in enumerate(S)]
    return m, p_new, S_new


def tangent_step(blocks: list[torch.Tensor], p: list[torch.Tensor],
                 fprobs: list[torch.Tensor], dphi: list[torch.Tensor],
                 w: torch.Tensor, alpha: torch.Tensor, dt: float) -> list[torch.Tensor]:
    """ The linearized step on 2M tangent blocks (..., k, Q_i), about the state whose hazards
    are fprobs/dphi; lyapunov.update_tangent with alpha (..., M) broadcast against (k, Q). """
    return lyap.update_tangent(blocks, p, fprobs, dphi, w,
                               [alpha[..., i:i+1, None] for i in range(len(p))], dt)


# Fixed points
# ~~~~~~~~~~~~

def fp_residual(m: torch.Tensor, w: torch.Tensor, E: torch.Tensor, alpha: torch.Tensor,
                beta: torch.Tensor, theta: torch.Tensor, R: list[torch.Tensor],
                hazard: Hazard, nu0: float, dt: float) -> torch.Tensor:
    """ G(m) = m - F(m), in overlaps; F is fixed_points.fp_overlap_map. The stationary state
    for a given m is fixed_points.fp_state, already pure and called as is. """
    return m - fpts.fp_overlap_map(m, w, E, alpha, beta, theta, R, hazard, nu0, dt)
