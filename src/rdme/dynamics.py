from __future__ import annotations

import torch

""" The per-step update, and the numerical guard every survival product needs.

The base layer of the package: these five operations are one timestep of the mean field, and
none of them knows which per-bin kernel is in use -- the hazard arrives as an argument
(fprobs) or not at all. Everything else in the package builds on this, and nothing here
builds on them.

hazard_clamp lives here rather than with the entropy production that motivated it, because the
stationary survival in rdme.fixed_points needs it too, and the alternative would be a
fixed_points -> epr import
for what is really a dtype-dependent numerical bound. """


# Per-step updates
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


# Numerical guards
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hazard_clamp(dtype: torch.dtype) -> float:
    """ Upper bound to clamp a hazard to before taking log1p(-phi).

    1 - 1e-12 is below float32's resolution -- it rounds to exactly 1, so a saturated hazard
    turns the survival into -inf and every EPR downstream into NaN. The bound therefore
    follows the dtype's own resolution, while float64 keeps its historical 1 - 1e-12. """
    return 1.0 - max(1e-12, torch.finfo(dtype).eps)
