from __future__ import annotations

import math

import torch

""" The exponential kernels the model is built from, and the bin counts they imply.

Pure functions of a time constant and a step: nothing here knows about populations, hazards or
networks. They are what ties the model's parameters, which are quoted in ms, to an age grid
quoted in bins -- which is why every one of them takes dt and why changing dt changes nothing
physical as long as it goes through these. """


def synaptic_kernel(Q, tau_m, delta):
    """ Synaptic kernel for the model.
    Normalized exponential kernel with time constant tau_m,
    evaluated at integer multiples of delta."""
    t = torch.arange(0, Q)
    return torch.exp(-t*delta/tau_m)/tau_m*delta


def refractory_kernel(Q: int, K: float, tau_r: float, dt: float = 1.0, device='cpu') -> torch.Tensor:
    """ Refractory kernel for the model. 
    Exponential kernel of with time constant tau_r and amplitude K, 
    evaluated at integer multiples of dt. """
    t = torch.arange(Q, device=device)
    return -K*torch.exp(-t*dt/tau_r)


def tau2alpha(tau: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Convert time constant tau to integration coefficient alpha. """
    return 1.0 - torch.exp(-dt / tau)


def q_from_tau(tau_m: float, dt: float, eps: float = 0.01) -> int:
    """
    Minimum number of bins Q such that the omitted tail mass of the
    normalized exponential kernel kappa(tau) = alpha * r**(tau-1),
    r = exp(-dt/tau_m), alpha = 1-r, falls below eps.

    Tail mass after Q terms is exactly r**Q = exp(-Q*dt/tau_m), so:
        r**Q < eps  =>  Q > -(tau_m/dt) * ln(eps)
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1)")
    tau_ratio = tau_m / dt          # tau_m expressed in units of dt (bins)
    Q = math.ceil(-tau_ratio * math.log(eps))
    return max(Q, 1)
