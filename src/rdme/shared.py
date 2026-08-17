import torch
import math


def q_from_tau(tau_max: float, eps: float = 0.01) -> int:
    """ Minimum Q such that the exponential kernel exp(-Q/tau_max) < eps. """
    return math.ceil(-tau_max * math.log(eps))

def synaptic_kernel(Q, tau_m, delta):
    """ Normalized synaptic kernel for a leaky integrate-and-fire neuron model. """
    t = torch.arange(0, Q)
    return torch.exp(-t*delta/tau_m)/tau_m*delta

def simple_refractory_kernel(Q: int, K: float, tau_r: float, device='cpu') -> torch.Tensor:
    """ Generate a refractory kernel of length Q with time constant tau_r. """
    t = torch.arange(Q, device=device)
    return -K * torch.exp(-t / tau_r)

def tau2alpha(tau: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Convert time constant tau to integration coefficient alpha.
    
    Works with scalar, 1D tensor, or any shape tensor.
    alpha = 1 - exp(-dt / tau)
    Assumes dt=1.
    """
    return 1.0 - torch.exp(-dt / tau)