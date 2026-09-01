import torch
import math

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