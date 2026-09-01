from typing import Dict, Sequence, Tuple

import torch
import math

# Kernel-related functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def exponential_kernel(Q, tau_m, delta):
    """ Normalized exponential kernel for a leaky integrate-and-fire neuron model. """
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

# Parameter-grid batching
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def flatten_param_grid(
        param_grid: Dict[str, float | Sequence[float] | torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype | None = None,
        ) -> Tuple[Dict[str, torch.Tensor], Tuple[int, ...], Dict[str, torch.Tensor]]:
    """ Build the outer product of a named parameter grid and flatten it into a batch axis.

    Every value is coerced to a 1D tensor, so scalars simply contribute a length-1 (singleton)
    axis -- `unflatten_batch` drops those again, meaning the caller pays no shape penalty for
    the parameters it isn't sweeping.

    Returns (flat, grid_shape, axes):
      flat       -- dict of (B,) tensors, one per name, B = prod(grid_shape)
      grid_shape -- axis lengths in `param_grid` insertion order, singletons *included*
      axes       -- ordered dict of the swept (length > 1) input vectors only, so its values
                    line up with the leading axes of a tensor passed through `unflatten_batch`
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    vecs = {name: torch.as_tensor(v, dtype=dtype, device=device).reshape(-1)
            for name, v in param_grid.items()}
    grid_shape = tuple(v.numel() for v in vecs.values())
    mesh = torch.meshgrid(*vecs.values(), indexing="ij")
    flat = {name: m.reshape(-1) for name, m in zip(vecs, mesh)}
    axes = {name: v for name, v in vecs.items() if v.numel() > 1}
    return flat, grid_shape, axes

def unflatten_batch(tensor: torch.Tensor,
                    grid_shape: Tuple[int, ...],
                    batch_dim: int = 0,
                    drop_singletons: bool = True) -> torch.Tensor:
    """ Reshape a flattened batch dimension back into explicit parameter-grid axes.

    Parameters
    - tensor: input containing a flattened batch dimension of size prod(grid_shape).
        Example: a trajectory of shape (B, T, M) where B == prod(grid_shape).
    - grid_shape: axis sizes in the same order used by `flatten_param_grid`.
    - batch_dim: index of the flattened batch dimension (default 0, i.e. (B, T, ...)).
    - drop_singletons: if True, remove grid axes of length 1 after unflattening -- these are
        the parameters that were passed as scalars. If *every* axis is a singleton (nothing was
        swept) one axis of length 1 is kept, so a B=1 batch never loses its batch dimension.

    Returns the tensor with the batch dimension replaced by the explicit grid axes.
    """
    B = 1
    for s in grid_shape:
        B *= int(s)
    if tensor.shape[batch_dim] != B:
        raise ValueError(f"Batch dimension size mismatch: tensor has {tensor.shape[batch_dim]} "
                         f"at dim {batch_dim} (shape {tuple(tensor.shape)}), expected "
                         f"prod(grid_shape)={B} from grid_shape={grid_shape}")

    prefix = tuple(int(s) for s in tensor.shape[:batch_dim])
    suffix = tuple(int(s) for s in tensor.shape[batch_dim + 1:])

    grid_sizes = tuple(int(s) for s in grid_shape)
    if drop_singletons:
        grid_sizes = tuple(s for s in grid_sizes if s != 1) or (1,)

    return tensor.contiguous().reshape(prefix + grid_sizes + suffix)


# Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def grab_closest_idxs(vector: torch.Tensor, values: Sequence[float]) -> list[int]:
    """ Indices of the entries of `vector` closest to each of `values`. """
    return [int(torch.argmin((vector - value).abs())) for value in values]

