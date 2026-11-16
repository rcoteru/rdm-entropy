from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import torch
import tqdm

import rdme.rdm as rdm
import rdme.equations as eq
import rdme.epr as epr
import rdme.hazards as hzd
import rdme.kernels as krn
import rdme.fixed_points as fpts
import rdme.lyapunov as lyap

""" The RDM driver: one class for one system and for a batch of them.

RDMNetwork holds state, parameters and the EPR observer's feed, and orchestrates; it writes no
equation of its own -- those are rdme.equations, rdme.epr, rdme.fixed_points and rdme.lyapunov,
all pure functions on tensors with arbitrary leading axes. Batching is the SHAPE of what the
driver holds: w is (M, M) for one system and (B, M, M) for a batch of B, every per-population
parameter (M,) or (B, M), and every age-resolved tensor (Q_i,) or (B, Q_i). `select(b)` slices
a batch down to one system. "One system" means NO leading axis, not B = 1: a B = 1 batch takes
a different reduction path in the backend and its rows differ from a 1-D run in the last bit.

The driver knows nothing about presets: rdme.presets turns a preset name and overrides into the
keyword arguments the factories at the bottom take.

Weights are in physical units and NOT divided by dt -- pass J, never J/dt. The input is built
from the rate r = m/dt, which is what makes a coupling independent of the step; see
rdm.compute_total_input, which states the convention once. """


# Parameter grids
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# How a sweep becomes a batch, and how its outputs get their axes back.

def flatten_param_grid(
        param_grid: dict[str, float | Sequence[float] | torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype | None = None,
        ) -> tuple[dict[str, torch.Tensor], tuple[int, ...], dict[str, torch.Tensor]]:
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
                    grid_shape: tuple[int, ...],
                    batch_dim: int = 0,
                    drop_singletons: bool = True) -> torch.Tensor:
    """ Reshape a flattened batch dimension back into explicit parameter-grid axes.

    - tensor: input containing a flattened batch dimension of size prod(grid_shape), e.g. a
        trajectory of shape (B, T, M).
    - grid_shape: axis sizes in the same order used by `flatten_param_grid`.
    - batch_dim: index of the flattened batch dimension (default 0, i.e. (B, T, ...)).
    - drop_singletons: if True, remove grid axes of length 1 after unflattening -- these are
        the parameters that were passed as scalars. If *every* axis is a singleton (nothing was
        swept) one axis of length 1 is kept, so a B=1 batch never loses its batch dimension.
    """
    B = math.prod(int(s) for s in grid_shape)
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


def _scalar(x: torch.Tensor) -> torch.Tensor | float:
    """ A per-system reduction: a float for one system, a (B,) tensor for a batch. """
    return x.item() if x.ndim == 0 else x


class RDMNetwork:

    """ One RDM network, or a batch of B independent ones sharing M, N and Qm. """

    M: int                    # number of populations
    B: int | None             # batch size; None for one system (no leading axis)
    dt: float                 # solver step AND age-bin width, in ms -- numerics, not model
    nu0: float                # rate unit of the Poisson reference, in 1/ms (fixed at 1)
    t_abs: float              # absolute refractory period, in ms -- a physical dead time
    hazard: str               # key into rdme.hazards.HAZARDS
    Qm: list[int]             # (M,) age bins per population, shared by the batch
    N: list[int]              # (M,) neurons per population (sets the N-weighting only)
    w: torch.Tensor           # (*b, M, M) weights, PHYSICAL units -- not divided by dt

    p: list[torch.Tensor]     # age MASS per population, (*b, Qm[i]), sums to 1; density p/dt
    S: list[torch.Tensor]     # synaptic term, (*b, Qm[i])
    R: list[torch.Tensor]     # refractory kernel, (*b, Qm[i])

    E: torch.Tensor           # (*b, M) external drive
    beta: torch.Tensor        # (*b, M) gain
    theta: torch.Tensor       # (*b, M) threshold
    tau_int: torch.Tensor     # (*b, M) integration time constant (ms)
    tau_ref: torch.Tensor     # (*b, M) refractory time constant (ms)
    K_ref: torch.Tensor       # (*b, M) refractory amplitude
    alpha_int: torch.Tensor   # (*b, M) 1 - exp(-dt/tau_int)

    m: torch.Tensor           # (*b, M) per-bin firing probability; the rate is m/dt
    device: str
    N_ratios: torch.Tensor    # (M,) population size fractions

    grid_shape: tuple[int, ...]         # axis sizes whose product is B; () for one system
    grid_axes: dict[str, torch.Tensor]  # the swept 1D vectors, in grid order

    # Construction
    # ~~~~~~~~~~~~

    def __init__(self, M: int,
                 w: torch.Tensor,
                 N: list[int],
                 E: torch.Tensor | list[float],
                 beta: torch.Tensor | list[float],
                 theta: torch.Tensor | list[float],
                 tau_int: torch.Tensor | list[float],
                 tau_ref: torch.Tensor | list[float],
                 K_ref: torch.Tensor | list[float],
                 *,
                 dt: float,
                 hazard: str,
                 nu0: float = 1.0,
                 t_abs: float = 1.0,
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 grid_shape: tuple[int, ...] | None = None,
                 grid_axes: dict[str, torch.Tensor] | None = None,
                 device: str = 'cpu') -> None:
        """ w is (M, M) for one system or (B, M, M) for a batch; E, beta, theta, tau_int,
        tau_ref, K_ref are then (M,) (a length-M list will do) or (B, M). N is a plain length-M
        list shared by the batch: population sizes are structural, not swept.

        dt is the SOLVER step and is required: under route B there is no time quantum to
        default to. hazard is required for the same reason. dt, nu0 and t_abs are scalars
        shared by a batch (see hazards.resolve_timesteps). nu0 is the rate unit, 1 ms^-1;
        t_abs the physical dead time in ms, defaulting to 1 ms -- under an exponential hazard
        the only thing that bounds the realised rate, at 1/t_abs. Pass t_abs=0.0 for no rate
        ceiling, which has to be asked for explicitly. dt must divide t_abs
        (kernels.n_abs_bins), so that the window does not move with the step.

        Qm, if given, is shared. Otherwise each population's bin count comes from the slowest
        (tau_int, tau_ref) pair anywhere in the batch, which makes U_max = -tau*ln(eps)
        independent of dt -- and a batch only as cheap as its slowest member. This bounds the
        kernels, not the age density: a low-rate regime wants a larger Qm (helpers.fit_age_grid).

        grid_shape/grid_axes describe a batch built from a flattened parameter grid (see
        flatten_param_grid): trajectory outputs then come back with the batch axis expanded
        into the grid axes. Omit both for a plain flat batch or one system. """
        assert w.ndim in (2, 3) and w.shape[-2:] == (M, M), f"w must be ({M},{M}) or (B,{M},{M})"
        lead = tuple(w.shape[:-2])
        assert len(N) == M and all(isinstance(n, int) for n in N), "N must be M integers"
        dt, t_abs, nu0 = hzd.resolve_timesteps(dt, t_abs, nu0, hazard)
        self.nu0, self.t_abs = nu0, t_abs

        dtype = torch.get_default_dtype()
        self.M, self.dt, self.device = M, dt, device
        self.B = lead[0] if lead else None
        self.hazard = hazard
        self.N = N
        self.w = w.to(device=device, dtype=dtype)

        for name, val in [("E", E), ("beta", beta), ("theta", theta), ("tau_int", tau_int),
                          ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            t = torch.as_tensor(val, dtype=dtype, device=device)
            assert t.shape == lead + (M,), f"{name} must have shape {lead + (M,)}"
            setattr(self, name, t)

        self.grid_shape = (tuple(int(s) for s in grid_shape) if grid_shape is not None
                           else lead)
        assert math.prod(self.grid_shape) == math.prod(lead), \
            f"grid_shape {self.grid_shape} does not have B={self.B} cells"
        self.grid_axes = dict(grid_axes) if grid_axes is not None else {}

        self.N_ratios = torch.as_tensor(N, dtype=dtype, device=device)
        self.N_ratios = self.N_ratios / self.N_ratios.sum()
        self.alpha_int = krn.tau2alpha(self.tau_int, dt)

        if Qm is None:
            Qm = [krn.q_from_tau(max(self.tau_int[..., i].max().item(),
                                     self.tau_ref[..., i].max().item()), dt, eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        # t_abs is a scalar shared by the batch, as dt is, so n_abs is too. R_abs is NOT -- it
        # is sized per element from that element's weights (abs_refractory_floor reduces w
        # over its last axis and is elementwise in everything else); r_max = 1/t_abs is what
        # the window itself allows, see kernels.abs_refractory_floor
        self.n_abs = krn.n_abs_bins(self.t_abs, dt)
        self.R_abs = (krn.abs_refractory_floor(self.w, self.E, self.theta, self.beta,
                                               r_max=1.0 / self.t_abs)
                      if self.t_abs > 0 else torch.zeros_like(self.theta))
        if not lead:
            self.R = [krn.refractory_kernel(Qm[i], self.K_ref[i].item(), self.tau_ref[i].item(),
                                            dt=dt, t_abs=self.t_abs, R_abs=self.R_abs[i],
                                            device=device) for i in range(M)]
        else:
            self.R = [
                torch.vmap(lambda K, tau, Ra, i=i: krn.refractory_kernel(
                    self.Qm[i], K, tau, dt=dt, t_abs=self.t_abs, R_abs=Ra, device=device)
                )(self.K_ref[:, i], self.tau_ref[:, i], self.R_abs[:, i])
                for i in range(M)
            ]

        # all age mass in the oldest bin, zero integrated field
        self.p = [torch.zeros(lead + (Qm[i],), device=device) for i in range(M)]
        for pi in self.p: pi[..., -1] = 1.0
        self.S = [torch.zeros(lead + (Qm[i],), device=device) for i in range(M)]
        self.m = torch.zeros(lead + (M,), device=device)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """ The leading axes: () for one system, (B,) for a batch. """
        return tuple(self.w.shape[:-2])

    def select(self, b: int) -> RDMNetwork:
        """ Batch element b as a standalone single system (no leading axis).

        Sliced rather than rebuilt, so Qm and the refractory kernels are *copied*: a batch
        sizes one Qm from its slowest member, and re-deriving it for one element would
        silently give it a different age grid. The state comes along, so the selected model
        sits exactly where the batch does. """
        assert self.B is not None, "select() slices a batch; this is one system already"
        obj = copy.copy(self)
        obj.B, obj.grid_shape, obj.grid_axes = None, (), {}
        obj.N, obj.Qm = list(self.N), list(self.Qm)
        for key in ("w", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "alpha_int",
                    "R_abs"):
            setattr(obj, key, getattr(self, key)[b])
        obj.R = [R_i[b].clone() for R_i in self.R]
        obj.p = [p_i[b].clone() for p_i in self.p]
        obj.S = [S_i[b].clone() for S_i in self.S]
        obj.m = self.m[b].clone()
        return obj

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Expand a leading flat batch axis (B, ...) into the parameter-grid axes.
        Pass-through for one system or a batch that did not come from a grid. Unrelated to
        unflatten_state, which splits a flat state vector into (p, S). """
        if len(self.grid_shape) <= 1:
            return x
        return unflatten_batch(x, self.grid_shape)

    # Serialization
    # ~~~~~~~~~~~~~

    _SAVE_TENSORS = ("w", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref",
                     "alpha_int", "N_ratios", "m")
    _SAVE_LISTS   = ("R", "p", "S")
    _SAVE_SCALARS = ("B", "M", "dt", "nu0", "t_abs", "hazard", "device")

    def save(self, path: str | Path) -> None:
        """ Full state, parameters and grid layout, for one system or a batch. """
        data = {k: getattr(self, k) for k in self._SAVE_SCALARS}
        data.update({"Qm": self.Qm, "N": self.N,
                     "grid_shape": self.grid_shape, "grid_axes": self.grid_axes})
        data.update({k: getattr(self, k) for k in self._SAVE_TENSORS})
        data.update({k: getattr(self, k) for k in self._SAVE_LISTS})
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> RDMNetwork:
        """ Load a network saved with save(), without rebuilding the parameter grid. Reads the
        files RDMNetworkBatch.save wrote, which have the same keys.

        No legacy branch: saves written under the dt0 convention are not loadable here and are
        not meant to be. They carry a model (lam0 = 1/dt0, t_abs = dt0) that route B does not
        have, so a pre-route-B file fails on the missing key, which is the honest outcome. """
        data = torch.load(path, map_location=device, weights_only=False)
        obj = object.__new__(cls)
        for k in cls._SAVE_SCALARS:
            setattr(obj, k, data[k])
        obj.Qm, obj.N = list(data["Qm"]), list(data["N"])
        obj.grid_shape = tuple(data["grid_shape"])
        obj.device = device if device is not None else data["device"]
        obj.grid_axes = {k: v.to(obj.device) for k, v in data["grid_axes"].items()}
        for key in cls._SAVE_TENSORS:
            setattr(obj, key, data[key].to(obj.device))
        for key in cls._SAVE_LISTS:
            setattr(obj, key, [t.to(obj.device) for t in data[key]])
        obj.n_abs = krn.n_abs_bins(obj.t_abs, obj.dt)
        obj.R_abs = (krn.abs_refractory_floor(obj.w, obj.E, obj.theta, obj.beta,
                                              r_max=1.0 / obj.t_abs)
                     if obj.t_abs > 0 else torch.zeros_like(obj.theta))
        return obj

    @property
    def is_flow(self) -> bool:
        """ True when this model is a sampled FLOW rather than a genuine map: the caller's half
        of lyapunov.classify_spectrum. Under route B every model is a flow, with the single
        exception of the legacy discrete model -- "bernoulli" exactly at nu0*dt = 1, where the
        per-bin kernel is the sigmoid and the update is a map in its own right. """
        return not (self.hazard == "bernoulli" and abs(self.nu0 * self.dt - 1.0) <= 1e-12)

    # Observables
    # ~~~~~~~~~~~

    @torch.inference_mode()
    def rates(self) -> torch.Tensor:
        """ Population firing rates r = m/dt, (*b, M), in 1/ms. The dt-invariant observable:
        compare these across resolutions, never the per-bin overlaps. """
        return eq.rates(self.m, self.dt)

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Per-bin firing probabilities m, (*b, M). O(dt), so not comparable across
        resolutions; rates() is. """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ N-weighted mean network rate, (*b,), in 1/ms. """
        return eq.activity(self.m, self.N_ratios, self.dt)

    @torch.inference_mode()
    def age_density(self) -> list[torch.Tensor]:
        """ q = p/dt, the age density of the transport equations, list of (*b, Qm[i]).
        Integrates to 1 over age; unlike the mass p, it has a dt -> 0 limit. """
        return eq.age_density(self.p, self.dt)

    @torch.inference_mode()
    def max_step_prob(self) -> torch.Tensor | float:
        """ max over bins and populations of Phi at the CURRENT state, per system (a float for
        one system, (B,) for a batch): the largest fraction of an age bin that fires in one
        step. Near 1 a bin can empty in a single step, which is the two-step flip.

        ONE INSTANT, which on an oscillating attractor is not the worst case; to answer "is dt
        small enough?" take the max along a trajectory covering at least one period. For
        "bernoulli" Phi <= nu0*dt everywhere; for the route-B kernels the rate is unbounded,
        so nu0*dt bounds nothing and only the realized max(Phi) sees it. The companion
        accuracy criterion is hazards.check_accuracy on max(hazard_rate()). """
        return _scalar(eq.max_step_prob(self.firing_prob()))

    @torch.inference_mode()
    def truncation_error(self) -> torch.Tensor | float:
        """ Rigorous relative bound on the firing-rate error from the finite age grid, per
        system (a float for one system, (B,) for a batch):

            err = p[-1] * |Phi(h_inf) - Phi(h[-1])| / m

        i.e. the fraction of the population's firing misassigned because the age axis stops
        at U_max = Q*dt; `tol = 1e-2` means "the rate is right to 1%".

        A bound, not an estimate: the boundary bin is absorbing, so every cohort older than
        U_max fires at Phi(h[-1]) when it should fire at Phi(h(u)). But h(u) = S + R - theta
        is MONOTONE increasing in age (S saturates up to I, R decays up to zero) and Phi is
        monotone in h, so every true Phi(h(u)) lies between Phi(h[-1]) and Phi(h_inf), and the
        displaced mass p[-1] cannot be wrong by more than their gap.

        It supersedes thresholding the boundary MASS p[-1] (a silent cell correctly keeps its
        mass at old ages, so any sweep containing one fails everywhere) and thresholding
        p[-1] * beta * |R(U_max)| (with K_ref = 0 it reports convergence however much mass is
        frozen). A population with m = 0 is not guarded: there is no rate to distort. """
        return _scalar(eq.truncation_error(self.p, self.field(), self.total_input(), self.m,
                                           self.beta, self.theta, hzd.HAZARDS[self.hazard].phi,
                                           self.nu0, self.dt))

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I^a = sum_b w^{ab} r^b + E^a, (*b, M). Built from the RATE, which is why w
        carries no 1/dt here. """
        return eq.total_input(self.m, self.w, self.E, self.dt)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ h^a(u) = S^a(u) + R^a(u) - theta^a, list of (*b, Qm[i]). """
        return eq.field(self.S, self.R, self.theta)

    @torch.inference_mode()
    def hazard_rate(self) -> list[torch.Tensor]:
        """ The firing rate lambda(h) the hazard represents, in 1/ms, list of (*b, Qm[i]).
        Bounded by nu0 for "bernoulli", unbounded and exponential for "poisson" /
        "poisson_exact", which share it exactly. """
        return eq.per_population(hzd.HAZARDS[self.hazard].rate, self.field(), self.beta,
                                 self.nu0, self.dt)

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Per-bin hazard Phi, list of (*b, Qm[i]). """
        return eq.per_population(hzd.HAZARDS[self.hazard].phi, self.field(), self.beta,
                                 self.nu0, self.dt)

    @torch.inference_mode()
    def d_firing_prob(self) -> list[torch.Tensor]:
        """ dPhi/dh at the current state, list of (*b, Qm[i]). Needed by the tangent map. """
        return eq.per_population(hzd.HAZARDS[self.hazard].dphi, self.field(), self.beta,
                                 self.nu0, self.dt)

    @torch.inference_mode()
    def stationary_rate(self) -> torch.Tensor:
        """ r* = [int_0^inf Survival(u) du]^{-1} at the CURRENT field, (*b, M).

        The fixed-point form of the renewal equation: at a genuine fixed point it equals
        rates() to roundoff, away from one it does not, so it is a consistency check on a
        settled run rather than a solver. The absorbing last bin contributes its exact
        geometric tail S[Q-1]/Phi[Q-1]; dropping it biases r* high by the mass past the
        truncation. """
        return eq.stationary_rate(self.firing_prob(), self.dt)

    # Fixed points
    # ~~~~~~~~~~~~

    def _fp_args(self) -> tuple:
        return (self.w, self.E, self.alpha_int, self.beta, self.theta, self.R,
                hzd.HAZARDS[self.hazard], self.nu0, self.dt)

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m), in OVERLAPS (per-bin), (*b, M): on the [0, 1] scale the generic
        Newton routines in rdme.fixed_points clamp to. Divide a root by dt for the rate. """
        return eq.fp_residual(m, *self._fp_args())

    def fp_state(self, m: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ The stationary (p, S) implied by the overlaps m. """
        return fpts.fp_state(m, *self._fp_args())

    def set_state(self, m: torch.Tensor) -> None:
        """ Place every system on the stationary state implied by its overlaps m (*b, M). """
        p, S = self.fp_state(m)
        self.p = [t.clone() for t in p]
        self.S = [t.clone() for t in S]
        self.m = torch.as_tensor(m, device=self.device).clone().reshape(self.batch_shape
                                                                        + (self.M,))

    def fixed_points(self,
                     method: str = 'auto',
                     guesses: torch.Tensor | None = None,
                     n_grid: int = 201,
                     tol: float = 1e-10,
                     merge_tol: float = 1e-6,
                     **kwargs) -> torch.Tensor:     # (R, M) roots, one row per fixed point
        """ All fixed points of ONE system, as overlap vectors (for a batch: fp_sweep, or
        select(b) first).

        method='bracket' scans [0, 1] for sign changes of G and bisects them; it is exhaustive
        up to the grid resolution and is the default for M=1, where it reliably picks up the
        unstable middle branch inside a bistable region. method='newton' runs Newton from a
        lattice of starting points and is the default for M>1; it finds whatever the basins
        reach, so a root missing from its output is not proof that none exists.

        Roots come back unclassified -- pass each to fp_eigenvalues (or is_stable). In
        overlaps, not rates: at small dt [0, 1] is a very coarse net over a root sitting at
        O(dt), so pass `guesses` explicitly when refining. """
        assert self.B is None, "fixed_points is per system: use fp_sweep, or select(b)"
        if method == 'auto':
            method = 'bracket' if self.M == 1 else 'newton'

        if method == 'bracket':
            assert self.M == 1, "bracketing is a scalar method; use method='newton' for M > 1"
            return fpts.bracket_roots(self.fp_residual, n_grid=n_grid,
                                      merge_tol=merge_tol, device=self.device, **kwargs)
        if method == 'newton':
            if guesses is None:
                guesses = fpts.fp_guess_lattice(self.M, n_per_axis=5, device=self.device)
            return fpts.newton_roots(self.fp_residual, guesses, tol=tol,
                                     merge_tol=merge_tol, **kwargs)
        raise ValueError(f"unknown method {method!r}, expected 'auto', 'bracket' or 'newton'")

    def _fp_residual_one(self, m: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """ G(m) for a *single* system, with that system's parameters passed in rather than
        read off self: the form torch.vmap needs to give every Newton trajectory its own
        parameters. Argument order matches fp_sweep's params tuple. """
        w, E, alpha, beta, theta = params[:5]
        return eq.fp_residual(m, w, E, alpha, beta, theta, list(params[5:]),
                              hzd.HAZARDS[self.hazard], self.nu0, self.dt)

    def fp_sweep(self,
                 guesses: torch.Tensor | None = None,
                 n_per_axis: int = 5,
                 tol: float = 1e-10,
                 res_tol: float | None = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ Newton from K starting points, for every system at once.

        Every (system, starting point) pair is one Newton trajectory and they all run
        simultaneously, so a phase map costs a single vmapped solve. Returns (roots,
        converged), shaped (*b, K, M) and (*b, K); the rows `converged` marks False hold junk.

        Deliberately *not* deduplicated: collapsing K guesses into a variable number of
        distinct roots is ragged per system; fixed_points.deduplicate is there for the systems
        worth inspecting. The default lattice spans overlaps in [0, 1] -- pass `guesses`
        explicitly when refining. """
        if guesses is None:
            guesses = fpts.fp_guess_lattice(self.M, n_per_axis, device=self.device)
        K, lead = guesses.shape[0], self.batch_shape
        B = math.prod(lead)

        def rep(t): return t.reshape((B,) + t.shape[len(lead):]).repeat_interleave(K, dim=0)
        params = (rep(self.w), rep(self.E), rep(self.alpha_int), rep(self.beta),
                  rep(self.theta), *[rep(R_i) for R_i in self.R])
        m0 = guesses.repeat(B, 1)                                    # (B*K, M)

        m = fpts.newton_iterate(self._fp_residual_one, m0, params, tol=tol, **kwargs)
        res = fpts.residual_norms(self._fp_residual_one, m, params)
        converged = torch.isfinite(res) & (res < (res_tol if res_tol is not None else 1e3 * tol))
        return m.reshape(lead + (K, self.M)), converged.reshape(lead + (K,))

    # State vector, for the Jacobian and the tangent dynamics
    # ~~~~~~~~~~~~

    def flatten_state(self, p: list[torch.Tensor], S: list[torch.Tensor]) -> torch.Tensor:
        """ Stack the per-population (p, S) into one flat vector along the last axis,
        interleaved as [p_0, S_0, p_1, S_1, ...] -- lyapunov's block convention. """
        return torch.cat([t for i in range(self.M) for t in (p[i], S[i])], dim=-1)

    def unflatten_state(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ Inverse of flatten_state. """
        p, S, off = [], [], 0
        for Q in self.Qm:
            p.append(x[..., off:off + Q]); off += Q
            S.append(x[..., off:off + Q]); off += Q
        return p, S

    def state_map(self, x: torch.Tensor) -> torch.Tensor:
        """ The one-step update on the FULL state, as a pure function -- the thing to
        differentiate. Deliberately not decorated with inference_mode and deliberately
        out-of-place, so torch.func can trace it. m is read back as p[i][..., 0] rather than
        carried in the state, so the map has no spurious zero mode from a redundant variable. """
        p, S = self.unflatten_state(x)
        m = torch.stack([p[i][..., 0] for i in range(self.M)], dim=-1)
        I = eq.total_input(m, self.w, self.E, self.dt)
        fprobs = eq.per_population(hzd.HAZARDS[self.hazard].phi, eq.field(S, self.R, self.theta),
                                   self.beta, self.nu0, self.dt)
        _, p_new, S_new = eq.advance(p, S, I, fprobs, self.alpha_int)
        return self.flatten_state(p_new, S_new)

    def fp_eigenvalues(self, m: torch.Tensor) -> torch.Tensor:
        """ Eigenvalues of the linearized map at the fixed point with overlaps m (one system). """
        assert self.B is None, "fp_eigenvalues is per system: select(b) first"
        x = self.flatten_state(*self.fp_state(m))
        return fpts.jacobian_eigvals(self.state_map, x)

    def is_stable(self, m: torch.Tensor, tol: float = 1e-8) -> bool:
        """ Linear stability of the fixed point with overlaps m: spectral radius < 1. """
        return bool(fpts.spectral_radius(self.fp_eigenvalues(m)) < 1.0 - tol)

    # Tangent dynamics (Lyapunov exponents)
    # ~~~~~~~~~~~~

    def init_tangent(self, k: int,
                     generator: torch.Generator | None = None,
                     ) -> list[torch.Tensor]:   # 2M blocks, each (*b, k, Qm[i])
        """ A random orthonormal, sum-zero tangent basis for every system. """
        return lyap.init_tangent_blocks(self.Qm, k, prefix=self.batch_shape,
                                        device=self.device, generator=generator)

    @torch.no_grad()
    def step_tangent(self, blocks: list[torch.Tensor]) -> None:
        """ Advance the state and a tangent basis together by one step, in place.

        One call, because the tangent is the linearization at the current state and the two
        must share one hazard evaluation. no_grad rather than inference_mode because the
        Lyapunov orthonormalizer updates these blocks in place, which inference tensors forbid
        outside inference mode. """
        I      = self.total_input()
        fprobs = self.firing_prob()
        dphi   = self.d_firing_prob()
        blocks[:] = eq.tangent_step(blocks, self.p, fprobs, dphi, self.w, self.alpha_int, self.dt)
        self._advance_from(I, fprobs)

    # Dynamics
    # ~~~~~~~~

    @torch.inference_mode()
    def _advance_from(self, I: torch.Tensor, fprobs: list[torch.Tensor]) -> None:
        """ Advance (m, p, S) one step given the input and hazards of the CURRENT state.

        Split out so step_tangent can reuse the same I and fprobs for state and tangent: this
        replaces the state, so a second evaluation afterwards would linearize about a state
        that had already moved.

        One system checks its age normalization every step (a NaN guard, in effect: the update
        keeps the mass at 1 by construction); a batch does not, as before the merge, since on
        a GPU the check is a host sync per step. """
        self.m, self.p, self.S = eq.advance(self.p, self.S, I, fprobs, self.alpha_int)
        if self.B is None:
            for pi in self.p:
                rdm.check_age_normalization(pi)

    @torch.inference_mode()
    def update(self) -> None:
        """ One step along the characteristics: age and time advance together by dt. """
        I      = self.total_input()     # from the OLD rates
        fprobs = self.firing_prob()     # from the OLD S/R
        self._advance_from(I, fprobs)

    @torch.inference_mode()
    def forward(self, T: int, pb: bool = False) -> None:
        for _ in tqdm.tqdm(range(T), disable=not pb):
            self.update()

    # Trajectories
    # ~~~~~~~~~~~~

    def _out_device_kwargs(self) -> dict:
        """ Allocation kwargs for trajectory accumulators, float64. These grow linearly with
        T, so on a GPU they are pinned in CPU memory and each step is streamed in with a
        non_blocking copy, which keeps that growth off the card. """
        if torch.device(self.device).type == 'cuda':
            return dict(device='cpu', pin_memory=True, dtype=torch.float64)
        return dict(device=self.device, dtype=torch.float64)

    def _unflatten_out(self, out: dict) -> dict:
        return {k: self.unflatten(v) if torch.is_tensor(v) else [self.unflatten(t) for t in v]
                for k, v in out.items()}

    @torch.inference_mode()
    def trajectory(self, T: int,
                   r_tot: bool = False,
                   q: bool = False,
                   S: bool = False,
                   pb: bool = True,
                   ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """ Run for T steps. Always returns "r" and "m" (*grid, T, M), the rates in 1/ms and
        the per-bin overlaps. Optional: r_tot (*grid, T) the N-weighted rate; q/S as lists of
        M tensors, each (*grid, T, Qm[i]). The leading axes are none for one system, (B,) for
        a batch, the grid axes for a batch built from a grid. CPU float64 tensors when the
        model runs on CPU or CUDA. Time axis is in steps; multiply by dt for ms. """
        ok, lead = self._out_device_kwargs(), self.batch_shape
        out: dict = {"r": torch.zeros(lead + (T, self.M), **ok),
                     "m": torch.zeros(lead + (T, self.M), **ok)}
        if r_tot: out["r_tot"] = torch.zeros(lead + (T,), **ok)
        if q: out["q"] = [torch.zeros(lead + (T, Q), **ok) for Q in self.Qm]
        if S: out["S"] = [torch.zeros(lead + (T, Q), **ok) for Q in self.Qm]

        for t in tqdm.tqdm(range(T), disable=not pb):
            out["r"][..., t, :].copy_(self.rates(), non_blocking=True)
            out["m"][..., t, :].copy_(self.m, non_blocking=True)
            if r_tot: out["r_tot"][..., t].copy_(self.activity(), non_blocking=True)
            if q:
                for i in range(self.M):
                    out["q"][i][..., t, :].copy_(self.p[i] / self.dt, non_blocking=True)
            if S:
                for i in range(self.M): out["S"][i][..., t, :].copy_(self.S[i], non_blocking=True)
            self.update()
        return self._unflatten_out(out)

    @torch.inference_mode()
    def entropy_trajectory(self, T: int,
                           pb: bool = True,
                           inplace: bool = True,
                           ) -> dict[str, torch.Tensor]:
        """ Per-population entropy production over T reported steps, by the O(Q) observer
        rdme.epr.epr_step (the joint-distribution oracle it is checked against lives in
        tests/epr_reference.py).

        The reverse field needs the input Qm[i] - 1 steps into the FUTURE, so each population's
        observer lags the model by its own Qm[i] - 1 steps: the model runs T + Q_max - 1 steps,
        and population i's first Qm[i] - 1 of them only fill its input window. Each observer
        step is fed the model's live S and input, its window's leading edge; it carries its own
        lagged S, and nothing O(Q^2) persists across steps. Afterwards the live state is rewound
        to the last reported step, so stepping on continues from T.

        `inplace` lets the hazard and survival chains in rdme.epr reuse their own temporaries;
        bit for bit with the out-of-place oracle (inplace=False), and ~10% faster at large Q.

        Returns r/m (*grid, T, M), r_tot/m_tot (*grid, T); sigma/H_fwd/H_rev (*grid, T, M) and
        their N-weighted aggregates. sigma is per bin; divide by dt for the rate, the only one
        of the three with a continuum limit (see the rdme package docstring). """
        return self._observe(T, epr.init_epr_state, epr.epr_step, pb=pb, inplace=inplace)

    @torch.inference_mode()
    def _observe(self, T: int, init: Callable, step: Callable, pb: bool,
                 inplace: bool) -> dict[str, torch.Tensor]:
        """ The entropy_trajectory loop, for any observer (init, step) with epr_step's feed:
        init(p_t, S_t, I_window, EPRParams, inplace) -> state, and
        step(state, S_live, I_new, EPRParams, inplace) -> (state, sigma, H_fwd, H_rev), where
        state.p and state.S are the lagged marginal and synaptic term. """
        M, Qm, lead = self.M, self.Qm, self.batch_shape
        Q_max = max(Qm)
        hz, nu0, dt = hzd.HAZARDS[self.hazard], self.nu0, self.dt
        ok = self._out_device_kwargs()

        out = {k: torch.zeros(lead + (T, M), **ok) for k in ("r", "m", "sigma", "H_fwd", "H_rev")}

        p0, S0 = [p.clone() for p in self.p], [S.clone() for S in self.S]
        I_hist = []                                 # I_0 .. I_{Q_max-2}: init only
        state = [None] * M
        prm = [epr.EPRParams(self.R[i], self.alpha_int[..., i:i+1], self.beta[..., i:i+1],
                             self.theta[..., i:i+1], hz, nu0, dt) for i in range(M)]

        # Model time tau runs T + Q_max - 1 steps. Population i's observer lags it by its own
        # Qm[i] - 1, so at tau it reports t = tau - Qm[i] + 1 and is fed the model's live S and
        # input, which are exactly the window edge S_{t+Qm[i]-1}, I_{t+Qm[i]-1} it needs.
        for tau in tqdm.tqdm(range(T + Q_max - 1), disable=not pb):
            I = self.total_input()
            for i in range(M):
                t = tau - (Qm[i] - 1)
                if not 0 <= t < T:
                    continue
                if state[i] is None:
                    I_win = torch.stack([I_hist[j][..., i] for j in range(Qm[i] - 1)], dim=-1)
                    state[i] = init(p0[i], S0[i], I_win, prm[i], inplace=inplace)
                    p0[i] = S0[i] = None            # the observer holds them from here on
                    if all(st is not None for st in state):
                        I_hist.clear()              # every observer holds its own window now
                # read the activity before the observer advances the marginal, so m_t is
                # reported against the sigma of the t -> t+1 transition rather than m_{t+1}
                out["m"][..., t, i].copy_(state[i].p[..., 0], non_blocking=True)
                state[i], *e = step(state[i], self.S[i], I[..., i], prm[i], inplace=inplace)
                for k, v in zip(("sigma", "H_fwd", "H_rev"), e):
                    out[k][..., t, i].copy_(v, non_blocking=True)
            if tau < Q_max - 1:
                I_hist.append(I)
            self.update()

        N_ratios = self.N_ratios.to(ok["device"])
        out["r"] = out["m"] / dt
        out["m_tot"] = (out["m"] * N_ratios).sum(dim=-1)
        out["r_tot"] = out["m_tot"] / dt

        # rewind the live state to the last reported step, discarding the look-ahead
        for i in range(M):
            self.p[i] = state[i].p
            self.S[i] = state[i].S
        self.m = torch.stack([self.p[i][..., 0] for i in range(M)], dim=-1)

        for k in ("sigma", "H_fwd", "H_rev"):
            out[k + "_tot"] = (out[k] * N_ratios).sum(dim=-1)
        return self._unflatten_out(out)


# Model factories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One system takes scalars and builds an unbatched network; the *Batch / *Points factories
# build a batch. J and every w are in physical units: pass J, never J/dt.

def RDMIsingModel(J: float, E: float, beta: float, theta: float, tau_int: float,
                  tau_ref: float, K_ref: float, *, dt: float, hazard: str,
                  nu0: float = 1.0, t_abs: float = 1.0, Qm: list[int] | None = None,
                  eps: float = 0.01, device: str = 'cpu') -> RDMNetwork:
    """ One single-population (M=1) network. """
    return RDMNetwork(M=1, w=torch.tensor([[J]]), N=[1000], E=[E], beta=[beta], theta=[theta],
                      tau_int=[tau_int], tau_ref=[tau_ref], K_ref=[K_ref],
                      dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, Qm=Qm, eps=eps,
                      device=device)


def RDMIsingModelBatch(J, E, beta, theta, tau_int, tau_ref, K_ref,
                       *, dt: float, hazard: str, nu0: float = 1.0,
                       t_abs: float = 1.0,
                       Qm: list[int] | None = None, eps: float = 0.01,
                       device: str = 'cpu') -> RDMNetwork:
    """ A batch of single-population (M=1) networks. Each parameter may be a scalar or a 1D
    tensor, and the batch is the **outer product** of them all, so passing two vectors gives
    the full 2D sweep with no meshgrid at the call site. Scalars contribute a singleton axis
    that is dropped again on output; the result always has a batch axis, even if B = 1. """
    flat, grid_shape, axes = flatten_param_grid(
        {"J": J, "E": E, "beta": beta, "theta": theta,
         "tau_int": tau_int, "tau_ref": tau_ref, "K_ref": K_ref}, device=device)
    col = lambda k: flat[k].unsqueeze(-1)                         # (B, 1)
    return RDMNetwork(
        M=1, w=flat["J"].reshape(-1, 1, 1), N=[1000], E=col("E"), beta=col("beta"),
        theta=col("theta"), tau_int=col("tau_int"), tau_ref=col("tau_ref"),
        K_ref=col("K_ref"), dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, Qm=Qm, eps=eps,
        grid_shape=grid_shape, grid_axes=axes, device=device)


def RDMWilsonCowan(E_ratio: float, w_EE: float, w_EI: float, w_IE: float, w_II: float,
                   E_exc: float, E_inh: float, beta_E: float, beta_I: float,
                   theta_E: float, theta_I: float, tau_int_E: float, tau_int_I: float,
                   tau_ref_E: float, tau_ref_I: float, K_ref_E: float, K_ref_I: float,
                   *, dt: float, hazard: str, nu0: float = 1.0, t_abs: float = 1.0,
                   Qm: list[int] | None = None, eps: float = 0.01,
                   device: str = 'cpu') -> RDMNetwork:
    """ One Wilson-Cowan-like two-population (M=2, order [E, I]) network. """
    N_E = int(E_ratio * 1000)
    # row = target, column = source, matching w^{ab} ("from b onto a")
    w = torch.tensor([[w_EE, w_IE],
                      [w_EI, w_II]])
    return RDMNetwork(M=2, w=w, N=[N_E, 1000 - N_E], E=[E_exc, E_inh], beta=[beta_E, beta_I],
                      theta=[theta_E, theta_I], tau_int=[tau_int_E, tau_int_I],
                      tau_ref=[tau_ref_E, tau_ref_I], K_ref=[K_ref_E, K_ref_I],
                      dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, Qm=Qm, eps=eps,
                      device=device)


def _wilson_cowan_from_flat(flat: dict[str, torch.Tensor], E_ratio: float, dt: float,
                            nu0, hazard, t_abs, Qm, eps, grid_shape, grid_axes,
                            device) -> RDMNetwork:
    """ Assemble a two-population batch from flat (B,) parameter columns. Shared by the two
    batched Wilson-Cowan factories, which differ only in how they get to `flat`: outer
    product in one, elementwise in the other. """
    N_E = int(E_ratio * 1000)
    # transposed as in RDMWilsonCowan: row = target, column = source
    w = torch.stack([
        torch.stack([flat["w_EE"], flat["w_IE"]], dim=-1),
        torch.stack([flat["w_EI"], flat["w_II"]], dim=-1),
    ], dim=1)   # (B, 2, 2)
    two = lambda a, b: torch.stack([flat[a], flat[b]], dim=-1)     # (B, 2)
    return RDMNetwork(
        M=2, w=w, N=[N_E, 1000 - N_E],
        E=two("E_exc", "E_inh"), beta=two("beta_E", "beta_I"),
        theta=two("theta_E", "theta_I"), tau_int=two("tau_int_E", "tau_int_I"),
        tau_ref=two("tau_ref_E", "tau_ref_I"), K_ref=two("K_ref_E", "K_ref_I"),
        dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, Qm=Qm, eps=eps,
        grid_shape=grid_shape, grid_axes=grid_axes, device=device)


_WC_NAMES = ["w_EE", "w_EI", "w_IE", "w_II", "E_exc", "E_inh", "beta_E", "beta_I",
             "theta_E", "theta_I", "tau_int_E", "tau_int_I", "tau_ref_E", "tau_ref_I",
             "K_ref_E", "K_ref_I"]


def RDMWilsonCowanBatch(E_ratio, w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                        theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                        K_ref_E, K_ref_I, *, dt: float, hazard: str,
                        nu0: float = 1.0, t_abs: float = 1.0,
                        Qm: list[int] | None = None,
                        eps: float = 0.01, device: str = 'cpu') -> RDMNetwork:
    """ A batch of two-population (M=2, order [E, I]) networks, the batch being the **outer
    product** of every argument. Scalars contribute a singleton axis that is dropped again on
    output. E_ratio is structural (it sets the population sizes) and stays a scalar. """
    vals = dict(zip(_WC_NAMES, [w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                                theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                                K_ref_E, K_ref_I]))
    flat, grid_shape, axes = flatten_param_grid(vals, device=device)
    return _wilson_cowan_from_flat(flat, E_ratio, dt, nu0, hazard, t_abs, Qm, eps,
                                   grid_shape, axes, device)


def RDMWilsonCowanPoints(E_ratio, w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                         theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                         K_ref_E, K_ref_I, *, dt: float, hazard: str,
                         nu0: float = 1.0, t_abs: float = 1.0,
                         Qm: list[int] | None = None,
                         eps: float = 0.01, grid_shape=None, grid_axes=None,
                         device: str = 'cpu') -> RDMNetwork:
    """ One parameter point per batch element -- the **elementwise** counterpart of
    RDMWilsonCowanBatch, for search axes that drive two parameters at once (a loop gain g with
    w_EI = g*rho and w_IE = -g/rho would come back from the outer product as the n**2 cross
    product). The caller reparametrises and passes the resulting columns.

    Every parameter may be a scalar or a 1D tensor; scalars and length-1 tensors broadcast
    against the longest, and B is that length. Mismatched lengths are an error rather than a
    broadcast. grid_shape/grid_axes are passed straight through: only the caller knows which
    search axes the flat batch came from. """
    dtype = torch.get_default_dtype()
    vals = dict(zip(_WC_NAMES, [w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                                theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                                K_ref_E, K_ref_I]))
    cols = {k: torch.as_tensor(v, dtype=dtype, device=device).reshape(-1)
            for k, v in vals.items()}
    lengths = {k: v.numel() for k, v in cols.items()}
    B = max(lengths.values())
    bad = {k: n for k, n in lengths.items() if n not in (1, B)}
    if bad:
        raise ValueError(f"RDMWilsonCowanPoints: every parameter must have length 1 or B={B}, "
                         f"got {bad}. Pass one value per batch element, or a scalar to hold it "
                         f"fixed.")
    flat = {k: (v.expand(B) if v.numel() == 1 else v) for k, v in cols.items()}
    return _wilson_cowan_from_flat(flat, E_ratio, dt, nu0, hazard, t_abs, Qm, eps,
                                   grid_shape, grid_axes, device)
