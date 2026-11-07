from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path

import torch
import tqdm

import rdme.rdm as rdm
import rdme.epr as epr
import rdme.hazards as hzd
import rdme.kernels as krn
import rdme.fixed_points as fpts
import rdme.lyapunov as lyap

from rdme.rdm_single import RDMNetwork

""" A batch of independent systems sharing an architecture but not their parameters.

The batched counterpart of rdme.rdm_single. Same model, same hazards, same observables; the
difference is that every parameter carries a leading batch axis and one `update()` advances the
whole batch, so a parameter sweep is one pass rather than B of them.

Everything the batch needs beyond the class itself is already rank-agnostic: rdme.lyapunov,
rdme.fixed_points and
rdme.epr index on trailing axes, so the overlap map, the tangent step and both entropy paths are
the same functions the single system calls, with a (B, ...) argument instead of a (...) one.
That is worth stating because the pre-port implementation could not do it -- its joint build was written for
one system and the batched case goes through torch.vmap with a chunk size. Here the gather
consumes exactly the last two axes, so no vmap is involved and the joint path is available to
the batch too, as the oracle the sliding recursions are checked against.

Weights are in physical units and NOT divided by dt -- pass J, never J/dt. The input is built
from the rate r = m/dt, which is what makes a coupling independent of the step; see
rdm.compute_total_input, which states the convention once for all three model classes. """


# Parameter grids
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# How a sweep becomes a batch, and how its outputs get their axes back. These live here rather
# than in a utility module because the batch axis is the only thing that ever uses them: the
# grid is flattened to build a batch and unflattened to read one.

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


class RDMNetworkBatch:

    """ B independent RDM networks, same M/N/Qm, their own parameters. """

    B: int                    # batch size
    M: int                    # number of populations
    dt: float                 # solver step and age-bin width, in ms
    nu0: float                # rate unit of the Poisson reference, in 1/ms (fixed at 1)
    t_abs: float              # absolute refractory period, in ms -- a physical dead time
    hazard: str               # key into rdme.hazards.HAZARDS
    Qm: list[int]             # (M,) age bins per population, shared across the batch
    N: list[int]              # (M,) neurons per population, shared
    w: torch.Tensor           # (B, M, M) weights, PHYSICAL units

    p: list[torch.Tensor]     # age mass, one (B, Qm[i]) per population
    S: list[torch.Tensor]     # synaptic term, one (B, Qm[i]) per population
    R: list[torch.Tensor]     # refractory kernel, one (B, Qm[i]) per population

    E: torch.Tensor           # (B, M) external drive
    beta: torch.Tensor        # (B, M)
    theta: torch.Tensor       # (B, M)
    tau_int: torch.Tensor     # (B, M)
    tau_ref: torch.Tensor     # (B, M)
    K_ref: torch.Tensor       # (B, M)
    alpha_int: torch.Tensor   # (B, M)

    m: torch.Tensor           # (B, M) per-bin firing probabilities; the rate is m/dt
    device: str

    grid_shape: tuple[int, ...]         # axis sizes whose product is B; (B,) if not a grid
    grid_axes: dict[str, torch.Tensor]  # the swept 1D vectors, in grid order
    N_ratios: torch.Tensor              # (M,) population size fractions

    # Construction
    # ~~~~~~~~~~~~

    def __init__(self, M: int,
                 w: torch.Tensor,
                 N: list[int],
                 E: torch.Tensor,
                 beta: torch.Tensor,
                 theta: torch.Tensor,
                 tau_int: torch.Tensor,
                 tau_ref: torch.Tensor,
                 K_ref: torch.Tensor,
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
        """ w is (B, M, M); E, beta, theta, tau_int, tau_ref, K_ref are (B, M), with B inferred
        from w. N is a plain length-M list, shared across the batch: population sizes are
        structural, not swept.

        dt, nu0 and t_abs are scalars shared by the batch (see hazards.resolve_timesteps):
        a sweep over the rate unit or the dead time is a sweep over *models* and is better
        written as separate batches. dt and hazard are required and keyword-only; nu0 is the
        rate unit, 1 ms^-1, and t_abs the physical dead time, 1 ms by default.

        Qm, if given, is shared. Otherwise each population's bin count comes from the slowest
        (tau_int, tau_ref) pair anywhere in the batch, so one Qm[i] safely represents every
        system -- which also means a batch is only as cheap as its slowest member.

        grid_shape/grid_axes describe a batch built from a flattened parameter grid (see
        shared.flatten_param_grid): trajectory outputs then come back with the batch axis
        expanded into the grid axes. Omit both for a plain flat batch. """
        assert w.ndim == 3 and w.shape[1:] == (M, M), f"w must be (B,{M},{M})"
        B = w.shape[0]
        for name, arr in [("E", E), ("beta", beta), ("theta", theta), ("tau_int", tau_int),
                          ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            assert arr.shape == (B, M), f"{name} must have shape (B={B},M={M})"
        assert len(N) == M and all(isinstance(n, int) for n in N), "N must be M integers"
        dt, t_abs, nu0 = hzd.resolve_timesteps(dt, t_abs, nu0, hazard)
        self.nu0, self.t_abs = nu0, t_abs

        dtype = torch.get_default_dtype()
        self.B, self.M, self.dt, self.device = B, M, dt, device
        self.hazard = hazard

        self.N = N
        self.w = w.to(device=device, dtype=dtype)

        self.grid_shape = tuple(int(s) for s in grid_shape) if grid_shape is not None else (B,)
        n_grid = 1
        for s in self.grid_shape: n_grid *= s
        assert n_grid == B, f"grid_shape {self.grid_shape} has {n_grid} cells, expected B={B}"
        self.grid_axes = dict(grid_axes) if grid_axes is not None else {}

        self.E       = torch.as_tensor(E,       dtype=dtype, device=device)
        self.beta    = torch.as_tensor(beta,    dtype=dtype, device=device)
        self.theta   = torch.as_tensor(theta,   dtype=dtype, device=device)
        self.tau_int = torch.as_tensor(tau_int, dtype=dtype, device=device)
        self.tau_ref = torch.as_tensor(tau_ref, dtype=dtype, device=device)
        self.K_ref   = torch.as_tensor(K_ref,   dtype=dtype, device=device)

        self.N_ratios = torch.as_tensor(N, dtype=dtype, device=device)
        self.N_ratios = self.N_ratios / self.N_ratios.sum()
        self.alpha_int = krn.tau2alpha(self.tau_int, dt)

        if Qm is None:
            Qm = [krn.q_from_tau(max(self.tau_int[:, i].max().item(),
                                      self.tau_ref[:, i].max().item()), dt, eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        # t_abs is a physical dead time (see kernels.refractory_kernel); it is a scalar
        # shared by the batch, as dt is, so n_abs is too. R_abs is NOT -- it is sized per
        # element from that element's weights, so it rides the vmap.
        # abs_refractory_floor reduces w over its last axis and is elementwise in everything
        # else, so (B, M, M) and (B, M) go straight through -- no vmap needed here
        self.n_abs = krn.n_abs_bins(self.t_abs, dt)
        self.R_abs = (krn.abs_refractory_floor(self.w, self.E, self.theta, self.beta,
                                               r_max=1.0 / self.t_abs)
                      if self.t_abs > 0 else torch.zeros_like(self.theta))
        self.R = [
            torch.vmap(lambda K, tau, Ra, i=i: krn.refractory_kernel(
                self.Qm[i], K, tau, dt=dt, t_abs=self.t_abs, R_abs=Ra, device=device)
            )(self.K_ref[:, i], self.tau_ref[:, i], self.R_abs[:, i])
            for i in range(M)
        ]

        self.p = [torch.zeros(B, Qm[i], device=device) for i in range(M)]
        for pi in self.p: pi[:, -1] = 1.0
        self.S = [torch.zeros(B, Qm[i], device=device) for i in range(M)]
        self.m = torch.zeros(B, M, device=device)

    def select(self, b: int) -> RDMNetwork:
        """ Extract batch element b as a standalone single-system RDMNetwork.

        Built by hand rather than through RDMNetwork.__init__ so Qm and the refractory kernels
        are *copied* rather than recomputed: a batch sizes one Qm from its slowest member, and
        re-deriving it for one element would silently give it a different age grid. The state
        comes along, so the selected model sits exactly where the batch does. """
        obj = object.__new__(RDMNetwork)
        obj.M, obj.dt, obj.device = self.M, self.dt, self.device
        obj.nu0, obj.t_abs, obj.hazard = self.nu0, self.t_abs, self.hazard
        obj.n_abs, obj.R_abs = self.n_abs, self.R_abs[b]
        obj.N, obj.Qm = list(self.N), list(self.Qm)
        obj.N_ratios = self.N_ratios
        obj.w = self.w[b]
        for key in ("E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "alpha_int"):
            setattr(obj, key, getattr(self, key)[b])
        obj.R = [R_i[b].clone() for R_i in self.R]
        obj.p = [p_i[b].clone() for p_i in self.p]
        obj.S = [S_i[b].clone() for S_i in self.S]
        obj.m = self.m[b].clone()
        return obj

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Expand a leading flat batch axis (B, ...) into the parameter-grid axes.

        Pass-through when this batch did not come from a grid. Unrelated to
        RDMNetwork.unflatten_state, which splits a flat state vector into (p, S). """
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
        """ Full batch state, parameters and grid layout. """
        data = {k: getattr(self, k) for k in self._SAVE_SCALARS}
        data.update({"Qm": self.Qm, "N": self.N,
                     "grid_shape": self.grid_shape, "grid_axes": self.grid_axes})
        data.update({k: getattr(self, k) for k in self._SAVE_TENSORS})
        data.update({k: getattr(self, k) for k in self._SAVE_LISTS})
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> RDMNetworkBatch:
        """ Load a batch saved with save(), without rebuilding the parameter grid. """
        data = torch.load(path, map_location=device, weights_only=False)
        obj = object.__new__(cls)

        # No legacy branch: saves written under the dt0 convention are not loadable here and
        # are not meant to be. They carry a model (lam0 = 1/dt0, t_abs = dt0) that route B
        # does not have, so back-filling one would invent parameters rather than read them --
        # and every cache written under that convention was deleted with it (B0). A pre-route-B
        # file therefore fails on the missing key, which is the honest outcome: re-run it.
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
        return obj

    @property
    def is_flow(self) -> bool:
        """ True when this model is a sampled FLOW rather than a genuine map.

        The distinction is the caller's half of lyapunov.classify_spectrum, which reads an
        attracting periodic orbit as a fixed point of the P-th iterate in a map and as a zero
        exponent in a flow. Under route B there is no time quantum, so a model is a flow
        whenever dt is a solver step -- which is always, with the single exception of the
        legacy discrete model: "bernoulli" exactly at nu0*dt = 1, where the per-bin kernel is
        the sigmoid and the update is a map in its own right, not a discretization of
        anything. That is what this tests, and it replaces the old `mf.dt < mf.dt0`
        comparison against a parameter the model no longer has. """
        return not (self.hazard == "bernoulli" and abs(self.nu0 * self.dt - 1.0) <= 1e-12)

    # Observables
    # ~~~~~~~~~~~

    @torch.inference_mode()
    def rates(self) -> torch.Tensor:
        """ Firing rates r = m/dt, shape (B, M), in 1/ms. The dt-invariant observable. """
        return self.m / self.dt

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Per-bin firing probabilities m, shape (B, M). O(dt), so not comparable across
        resolutions; rates() is. """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ N-weighted mean network rate per batch element, shape (B,), in 1/ms. """
        return (self.N_ratios * self.rates()).sum(dim=-1)

    @torch.inference_mode()
    def age_density(self) -> list[torch.Tensor]:
        """ q = p/dt, the density of the transport equations, list of (B, Qm[i]). """
        return [p_i / self.dt for p_i in self.p]

    @torch.inference_mode()
    def max_step_prob(self) -> torch.Tensor:   # (B,)
        """ Per batch element, max over bins and populations of Phi at the CURRENT state: the
        largest fraction of an age bin that fires in one step. See RDMNetwork.max_step_prob for
        what it is for -- in a sweep it is also the cheapest way to find the cells where dt has
        stopped resolving the hazard, which are not usually the ones you would have guessed. """
        return torch.stack([phi.flatten(1).max(dim=1).values
                            for phi in self.firing_prob()], dim=0).max(dim=0).values

    @torch.inference_mode()
    def truncation_error(self) -> torch.Tensor:   # (B,)
        """ Rigorous relative bound on the firing-rate error committed by the finite age grid.

            err = p[-1] * |Phi(h_inf) - Phi(h[-1])| / m

        Read it as: the fraction of the population's firing that is misassigned because the
        age axis stops at U_max = Q*dt. `tol = 1e-2` therefore means "the rate is right to 1%",
        which is a statement about the answer rather than about a kernel.

        Why it is a bound and not an estimate. The boundary bin is absorbing: every cohort
        older than U_max is lumped into it and fires at Phi(h[-1]), when cohort u should fire
        at Phi(h(u)). But h(u) = S(u) + R(u) - theta is MONOTONE increasing in age -- the
        synaptic term saturates up to the total input I and the refractory kernel decays up to
        zero -- and Phi is monotone in h. So every true Phi(h(u)) for u > U_max lies between
        Phi(h[-1]) and Phi(h_inf), and the displaced mass p[-1] cannot be wrong by more than
        the gap between them. Nothing here is linearized and no gain is approximated.

        This supersedes two earlier attempts, both wrong in instructive ways:

          - thresholding the boundary MASS p[-1]. A cell at a low-rate fixed point correctly
            keeps nearly all its mass at old ages, so any sweep containing a silent cell fails
            such a check everywhere. On the theta scan the mass stays near 0.94 however far Q
            is pushed, while this error falls through 1e-6.
          - thresholding `p[-1] * beta * |R(U_max)|`. With K_ref = 0 the refractory term
            vanishes identically and the check reports convergence however much mass is
            frozen -- worse than no check. The synaptic tail is what is left.

        Divide-by-zero is not guarded: a population with m = 0 is not firing at all, and the
        age grid cannot be distorting a rate that does not exist. """
        I = self.total_input()                              # (B, M), the asymptotic field
        h = self.field()
        phi = hzd.HAZARDS[self.hazard].phi
        out = []
        for i in range(self.M):
            h_inf = (I[:, i] - self.theta[:, i]).unsqueeze(-1)            # (B, 1)
            b     = self.beta[:, i:i+1]
            gap   = (phi(h_inf, b, self.nu0, self.dt).squeeze(-1)
                     - phi(h[i][:, -1:], b, self.nu0, self.dt).squeeze(-1)).abs()
            out.append(self.p[i][:, -1] * gap / self.m[:, i].clamp_min(1e-300))
        return torch.stack(out, dim=0).max(dim=0).values

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I^a = sum_b w^{ab} r^b + E^a, shape (B, M). Built from the RATE, which is why w
        carries no 1/dt here. """
        return rdm.compute_total_input(self.rates(), self.w, self.E)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ h^a(u) = S^a + R^a - theta^a, list of (B, Qm[i]). """
        return [self.S[i] + self.R[i] - self.theta[:, i:i+1] for i in range(self.M)]

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Per-bin hazard Phi, list of (B, Qm[i]). """
        fn, h = hzd.HAZARDS[self.hazard].phi, self.field()
        return [fn(h[i], self.beta[:, i:i+1], self.nu0, self.dt) for i in range(self.M)]

    @torch.inference_mode()
    def d_firing_prob(self) -> list[torch.Tensor]:
        """ dPhi/dh at the current state, list of (B, Qm[i]). Needed by the tangent map. """
        fn, h = hzd.HAZARDS[self.hazard].dphi, self.field()
        return [fn(h[i], self.beta[:, i:i+1], self.nu0, self.dt) for i in range(self.M)]

    @torch.inference_mode()
    def hazard_rate(self) -> list[torch.Tensor]:
        """ The rate lambda(h) the hazard represents, in 1/ms, list of (B, Qm[i]). """
        fn, h = hzd.HAZARDS[self.hazard].rate, self.field()
        return [fn(h[i], self.beta[:, i:i+1], self.nu0, self.dt) for i in range(self.M)]

    # Fixed points
    # ~~~~~~~~~~~~

    def _fp_args(self) -> tuple:
        return (self.w, self.E, self.alpha_int, self.beta, self.theta, self.R,
                hzd.HAZARDS[self.hazard], self.nu0, self.dt)

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m) for the whole batch at once: m is (B, M), out is (B, M). In
        overlaps, not rates, so the residual sits on the [0, 1] scale the generic Newton
        routines clamp to. """
        return m - fpts.fp_overlap_map(m, *self._fp_args())

    def fp_state(self, m: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ The stationary (p, S) implied by the overlaps m. """
        return fpts.fp_state(m, *self._fp_args())

    def set_state(self, m: torch.Tensor) -> None:
        """ Place every system on the stationary state implied by its overlaps. """
        p, S = self.fp_state(m)
        self.p = [t.clone() for t in p]
        self.S = [t.clone() for t in S]
        self.m = torch.as_tensor(m, device=self.device).clone().reshape(self.B, self.M)

    def _fp_residual_one(self, m: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """ G(m) for a *single* system, with that system's parameters passed in rather than
        read off self: the form torch.vmap needs to give every Newton trajectory its own
        parameters. The hazard, nu0 and dt are closed over instead, being shared by the batch
        and not tensors. Argument order matches fp_sweep's params tuple. """
        w, E, alpha, beta, theta = params[:5]
        R = list(params[5:])
        return m - fpts.fp_overlap_map(m, w, E, alpha, beta, theta, R,
                                  hzd.HAZARDS[self.hazard], self.nu0, self.dt)

    def fp_sweep(self,
                 guesses: torch.Tensor | None = None,
                 n_per_axis: int = 5,
                 tol: float = 1e-10,
                 res_tol: float | None = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ Newton fixed-point search across the whole parameter batch in one call.

        Every (system, starting point) pair is one Newton trajectory and they all run
        simultaneously, so a phase map costs a single vmapped solve rather than B of them.
        Returns (roots, converged), shaped (B, K, M) and (B, K) with K the number of starting
        points; the rows `converged` marks False hold junk.

        Deliberately *not* deduplicated: collapsing K guesses into a variable number of
        distinct roots is a ragged, per-system operation that would force a Python loop over a
        batch that is often tens of thousands of cells. Counting distinct roots is what a
        bistability map wants anyway; fixed_points.deduplicate is there for the individual
        systems worth inspecting.

        The default lattice spans overlaps in [0, 1], which at small dt is a very coarse net
        over a root sitting at O(dt) -- pass `guesses` explicitly when refining. """
        if guesses is None:
            guesses = fpts.fp_guess_lattice(self.M, n_per_axis, device=self.device)
        K = guesses.shape[0]

        def rep(t): return t.repeat_interleave(K, dim=0)
        params = (rep(self.w), rep(self.E), rep(self.alpha_int), rep(self.beta),
                  rep(self.theta), *[rep(R_i) for R_i in self.R])
        m0 = guesses.repeat(self.B, 1)                               # (B*K, M)

        m = fpts.newton_iterate(self._fp_residual_one, m0, params, tol=tol, **kwargs)
        res = fpts.residual_norms(self._fp_residual_one, m, params)
        converged = torch.isfinite(res) & (res < (res_tol if res_tol is not None else 1e3 * tol))
        return m.reshape(self.B, K, self.M), converged.reshape(self.B, K)

    # Dynamics
    # ~~~~~~~~

    @torch.inference_mode()
    def _advance_from(self, I: torch.Tensor, fprobs: list[torch.Tensor]) -> None:
        """ Advance (m, p, S) one step given the input and hazards of the CURRENT state.

        Split out of update() so step_tangent can reuse the same I and fprobs for the state and
        the tangent: the tangent is the linearization *at this state*, so a second evaluation
        after the state had moved would linearize about the wrong point. """
        self.m = torch.stack(
            [rdm.compute_firing_rate(self.p[i], fprobs[i]) for i in range(self.M)], dim=1)
        for i in range(self.M):
            self.p[i] = rdm.update_age_distribution(self.p[i], fprobs[i])
            self.S[i] = rdm.update_synaptic_term(self.S[i], self.alpha_int[:, i:i+1], I[:, i:i+1])

    @torch.inference_mode()
    def update(self) -> None:
        """ One step along the characteristics, across the whole batch. """
        I = self.total_input()          # (B, M), from the OLD rates
        fprobs = self.firing_prob()     # list of (B, Qm[i]), from the OLD S/R
        self._advance_from(I, fprobs)

    @torch.inference_mode()
    def forward(self, T: int, pb: bool = True) -> None:
        for _ in tqdm.tqdm(range(T), disable=not pb):
            self.update()

    # Tangent dynamics (Lyapunov exponents)
    # ~~~~~~~~~~~~

    def init_tangent(self, k: int,
                     generator: torch.Generator | None = None,
                     ) -> list[torch.Tensor]:   # 2M blocks, each (B, k, Qm[i])
        """ A random orthonormal, sum-zero tangent basis for the whole batch. """
        return lyap.init_tangent_blocks(self.Qm, k, prefix=(self.B,),
                                   device=self.device, generator=generator)

    @torch.no_grad()
    def step_tangent(self, blocks: list[torch.Tensor]) -> None:
        """ Advance the state and a tangent basis together by one step, in place.

        no_grad rather than inference_mode: the Lyapunov orthonormalizer updates these blocks
        in place, which inference tensors forbid outside inference mode. """
        I      = self.total_input()
        fprobs = self.firing_prob()
        dphi   = self.d_firing_prob()
        blocks[:] = lyap.update_tangent(
            blocks, self.p, fprobs, dphi, self.w,
            [self.alpha_int[:, i].reshape(-1, 1, 1) for i in range(self.M)], self.dt)
        self._advance_from(I, fprobs)

    # Trajectories
    # ~~~~~~~~~~~~

    def _out_device_kwargs(self) -> dict:
        """ Allocation kwargs for trajectory accumulators. These grow linearly with T, so on
        long runs they can rival the fixed-size compute buffers; pinning them in CPU memory and
        streaming each step in via non_blocking copies keeps that growth off the GPU. No-op on
        a CPU device. """
        if torch.device(self.device).type == 'cuda':
            return dict(device='cpu', pin_memory=True, dtype=torch.float64)
        return dict(device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def trajectory(self, T: int,
                   r_tot: bool = False,
                   q: bool = False,
                   S: bool = False,
                   pb: bool = True,
                   ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """ Run for T steps. Always returns "r" and "m" (*grid, T, M), the rates in 1/ms and
        the per-bin overlaps. Optional: r_tot (*grid, T); q/S as lists of M tensors, each
        (*grid, T, Qm[i]). Leading grid axes are (B,) unless this batch came from a grid. """
        out_kwargs = self._out_device_kwargs()
        out: dict = {"r": torch.zeros(self.B, T, self.M, **out_kwargs),
                     "m": torch.zeros(self.B, T, self.M, **out_kwargs)}
        if r_tot: out["r_tot"] = torch.zeros(self.B, T, **out_kwargs)
        if q: out["q"] = [torch.zeros(self.B, T, self.Qm[i], **out_kwargs) for i in range(self.M)]
        if S: out["S"] = [torch.zeros(self.B, T, self.Qm[i], **out_kwargs) for i in range(self.M)]

        for t in tqdm.tqdm(range(T), disable=not pb):
            out["r"][:, t].copy_(self.rates(), non_blocking=True)
            out["m"][:, t].copy_(self.m, non_blocking=True)
            if r_tot: out["r_tot"][:, t].copy_(self.activity(), non_blocking=True)
            if q:
                for i in range(self.M):
                    out["q"][i][:, t].copy_(self.p[i] / self.dt, non_blocking=True)
            if S:
                for i in range(self.M): out["S"][i][:, t].copy_(self.S[i], non_blocking=True)
            self.update()

        def _uf(v): return self.unflatten(v) if torch.is_tensor(v) else [self.unflatten(t) for t in v]
        return {k: _uf(v) for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory(self, T: int,
                           method: str = "sliding",
                           pb: bool = True,
                           chunk: int | None = None,
                           inplace: bool = False,
                           ) -> dict[str, torch.Tensor]:
        """ Per-population entropy production over T reported steps, for the whole batch.

        method="sliding" (default) runs the O(B*Q) recursions; method="joint" builds the
        explicit (B, Q, K) distribution every step at O(B*Q^2) and reads the EPR off it. As in
        the single-system class the joint path is the oracle and the sliding path is what makes
        long runs affordable -- but here the difference is memory as much as time, since the
        joint is the only (B, Q, Q) object either path would ever allocate.

        `chunk` caps how many batch elements go through the joint at once, trading speed for
        peak memory. The joint path builds one every step; the sliding path builds exactly one,
        in init_epr_state, and then never again -- but that single (B, Q, Q) build is the peak
        of the whole run, so `chunk` caps it too rather than being ignored there. Results are
        unaffected either way: every operation in the joint is elementwise or reduces over the
        trailing age axes, so the batch axis carries no information across elements. (Not
        bit-for-bit, mind: a reduction over a shorter axis may reassociate. `inplace` is the
        one that promises bits.)

        `inplace` is the other memory knob, and the one that matters at large B: it lets the
        hazard and survival chains in rdme.epr reuse their own temporaries instead of
        allocating a fresh (B, Q, Q) buffer per operation, and stops init_epr_state from
        pinning one such buffer for the whole run by returning phi_end as a view into it. The
        default is off, so the out-of-place path stays the oracle: both branches do the same
        arithmetic in the same order and agree bit for bit, which the tests assert directly.

        The reverse field needs the input Qm[i] steps into the FUTURE, so the run opens with a
        fill phase of Q_max = max(Qm) shared ticks that is not reported. Populations share one
        clock but keep their own Qm, so a population with Qm[i] < Q_max accumulates its
        leftover snapshots in a FIFO the main loop keeps pushing to and popping from, holding
        its tail-writes a fixed Q_max - Qm[i] steps behind. No batch element or population ever
        loses, re-reads or skips a timestep.

        Returns CPU tensors: r/m (*grid, T, M), r_tot/m_tot (*grid, T); sigma/H_fwd/H_rev
        (*grid, T, M) and their N-weighted aggregates. sigma is per bin; divide by dt for the
        rate, which is the only one of the three with a continuum limit. """
        if method not in ("sliding", "joint"):
            raise ValueError(f"method must be 'sliding' or 'joint', got {method!r}")

        B, M, Qm, device = self.B, self.M, self.Qm, self.device
        Q_max = max(Qm)
        hz, nu0, dt = hzd.HAZARDS[self.hazard], self.nu0, self.dt
        ok = self._out_device_kwargs()

        out = {k: torch.zeros(B, T, M, **ok) for k in ("r", "m", "sigma", "H_fwd", "H_rev")}
        out["r_tot"] = torch.zeros(B, T, **ok)
        out["m_tot"] = torch.zeros(B, T, **ok)

        p_cur   = [self.p[i].clone() for i in range(M)]
        buff_S  = [torch.zeros(B, Qm[i], Qm[i], device=device) for i in range(M)]
        buff_I  = [torch.zeros(B, Qm[i], device=device) for i in range(M)]
        pending = [deque() for _ in range(M)]

        for j in range(Q_max):                      # fill phase, off the reported clock
            I = self.total_input()
            for i in range(M):
                snap = (self.S[i].clone(), I[:, i].clone())
                if j < Qm[i]: buff_S[i][:, j], buff_I[i][:, j] = snap
                else:         pending[i].append(snap)
            self.update()

        idx = [epr.make_epr_indices(Qm[i], Qm[i], device) for i in range(M)]

        def _init_state(i):
            """ The sliding state for population i, in batch chunks. Each field of EPRState is
            a per-element quantity, so the chunks concatenate back along the batch axis. """
            cs = B if chunk is None else chunk
            parts = [epr.init_epr_state(p_cur[i][sub], buff_S[i][sub], self.R[i][sub],
                                    self.beta[sub, i:i+1], self.theta[sub, i:i+1],
                                    hz, nu0, dt, inplace=inplace)
                     for sub in (slice(lo, min(lo + cs, B)) for lo in range(0, B, cs))]
            return epr.EPRState(*(torch.cat(f, dim=0) for f in zip(*parts)))

        state = [_init_state(i) if method == "sliding" else None for i in range(M)]
        head = 0

        for t in tqdm.tqdm(range(T), disable=not pb):
            for i in range(M):
                out["m"][:, t, i].copy_(p_cur[i][:, 0], non_blocking=True)
                bet, the = self.beta[:, i:i+1], self.theta[:, i:i+1]

                if method == "sliding":
                    e, p_cur[i], state[i] = epr.step_epr(
                        p_cur[i], state[i], buff_S[i], buff_I[i], idx[i], head,
                        self.R[i], self.alpha_int[:, i:i+1], bet, the, hz, nu0, dt,
                        inplace=inplace)
                else:
                    sl = idx[i].slots[head % Qm[i]]
                    # inplace also reorders the gather: buff_S[:, sl] is advanced indexing, so
                    # it COPIES the whole (B, K, Q) ring, defeating the chunking it feeds. Per
                    # chunk it is the same rows in the same order, just Q^2 * cs at a time.
                    Sb = None if inplace else buff_S[i][:, sl]   # logical order, (B, K, Q)
                    S0 = buff_S[i][:, sl[0]]                     # (B, Q) view, row j = 0
                    phi_f = hz.phi(epr.hazard_field(S0, self.R[i], the, inplace), bet, nu0, dt,
                                   inplace=inplace)
                    S_rev = epr.anchor_reverse_synaptic_term(buff_I[i][:, sl],
                                                         self.alpha_int[:, i:i+1])
                    phi_r = hz.phi(epr.hazard_field(S_rev, self.R[i], the, inplace), bet, nu0, dt,
                                   inplace=inplace)
                    # the joint is the one (B, Q, Q) object here, so it is what `chunk` splits
                    cs = B if chunk is None else chunk
                    parts = []
                    for lo in range(0, B, cs):
                        sub = slice(lo, min(lo + cs, B))
                        Sb_sub = buff_S[i][sub][:, sl] if inplace else Sb[sub]
                        J = epr.joint_distribution(p_cur[i][sub], Sb_sub, self.R[i][sub],
                                               bet[sub], the[sub], hz, nu0, dt,
                                               inplace=inplace)
                        parts.append(epr.epr_from_joint(J, phi_f[sub], phi_r[sub]))
                        del J, Sb_sub
                    e = torch.cat(parts, dim=0)
                    p_cur[i] = rdm.update_age_distribution(p_cur[i], phi_f)

                out["sigma"][:, t, i].copy_(e[:, 0], non_blocking=True)
                out["H_fwd"][:, t, i].copy_(e[:, 1], non_blocking=True)
                out["H_rev"][:, t, i].copy_(e[:, 2], non_blocking=True)

            out["r"][:, t] = out["m"][:, t] / dt
            out["m_tot"][:, t] = (out["m"][:, t] * self.N_ratios.to(ok["device"])).sum(dim=-1)
            out["r_tot"][:, t] = out["m_tot"][:, t] / dt

            # read the tail from the CURRENT (pre-update) live state, THEN advance: this
            # ordering, not update-then-read, is what avoids a timestep skip
            I = self.total_input()
            for i in range(M):
                pending[i].append((self.S[i].clone(), I[:, i].clone()))
                S_val, I_val = pending[i].popleft()
                retired = idx[i].slots[head % Qm[i]][0]   # logical row 0 becomes t + Qm[i]
                buff_S[i][:, retired], buff_I[i][:, retired] = S_val, I_val
            head += 1
            self.update()

        # rewind the live state to the last reported step, discarding the lookahead
        for i in range(M):
            self.p[i] = p_cur[i]
            self.S[i] = buff_S[i][:, idx[i].slots[head % Qm[i]][0]].clone()
        self.m = torch.stack([self.p[i][:, 0] for i in range(M)], dim=1)

        N_ratios = self.N_ratios.to(out["sigma"].device)
        for k in ("sigma", "H_fwd", "H_rev"):
            out[k + "_tot"] = (out[k] * N_ratios).sum(dim=-1)
        return {k: self.unflatten(v) for k, v in out.items()}


# Model factories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def RDMIsingModelBatch(J, E, beta, theta, tau_int, tau_ref, K_ref,
                       *, dt: float, hazard: str, nu0: float = 1.0,
                       t_abs: float = 1.0,
                       Qm: list[int] | None = None, eps: float = 0.01,
                       device: str = 'cpu') -> RDMNetworkBatch:
    """ Batched single-population (M=1) network. Each parameter may be a scalar or a 1D tensor,
    and the batch is the **outer product** of them all, so passing two vectors gives the full 2D
    sweep with no meshgrid at the call site. Scalars contribute a singleton axis that is dropped
    again on output.

    J is in physical units: pass J, never J/dt (see rdm.compute_total_input). """
    flat, grid_shape, axes = flatten_param_grid(
        {"J": J, "E": E, "beta": beta, "theta": theta,
         "tau_int": tau_int, "tau_ref": tau_ref, "K_ref": K_ref}, device=device)
    col = lambda k: flat[k].unsqueeze(-1)                         # (B, 1)
    return RDMNetworkBatch(
        M=1, w=flat["J"].reshape(-1, 1, 1), N=[1000], E=col("E"), beta=col("beta"),
        theta=col("theta"), tau_int=col("tau_int"), tau_ref=col("tau_ref"),
        K_ref=col("K_ref"), dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, Qm=Qm, eps=eps,
        grid_shape=grid_shape, grid_axes=axes, device=device)


def _wilson_cowan_from_flat(flat: dict[str, torch.Tensor], E_ratio: float, dt: float,
                            nu0, hazard, t_abs, Qm, eps, grid_shape, grid_axes,
                            device) -> RDMNetworkBatch:
    """ Assemble a two-population batch from flat (B,) parameter columns. Shared by the two
    Wilson-Cowan factories, which differ only in how they get from their arguments to `flat`:
    outer product in one, elementwise in the other. """
    N_E = int(E_ratio * 1000)
    # transposed as in RDMWilsonCowan: row = target, column = source
    w = torch.stack([
        torch.stack([flat["w_EE"], flat["w_IE"]], dim=-1),
        torch.stack([flat["w_EI"], flat["w_II"]], dim=-1),
    ], dim=1)   # (B, 2, 2)
    two = lambda a, b: torch.stack([flat[a], flat[b]], dim=-1)     # (B, 2)
    return RDMNetworkBatch(
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
                        eps: float = 0.01, device: str = 'cpu') -> RDMNetworkBatch:
    """ Batched two-population (M=2, order [E, I]) network, the batch being the **outer
    product** of every argument: pass w_EE and w_II as vectors and you get the full 2D sweep.
    Scalars contribute a singleton axis that is dropped again on output, so the caller pays no
    shape penalty for what it is not sweeping.

    Weights are in physical units: pass w_EE, not w_EE/dt. E_ratio is structural (it sets the
    population sizes) and stays a scalar. """
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
                         device: str = 'cpu') -> RDMNetworkBatch:
    """ One parameter point per batch element -- the **elementwise** counterpart of
    RDMWilsonCowanBatch.

    The outer product is what you want when the sweep axes *are* the model's parameters. It
    cannot express a search axis that drives two at once: a loop gain g with w_EI = g*rho and
    w_IE = -g/rho would come back as the n**2 cross product with the off-diagonal cells not on
    the intended sweep. Here the vectors are zipped, so the caller does its own
    reparametrisation and passes the resulting columns.

    Every parameter may be a scalar or a 1D tensor; scalars and length-1 tensors broadcast
    against the longest, and B is that length. Mismatched lengths are an error rather than a
    broadcast, since silently recycling a short column is the mistake this exists to prevent.

    grid_shape/grid_axes are passed straight through and are not derived from the arguments --
    only the caller knows which search axes the flat batch came from. """
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
