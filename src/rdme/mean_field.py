""" The RDM driver: one class for one system and for a batch of them.

RDMNetwork holds state, parameters and the EPR observer's feed, and orchestrates; it writes no
equation of its own -- those are rdme.equations, rdme.epr, rdme.stability and rdme.lyapunov,
all pure functions on tensors with arbitrary leading axes. Batching is the SHAPE of what the
driver holds: w is (M, M) for one system and (B, M, M) for a batch of B, every per-population
parameter (M,) or (B, M), and every age-resolved tensor (M, Q) or (B, M, Q) -- one age grid Q
shared by every population, so the population axis is a tensor axis like any other. `select(b)` slices
a batch down to one system. "One system" means NO leading axis, not B = 1: a B = 1 batch takes
a different reduction path in the backend and its rows differ from a 1-D run in the last bit.

Weights are in physical units and NOT divided by dt -- pass J, never J/dt. The input is built
from the rate r = m/dt, which is what makes a coupling independent of the step; see
equations.compute_total_input, which states the convention once. """

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence

import torch
import tqdm

import rdme.equations as eq
import rdme.hazards as hzd
import rdme.lyapunov as lyap
import rdme.stability as stab
from rdme import epr

# Parameter grids
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# How a sweep becomes a batch, and how its outputs get their axes back.

def flatten_param_grid(
        param_grid: dict[str, float | Sequence[float] | torch.Tensor],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float64,
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


class RDMNetwork:

    """ One RDM network, or a batch of B independent ones sharing M, N and Q. """

    M: int                    # number of populations
    Q: int                    # age bins, shared by every population and the batch
    dt: float                 # solver step AND age-bin width, in ms -- numerics
    nu0: float                # rate unit of the Poisson reference, in 1/ms (fixed at 1)
    t_abs: torch.Tensor       # (*b, M) absolute refractory period, in ms -- a physical dead time
    hazard: str               # key into rdme.hazards.HAZARDS
    
    N: list[int]              # (M,) neurons per population (sets the N-weighting only)
    w: torch.Tensor           # (*b, M, M) weights, PHYSICAL units -- not divided by dt

    m: torch.Tensor           # (*b, M) per-bin firing probability; the rate is m/dt
    p: torch.Tensor           # (*b, M, Q) age MASS per population, sums to 1; density p/dt
    S: torch.Tensor           # (*b, M, Q) synaptic term
    R: torch.Tensor           # (*b, M, Q) refractory kernel, relative part (see blocked)
    blocked: torch.Tensor     # (*b, M, Q) bool, the absolute window: Phi = 0 exactly there

    E: torch.Tensor           # (*b, M) external drive
    beta: torch.Tensor        # (*b, M) gain
    theta: torch.Tensor       # (*b, M) threshold
    tau_int: torch.Tensor     # (*b, M) integration time constant (ms)
    tau_ref: torch.Tensor     # (*b, M) refractory time constant (ms)
    K_ref: torch.Tensor       # (*b, M) refractory amplitude
    alpha_int: torch.Tensor   # (*b, M) 1 - exp(-dt/tau_int)

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
                 t_abs: float | torch.Tensor | list[float] = 1.0,
                 Q: int | None = None,
                 tol: float = 1e-2,
                 ic: str = "silent",
                 age_max: float = 25.0,
                 grid_shape: tuple[int, ...] | None = None,
                 grid_axes: dict[str, torch.Tensor] | None = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float64) -> None:
        """ w is (M, M) for one system or (B, M, M) for a batch; E, beta, theta, tau_int,
        tau_ref, K_ref are then (M,) (a length-M list will do) or (B, M). N is a plain length-M
        list shared by the batch: population sizes are structural, not swept.

        dtype is the model's precision, float64 unless asked otherwise, and independent of
        torch's global default: every parameter and state tensor is cast to it on the way in.
        float64 because sigma is a difference of two entropies of order 1e-2, so float32's
        resolution puts a floor of a few 1e-9 per bin under it -- above the sigma of a cell
        near its fixed point or near onset.

        dt is the SOLVER step and is required: under route B there is no time quantum to
        default to. hazard is required for the same reason. dt and nu0 are scalars shared by a
        batch (see hazards.resolve_timesteps); nu0 is the rate unit, 1 ms^-1.

        t_abs is the physical dead time in ms, a parameter of each population's refractory
        kernel like tau_ref and K_ref: (M,) or (B, M), or a scalar for all of them. It defaults
        to 1 ms -- under an exponential hazard the only thing that bounds the realised rate, at
        1/t_abs. Pass t_abs=0.0 for no rate ceiling, which has to be asked for explicitly. dt
        must divide every t_abs (equations.n_abs_bins), so that no window moves with the step.

        tol sizes the age grid, and is the only thing that does: U_max is the smallest age
        window that holds the truncation's relative firing error below tol at every step, for
        every population (equations.age_window -- a bound on the kernels, the gain and the
        dead-time rate ceiling, with nothing fitted and no dependence on dt), and
        Q = ceil(U_max/dt) + 1. A batch shares one grid, so it takes the largest U_max of any
        population anywhere in it. Q overrides that, for putting two models on one grid
        (comparing a batch element to its single system); truncation_error() then reports what
        the grid costs.

        ic is the initial condition, "silent" or "random", with age_max (ms) the random one's
        age window: the definitions are shared with the spin model, see
        equations.INITIAL_CONDITIONS. set_state replaces it with a stationary state.

        grid_shape/grid_axes describe a batch built from a flattened parameter grid (see
        flatten_param_grid): trajectory outputs then come back with the batch axis expanded
        into the grid axes. Omit both for a plain flat batch or one system. """
        assert w.ndim in (2, 3) and w.shape[-2:] == (M, M), f"w must be ({M},{M}) or (B,{M},{M})"
        lead = tuple(w.shape[:-2])
        assert len(N) == M and all(isinstance(n, int) for n in N), "N must be M integers"
        t_abs = torch.as_tensor(t_abs, dtype=dtype, device=device)
        dt, t_abs, nu0 = hzd.resolve_timesteps(dt, t_abs, nu0, hazard)
        self.nu0 = nu0
        self.t_abs = t_abs.expand(lead + (M,)).clone() if t_abs.ndim == 0 else t_abs
        assert self.t_abs.shape == lead + (M,), f"t_abs must be a scalar or {lead + (M,)}"

        self.M, self.dt, self.device, self.dtype = M, dt, device, dtype
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
        self.alpha_int = eq.tau2alpha(self.tau_int, dt)

        self.n_abs = eq.n_abs_bins(self.t_abs, self.dt)
        self.tol = tol
        self.drive = eq.drive_bound(self.w, self.E, self.t_abs, self.tau_int)
        if Q is None:
            U = eq.age_window(self.beta, self.K_ref, self.tau_ref, self.t_abs, self.tau_int,
                              self.drive, tol)
            Q = max(math.ceil(U.max().item() / dt) + 1, int(self.n_abs.max()) + 1, 2)
        assert isinstance(Q, int) and Q > 0, f"Q must be a positive int, got {Q!r}"
        if Q <= int(self.n_abs.max()):
            # the lumped last bin must be eligible, or its mass could never fire again
            raise ValueError(f"Q={Q} does not reach past the absolute window "
                             f"({int(self.n_abs.max())} blocked bins); need Q > n_abs.")
        self.Q = Q
        self.R = eq.refractory_kernel(Q, self.K_ref[..., None], self.tau_ref[..., None], dt=dt,
                                      t_abs=self.t_abs[..., None], device=device)
        # the absolute window, as a mask on the hazard rather than a field value: see
        # equations.refractory_kernel
        self.blocked = torch.arange(Q, device=device) < self.n_abs[..., None]

        # silent: all age mass in the oldest bin; random: uniform on the first K bins, any
        # past the grid counted in the last one, as SpinModel.fdist does. Zero field either way.
        if ic not in eq.INITIAL_CONDITIONS:
            raise ValueError(f"Unknown initial condition {ic!r}; use one of {eq.INITIAL_CONDITIONS}")
        ages = (torch.tensor([Q - 1], device=device) if ic == "silent" else
                torch.arange(eq.random_age_bins(age_max, dt), device=device).clamp(max=Q - 1))
        mass = torch.bincount(ages, minlength=Q).to(dtype) / len(ages)
        self.p = mass.expand(lead + (M, Q)).clone()
        self.S = torch.zeros(lead + (M, Q), device=device, dtype=dtype)
        self.m = self.p[..., 0].clone()             # the age-0 mass is the cohort firing now

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """ The leading axes: () for one system, (B,) for a batch. """
        return tuple(self.w.shape[:-2])

    @property
    def B(self) -> int | None:
        """ Batch size; None for one system (no leading axis). """
        return self.batch_shape[0] if self.batch_shape else None

    # every tensor with the batch axis leading: what select() slices
    _PER_SYSTEM = ("w", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "alpha_int",
                   "t_abs", "n_abs", "R", "blocked", "p", "S", "m")

    def select(self, b: int) -> RDMNetwork:
        """ Batch element b as a standalone single system (no leading axis).

        Sliced rather than rebuilt, so Q and the refractory kernels are *copied*: a batch
        sizes one Q from its slowest member, and re-deriving it for one element would
        silently give it a different age grid. The state comes along, so the selected model
        sits exactly where the batch does. """
        assert self.B is not None, "select() slices a batch; this is one system already"
        obj = copy.copy(self)
        obj.grid_shape, obj.grid_axes = (), {}
        obj.N = list(self.N)
        for key in self._PER_SYSTEM:
            setattr(obj, key, getattr(self, key)[b].clone())
        return obj

    def unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Expand a leading flat batch axis (B, ...) into the parameter-grid axes.
        Pass-through for one system or a batch that did not come from a grid. Unrelated to
        unflatten_state, which splits a flat state vector into (p, S). """
        if len(self.grid_shape) <= 1:
            return x
        return unflatten_batch(x, self.grid_shape)

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
    def age_density(self) -> torch.Tensor:
        """ q = p/dt, the age density of the transport equations, (*b, M, Q).
        Integrates to 1 over age; unlike the mass p, it has a dt -> 0 limit. """
        return eq.age_density(self.p, self.dt)

    @torch.inference_mode()
    def max_step_prob(self) -> torch.Tensor:
        """ max over bins and populations of Phi at the CURRENT state, per system, (*b,): the
        largest fraction of an age bin that fires in one
        step. Near 1 a bin can empty in a single step, which is the two-step flip.

        ONE INSTANT, which on an oscillating attractor is not the worst case; to answer "is dt
        small enough?" take the max along a trajectory covering at least one period. For
        "bernoulli" Phi <= nu0*dt everywhere; for the route-B kernels the rate is unbounded,
        so nu0*dt bounds nothing and only the realized max(Phi) sees it. The companion
        accuracy criterion is hazards.check_accuracy on max(hazard_rate()). """
        return eq.max_step_prob(self.firing_prob())

    @torch.inference_mode()
    def truncation_error(self) -> torch.Tensor:
        """ Bound on the relative firing error the finite age grid commits at the current
        state, per system, (*b,): the age-window bound of equations.age_window at
        U_max = (Q-1)*dt, with the boundary bin's actual share of the firing in place of its
        worst case. At or below tol whenever Q came from tol; with an explicit Q it is what that
        grid costs. It bounds the error made per step, not its consequence for a long-time
        statistic. """
        delta = eq.field_tail((self.Q - 1) * self.dt, self.K_ref, self.tau_ref, self.t_abs,
                              self.tau_int, self.drive)
        return eq.truncation_error(self.p, self.firing_prob(), self.beta, delta)

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I^a = sum_b w^{ab} r^b + E^a, (*b, M). Built from the RATE, which is why w
        carries no 1/dt here. """
        return eq.total_input(self.m, self.w, self.E, self.dt)

    @torch.inference_mode()
    def field(self) -> torch.Tensor:
        """ h^a(u) = S^a(u) + R^a(u) - theta^a, (*b, M, Q). """
        return eq.field(self.S, self.R, self.theta)

    @torch.inference_mode()
    def hazard_rate(self) -> torch.Tensor:
        """ The firing rate lambda(h) the hazard represents, in 1/ms, (*b, M, Q).
        Bounded by nu0 for "bernoulli", unbounded and exponential for "poisson" /
        "poisson_exact", which share it exactly. """
        return self._per_age(hzd.HAZARDS[self.hazard].rate, self.field())

    @torch.inference_mode()
    def firing_prob(self) -> torch.Tensor:
        """ Per-bin hazard Phi, (*b, M, Q). """
        return self._per_age(hzd.HAZARDS[self.hazard].phi, self.field())

    @torch.inference_mode()
    def d_firing_prob(self) -> torch.Tensor:
        """ dPhi/dh at the current state, (*b, M, Q). Needed by the tangent map. """
        return self._per_age(hzd.HAZARDS[self.hazard].dphi, self.field())

    def _per_age(self, fn: Callable[..., torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        """ A hazard function (phi, dphi or rate) on the field (*b, M, Q), exactly 0 on the
        blocked bins of the absolute window. """
        return eq.per_population(fn, h, self.beta, self.nu0, self.dt).masked_fill(self.blocked,
                                                                                  0.0)

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
        return (self.w, self.E, self.alpha_int, self.beta, self.theta, self.R, self.blocked,
                hzd.HAZARDS[self.hazard], self.nu0, self.dt)

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m), in OVERLAPS (per-bin), (*b, M): on the [0, 1] scale the generic
        Newton routines in rdme.stability clamp to. Divide a root by dt for the rate. """
        return eq.fp_residual(m, *self._fp_args())

    def fp_state(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ The stationary (p, S), each (*b, M, Q), implied by the overlaps m. """
        return eq.fp_state(m, *self._fp_args())

    def set_state(self, m: torch.Tensor) -> None:
        """ Place every system on the stationary state implied by its overlaps m (*b, M). """
        p, S = self.fp_state(m)
        self.p, self.S = p.clone(), S.clone()
        self.m = torch.as_tensor(m, device=self.device, dtype=self.dtype).clone().reshape(self.batch_shape
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
        unstable middle branch inside a bistable region. method='newton' runs Newton and is the
        default for M>1, from stability.default_guesses unless `guesses` are given: a lattice
        over the physical box [0, dt/t_abs]^M, plus, for M = 2 with w(I→I) = 0, a start in every
        sign change of the exact 1-D reduction. Elsewhere it finds whatever the basins reach,
        so a root missing from its output is not proof that none exists.

        Roots come back unclassified -- pass each to leading_modes or leading_growth. In
        overlaps, not rates. """
        assert self.B is None, "fixed_points is per system: use fp_sweep, or select(b)"
        if method == 'auto':
            method = 'bracket' if self.M == 1 else 'newton'

        if method == 'bracket':
            assert self.M == 1, "bracketing is a scalar method; use method='newton' for M > 1"
            return stab.bracket_roots(self.fp_residual, n_grid=n_grid, merge_tol=merge_tol,
                                      device=self.device, dtype=self.dtype, **kwargs)
        if method == 'newton':
            if guesses is None:
                guesses = stab.default_guesses(self)
            return stab.newton_roots(self.fp_residual, guesses.to(self.device, self.dtype),
                                     tol=tol, merge_tol=merge_tol, **kwargs)
        raise ValueError(f"unknown method {method!r}, expected 'auto', 'bracket' or 'newton'")

    def _fp_residual_one(self, m: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """ G(m) for a *single* system, with that system's parameters passed in rather than
        read off self: the form torch.vmap needs to give every Newton trajectory its own
        parameters. Argument order matches fp_sweep's params tuple. """
        return eq.fp_residual(m, *params, hazard=hzd.HAZARDS[self.hazard], nu0=self.nu0,
                              dt=self.dt)

    def fp_sweep(self,
                 guesses: torch.Tensor | None = None,
                 n_per_axis: int = 5,
                 tol: float = 1e-10,
                 res_tol: float | None = None,
                 chunk: int | None = None,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ Newton from K starting points, for every system at once.

        Every (system, starting point) pair is one Newton trajectory and they all run
        simultaneously, so a phase map costs a single vmapped solve -- on the model's device,
        which for a batch should be the GPU (81 x 36 trajectories: 78 s on the CPU, ~1 s on the
        GPU). Its memory grows with the number of trajectories, so they run in chunks of `chunk`
        rows (default: all), halved on out-of-memory (stability.run_chunked). Returns (roots,
        converged), shaped (*b, K, M) and (*b, K); the rows `converged` marks False hold junk.

        Deliberately *not* deduplicated: collapsing K guesses into a variable number of
        distinct roots is ragged per system; stability.deduplicate is there for the systems
        worth inspecting. The default lattice spans the physical box [0, dt/t_abs]^M. """
        if guesses is None:
            guesses = stab.fp_guess_lattice(self.M, n_per_axis, 0.0, stab.overlap_ceiling(self),
                                            device=self.device, dtype=self.dtype)
        guesses = guesses.to(self.device, self.dtype)
        K, lead = guesses.shape[0], self.batch_shape
        B = math.prod(lead)

        def rep(t): return t.reshape((B,) + t.shape[len(lead):]).repeat_interleave(K, dim=0)
        params = (rep(self.w), rep(self.E), rep(self.alpha_int), rep(self.beta),
                  rep(self.theta), rep(self.R), rep(self.blocked))
        m0 = guesses.repeat(B, 1)                                    # (B*K, M)

        def solve(sl: slice) -> torch.Tensor:          # (rows, M + 1): roots and residual
            ps = tuple(q[sl] for q in params)
            m = stab.newton_iterate(self._fp_residual_one, m0[sl], ps, tol=tol, **kwargs)
            return torch.cat([m, stab.residual_norms(self._fp_residual_one, m, ps)[:, None]], -1)

        out, _ = stab.run_chunked(solve, B * K, chunk)
        m, res = out[:, :-1], out[:, -1]
        converged = torch.isfinite(res) & (res < (res_tol if res_tol is not None else 1e3 * tol))
        return m.reshape(lead + (K, self.M)), converged.reshape(lead + (K,))

    # State vector, for the Jacobian and the tangent dynamics
    # ~~~~~~~~~~~~

    def flatten_state(self, p: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """ Stack (p, S), each (*b, M, Q), into one flat vector along the last axis,
        interleaved as [p_0, S_0, p_1, S_1, ...] -- lyapunov's basis convention. """
        return torch.stack([p, S], dim=-2).flatten(-3)

    def unflatten_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Inverse of flatten_state: views into x, not copies. """
        y = x.unflatten(-1, (self.M, 2, self.Q))
        return y[..., 0, :], y[..., 1, :]

    def state_map(self, x: torch.Tensor) -> torch.Tensor:
        """ The one-step update on the FULL state, as a pure function -- the thing to
        differentiate. Deliberately not decorated with inference_mode and deliberately
        out-of-place, so torch.func can trace it. m is read back as p[..., 0] rather than
        carried in the state, so the map has no spurious zero mode from a redundant variable. """
        p, S = self.unflatten_state(x)
        I = eq.total_input(p[..., 0], self.w, self.E, self.dt)
        fprobs = self._per_age(hzd.HAZARDS[self.hazard].phi, eq.field(S, self.R, self.theta))
        _, p_new, S_new = eq.advance(p, S, I, fprobs, self.alpha_int, self.blocked)
        return self.flatten_state(p_new, S_new)

    def fp_eigenvalues(self, m: torch.Tensor) -> torch.Tensor:
        """ Eigenvalues of the linearized map at the fixed point with overlaps m (one system). """
        assert self.B is None, "fp_eigenvalues is per system: select(b) first"
        x = self.flatten_state(*self.fp_state(m))
        return stab.jacobian_eigvals(self.state_map, x)

    def is_stable(self, m: torch.Tensor, tol: float = 1e-8) -> bool:
        """ Linear stability of the fixed point with overlaps m: spectral radius < 1. Dense, so
        for small systems and tests; leading_modes / leading_growth for the rest. """
        return bool(stab.spectral_radius(self.fp_eigenvalues(m)) < 1.0 - tol)

    def leading_modes(self, m: torch.Tensor, **kwargs) -> dict:
        """ The leading modes (mu in 1/ms, omega in rad/ms, warm start v) at the fixed point
        with overlaps m, for one system: stability.leading_modes (ARPACK; CPU). """
        assert self.B is None, "leading_modes is per system: select(b) first"
        return stab.leading_modes(self.state_map, self.flatten_state(*self.fp_state(m)),
                                  self.dt, **kwargs)

    def leading_growth(self, m: torch.Tensor, t_ms: float = 1000.0, **kwargs) -> torch.Tensor:
        """ The leading growth rate (1/ms) at the fixed points with overlaps m (*b, M), one per
        system: stability.leading_growth (power iteration; batches on the GPU). Works where the
        leading modes are too degenerate for ARPACK. """
        m = torch.as_tensor(m, device=self.device, dtype=self.dtype)
        return stab.leading_growth(self.state_map, self.flatten_state(*self.fp_state(m)),
                                   self.dt, t_ms, **kwargs)

    # Tangent dynamics (Lyapunov exponents)
    # ~~~~~~~~~~~~

    def init_tangent(self, k: int,
                     generator: torch.Generator | None = None,
                     ) -> torch.Tensor:      # (*b, k, M*2*Q)
        """ A random orthonormal, sum-zero tangent basis for every system. """
        return lyap.init_tangent(self.M, self.Q, k, prefix=self.batch_shape,
                                 device=self.device, dtype=self.dtype, generator=generator)

    @torch.no_grad()
    def step_tangent(self, V: torch.Tensor) -> None:
        """ Advance the state and a tangent basis together by one step, both in place.

        One call, because the tangent is the linearization at the current state and the two
        must share one hazard evaluation. no_grad rather than inference_mode because the
        Lyapunov orthonormalizer updates the basis in place, which inference tensors forbid
        outside inference mode. """
        I      = self.total_input()
        fprobs = self.firing_prob()
        dphi   = self.d_firing_prob()
        V.copy_(lyap.tangent_step(V, self.p, fprobs, dphi, self.w, self.alpha_int, self.blocked,
                                   self.dt))
        self._advance_from(I, fprobs)

    # Dynamics
    # ~~~~~~~~

    @torch.inference_mode()
    def _advance_from(self, I: torch.Tensor, fprobs: torch.Tensor) -> None:
        """ Advance (m, p, S) one step given the input and hazards of the CURRENT state.

        Split out so step_tangent can reuse the same I and fprobs for state and tangent: this
        replaces the state, so a second evaluation afterwards would linearize about a state
        that had already moved.

        One system checks its age normalization every step (a NaN guard, in effect: the update
        keeps the mass at 1 by construction); a batch does not, as before the merge, since on
        a GPU the check is a host sync per step. """
        self.m, self.p, self.S = eq.advance(self.p, self.S, I, fprobs, self.alpha_int,
                                            self.blocked)
        if self.B is None:
            eq.check_age_normalization(self.p)

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
            return {'device': 'cpu', 'pin_memory': True, 'dtype': torch.float64}
        return {'device': self.device, 'dtype': torch.float64}

    def _unflatten_out(self, out: dict) -> dict:
        return {k: self.unflatten(v) for k, v in out.items()}

    @torch.inference_mode()
    def trajectory(self, T: int,
                   q: bool = False,
                   S: bool = False,
                   pb: bool = True,
                   ) -> dict[str, torch.Tensor]:
        """ Run for T steps. Always returns "m" (*grid, T, M), the per-bin overlaps, and
        "m_tot" (*grid, T) their N-weighted aggregate; divide by dt for the rate in 1/ms.
        Optional: q/S, each (*grid, T, M, Q). The leading axes are none for one system, (B,) for
        a batch, the grid axes for a batch built from a grid. CPU float64 tensors when the
        model runs on CPU or CUDA. Time axis is in steps; multiply by dt for ms. """
        ok, lead = self._out_device_kwargs(), self.batch_shape
        out: dict = {"m": torch.zeros(lead + (T, self.M), **ok)}
        if q: out["q"] = torch.zeros(lead + (T, self.M, self.Q), **ok)
        if S: out["S"] = torch.zeros(lead + (T, self.M, self.Q), **ok)

        for t in tqdm.tqdm(range(T), disable=not pb):
            out["m"][..., t, :].copy_(self.m, non_blocking=True)
            if q: out["q"][..., t, :, :].copy_(self.p / self.dt, non_blocking=True)
            if S: out["S"][..., t, :, :].copy_(self.S, non_blocking=True)
            self.update()
        out["m_tot"] = (out["m"] * self.N_ratios.to(ok["device"])).sum(dim=-1)
        return self._unflatten_out(out)

    @torch.inference_mode()
    def entropy_trajectory(self, T: int,
                           pb: bool = True,
                           inplace: bool = True,
                           method: str = "sliding",
                           ) -> dict[str, torch.Tensor]:
        """ Per-population entropy production over T reported steps, by the O(Q) observer
        rdme.epr.epr_step. method="joint" swaps in the O(Q^2) joint-distribution oracle
        (rdme.epr.epr_step_joint) it is checked against, through the same loop.

        The reverse field needs the input Q - 1 steps into the FUTURE, so the observer lags the
        model by Q - 1 steps: the model runs T + Q - 1 steps, and the first Q - 1 of them only
        fill the input window. Each observer step is fed the model's live S and input, its
        window's leading edge; it carries its own lagged S, and nothing O(Q^2) persists across
        steps. Every population shares Q and so the lag: one observer step covers all of them. Afterwards the live state is rewound
        to the last reported step, so stepping on continues from T.

        `inplace` lets the hazard and survival chains in rdme.epr reuse their own temporaries;
        bit for bit with the out-of-place oracle (inplace=False), and ~10% faster at large Q.

        Returns m (*grid, T, M), m_tot (*grid, T); sigma/H_fwd/H_rev (*grid, T, M) and
        their N-weighted aggregates. sigma is per bin; divide by dt for the rate, the only one
        of the three with a continuum limit (see the rdme package docstring). """
        observers = {"sliding": (epr.init_epr_state, epr.epr_step),
                     "joint": (epr.init_joint_state, epr.epr_step_joint)}
        return self._observe(T, *observers[method], pb=pb, inplace=inplace)

    @torch.inference_mode()
    def _observe(self, T: int, init: Callable, step: Callable, pb: bool,
                 inplace: bool) -> dict[str, torch.Tensor]:
        """ The entropy_trajectory loop, for any observer (init, step) with epr_step's feed:
        init(p_t, S_t, I_window, *params, inplace) -> state, and
        step(state, S_live, I_new, *params, inplace) -> (state, sigma, H_fwd, H_rev), where
        params is (R, blocked, alpha, beta, theta, hazard, nu0, dt) with the population axis leading the
        age axis, and state is a tuple whose first two entries are the lagged marginal p_t and
        synaptic term S_t. """
        M, Q, lead = self.M, self.Q, self.batch_shape
        ok = self._out_device_kwargs()

        out = {k: torch.zeros(lead + (T, M), **ok) for k in ("m", "sigma", "H_fwd", "H_rev")}

        p0, S0 = self.p.clone(), self.S.clone()
        I_hist = []                                 # I_0 .. I_{Q-2}: init only
        state = None
        prm = (self.R, self.blocked, self.alpha_int[..., None], self.beta[..., None], self.theta[..., None],
               hzd.HAZARDS[self.hazard], self.nu0, self.dt)

        # Model time tau runs T + Q - 1 steps. The observer lags it by Q - 1, so at tau it
        # reports t = tau - Q + 1 and is fed the model's live S and input, which are exactly the
        # window edge S_{t+Q-1}, I_{t+Q-1} it needs.
        for tau in tqdm.tqdm(range(T + Q - 1), disable=not pb):
            I = self.total_input()
            t = tau - (Q - 1)
            if t < 0:
                I_hist.append(I)
            else:
                if state is None:
                    state = init(p0, S0, torch.stack(I_hist, dim=-1), *prm, inplace=inplace)
                    p0 = S0 = None                  # the observer holds them from here on
                    I_hist.clear()
                # read the activity before the observer advances the marginal, so m_t is
                # reported against the sigma of the t -> t+1 transition rather than m_{t+1}
                out["m"][..., t, :].copy_(state[0][..., 0], non_blocking=True)
                state, *e = step(state, self.S, I, *prm, inplace=inplace)
                for k, v in zip(("sigma", "H_fwd", "H_rev"), e):
                    out[k][..., t, :].copy_(v, non_blocking=True)
            self.update()

        N_ratios = self.N_ratios.to(ok["device"])
        out["m_tot"] = (out["m"] * N_ratios).sum(dim=-1)

        # rewind the live state to the last reported step, discarding the look-ahead
        assert state is not None
        self.p, self.S = state[:2]
        self.m = self.p[..., 0]

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
                  nu0: float = 1.0, t_abs: float = 1.0, Q: int | None = None,
                  tol: float = 1e-2, ic: str = "silent", age_max: float = 25.0,
                  device: str = 'cpu', dtype: torch.dtype = torch.float64) -> RDMNetwork:
    """ One single-population (M=1) network. """
    return RDMNetwork(M=1, w=torch.tensor([[J]], dtype=dtype), N=[1000], E=[E], beta=[beta], theta=[theta],
                      tau_int=[tau_int], tau_ref=[tau_ref], K_ref=[K_ref],
                      dt=dt, nu0=nu0, hazard=hazard, t_abs=[t_abs], Q=Q, tol=tol,
                      ic=ic, age_max=age_max, device=device, dtype=dtype)


def RDMIsingModelBatch(J, E, beta, theta, tau_int, tau_ref, K_ref,
                       *, dt: float, hazard: str, nu0: float = 1.0,
                       t_abs: float | torch.Tensor = 1.0,
                       Q: int | None = None, tol: float = 1e-2,
                       ic: str = "silent", age_max: float = 25.0,
                       device: str = 'cpu', dtype: torch.dtype = torch.float64) -> RDMNetwork:
    """ A batch of single-population (M=1) networks. Each parameter may be a scalar or a 1D
    tensor, and the batch is the **outer product** of them all, so passing two vectors gives
    the full 2D sweep with no meshgrid at the call site. Scalars contribute a singleton axis
    that is dropped again on output; the result always has a batch axis, even if B = 1.
    t_abs is a model parameter like the others and may be swept too. """
    flat, grid_shape, axes = flatten_param_grid(
        {"J": J, "E": E, "beta": beta, "theta": theta, "tau_int": tau_int,
         "tau_ref": tau_ref, "K_ref": K_ref, "t_abs": t_abs}, device=device, dtype=dtype)
    col = lambda k: flat[k].unsqueeze(-1)                         # (B, 1)
    return RDMNetwork(
        M=1, w=flat["J"].reshape(-1, 1, 1), N=[1000], E=col("E"), beta=col("beta"),
        theta=col("theta"), tau_int=col("tau_int"), tau_ref=col("tau_ref"),
        K_ref=col("K_ref"), dt=dt, nu0=nu0, hazard=hazard, t_abs=col("t_abs"), Q=Q, tol=tol,
        ic=ic, age_max=age_max, grid_shape=grid_shape, grid_axes=axes, device=device, dtype=dtype)


def RDMWilsonCowan(E_ratio: float, w_EE: float, w_EI: float, w_IE: float, w_II: float,
                   E_exc: float, E_inh: float, beta_E: float, beta_I: float,
                   theta_E: float, theta_I: float, tau_int_E: float, tau_int_I: float,
                   tau_ref_E: float, tau_ref_I: float, K_ref_E: float, K_ref_I: float,
                   *, dt: float, hazard: str, nu0: float = 1.0, t_abs: float = 1.0,
                   Q: int | None = None, tol: float = 1e-2,
                   ic: str = "silent", age_max: float = 25.0,
                   device: str = 'cpu', dtype: torch.dtype = torch.float64) -> RDMNetwork:
    """ One Wilson-Cowan-like two-population (M=2, order [E, I]) network. t_abs is the dead
    time of both populations; build an RDMNetwork directly to give them different ones. """
    N_E = int(E_ratio * 1000)
    # row = target, column = source, matching w^{ab} ("from b onto a")
    w = torch.tensor([[w_EE, w_IE],
                      [w_EI, w_II]], dtype=dtype)
    return RDMNetwork(M=2, w=w, N=[N_E, 1000 - N_E], E=[E_exc, E_inh], beta=[beta_E, beta_I],
                      theta=[theta_E, theta_I], tau_int=[tau_int_E, tau_int_I],
                      tau_ref=[tau_ref_E, tau_ref_I], K_ref=[K_ref_E, K_ref_I],
                      dt=dt, nu0=nu0, hazard=hazard, t_abs=[t_abs, t_abs], Q=Q, tol=tol,
                      ic=ic, age_max=age_max, device=device, dtype=dtype)


def _wilson_cowan_from_flat(flat: dict[str, torch.Tensor], E_ratio: float, dt: float,
                            nu0, hazard, Q, tol, ic, age_max, grid_shape, grid_axes,
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
        dt=dt, nu0=nu0, hazard=hazard, t_abs=two("t_abs", "t_abs"), Q=Q, tol=tol,
        ic=ic, age_max=age_max, grid_shape=grid_shape, grid_axes=grid_axes, device=device, dtype=w.dtype)


_WC_NAMES = ["w_EE", "w_EI", "w_IE", "w_II", "E_exc", "E_inh", "beta_E", "beta_I",
             "theta_E", "theta_I", "tau_int_E", "tau_int_I", "tau_ref_E", "tau_ref_I",
             "K_ref_E", "K_ref_I"]


def RDMWilsonCowanBatch(E_ratio, w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                        theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                        K_ref_E, K_ref_I, *, dt: float, hazard: str,
                        nu0: float = 1.0, t_abs: float | torch.Tensor = 1.0,
                        Q: int | None = None, tol: float = 1e-2,
                        ic: str = "silent", age_max: float = 25.0, device: str = 'cpu',
                        dtype: torch.dtype = torch.float64) -> RDMNetwork:
    """ A batch of two-population (M=2, order [E, I]) networks, the batch being the **outer
    product** of every argument. Scalars contribute a singleton axis that is dropped again on
    output. E_ratio is structural (it sets the population sizes) and stays a scalar. t_abs,
    the dead time of both populations, may be swept like the rest. """
    vals = dict(zip(_WC_NAMES, [w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                                theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                                K_ref_E, K_ref_I])) | {"t_abs": t_abs}
    flat, grid_shape, axes = flatten_param_grid(vals, device=device, dtype=dtype)
    return _wilson_cowan_from_flat(flat, E_ratio, dt, nu0, hazard, Q, tol, ic, age_max,
                                   grid_shape, axes, device)


def RDMWilsonCowanPoints(E_ratio, w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                         theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                         K_ref_E, K_ref_I, *, dt: float, hazard: str,
                         nu0: float = 1.0, t_abs: float | torch.Tensor = 1.0,
                         Q: int | None = None,
                         tol: float = 1e-2, ic: str = "silent", age_max: float = 25.0,
                         grid_shape=None, grid_axes=None,
                         device: str = 'cpu', dtype: torch.dtype = torch.float64) -> RDMNetwork:
    """ One parameter point per batch element -- the **elementwise** counterpart of
    RDMWilsonCowanBatch, for search axes that drive two parameters at once (a loop gain g with
    w_EI = g*rho and w_IE = -g/rho would come back from the outer product as the n**2 cross
    product). The caller reparametrises and passes the resulting columns.

    Every parameter may be a scalar or a 1D tensor; scalars and length-1 tensors broadcast
    against the longest, and B is that length. Mismatched lengths are an error rather than a
    broadcast. grid_shape/grid_axes are passed straight through: only the caller knows which
    search axes the flat batch came from. """
    vals = dict(zip(_WC_NAMES, [w_EE, w_EI, w_IE, w_II, E_exc, E_inh, beta_E, beta_I,
                                theta_E, theta_I, tau_int_E, tau_int_I, tau_ref_E, tau_ref_I,
                                K_ref_E, K_ref_I])) | {"t_abs": t_abs}
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
    return _wilson_cowan_from_flat(flat, E_ratio, dt, nu0, hazard, Q, tol, ic, age_max,
                                   grid_shape, grid_axes, device)
