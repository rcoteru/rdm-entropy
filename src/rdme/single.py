from __future__ import annotations

from collections import deque

import torch
import tqdm

from rdme.dynamics import (compute_total_input, compute_firing_rate,
                               update_synaptic_term, update_age_distribution,
                               check_age_normalization)
from rdme.epr import (anchor_reverse_synaptic_term, joint_distribution, epr_from_joint,
                          make_epr_indices, init_epr_state, step_epr)
from rdme.fixed_points import fp_overlap_map, fp_state, fp_guess_lattice
from rdme.lyapunov import update_tangent, init_tangent_blocks

from rdme.hazards import HAZARDS
import rdme.fixed_points as fpts
import rdme.kernels as krn

""" The single-system network. See rdme for what the model is; this file is the class. """



# Continuous-time network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RDMNetwork:

    """ Continuous-time mean field on an age grid, solved with du = dt. """

    M: int                    # number of populations
    dt: float                 # time step AND age-bin width, in ms -- a numerical parameter here
    lam0: float               # rate calibration, 1/dt0 with dt0 the discrete model's quantum
    hazard: str               # key into HAZARDS
    Qm: list[int]             # (M,) age bins per population; U_max[i] = Qm[i]*dt
    N: list[int]              # (M,) neurons per population (sets the N-weighting only)
    w: torch.Tensor           # (M, M) weights, in PHYSICAL units -- not divided by dt

    p: list[torch.Tensor]     # age MASS per population, (Qm[i],), sums to 1; density is p/dt
    S: list[torch.Tensor]     # synaptic term, (Qm[i],)
    R: list[torch.Tensor]     # refractory kernel, (Qm[i],)

    E: torch.Tensor           # (M,) external drive
    beta: torch.Tensor        # (M,) gain
    theta: torch.Tensor       # (M,) threshold
    tau_int: torch.Tensor     # (M,) integration time constant (ms)
    tau_ref: torch.Tensor     # (M,) refractory time constant (ms)
    K_ref: torch.Tensor       # (M,) refractory amplitude
    alpha_int: torch.Tensor   # (M,) 1 - exp(-dt/tau_int)

    m: torch.Tensor           # (M,) per-bin firing probability; the rate is m/dt
    device: str
    N_ratios: torch.Tensor    # (M,) population size fractions

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
                 dt: float = 1.0,
                 dt0: float | None = None,
                 hazard: str = "escape",
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 device: str = 'cpu') -> None:
        """ Initialize the continuous-time mean field.

        dt is the resolution of the solution, and is the quantity to refine when checking
        convergence. dt0 is the *model's* time quantum, which fixes lam0 = 1/dt0 and does not
        change when dt does; it is what ties this model to a particular member of the discrete
        family. dt0 defaults to dt, i.e. to sitting at the quantum, where the escape hazard and
        the sigmoid coincide -- convenient as a starting point, but then refining dt alone would
        move the model, so a convergence study must pin dt0 explicitly.

        w is in physical units and is NOT divided by dt (see the module docstring).

        Qm is derived from the slower of the two kernels, as in RDMNetwork, which makes the
        truncation age U_max = -tau*ln(eps) independent of dt. Note this bounds the kernels, not
        the age density: if the firing rate is low enough that appreciable mass survives past
        U_max it piles up in the absorbing last bin, so a low-rate regime wants a larger Qm. """
        assert w.shape == (M, M), f"w must be ({M},{M})"
        for name, seq in [("N", N), ("E", E), ("beta", beta), ("theta", theta),
                          ("tau_int", tau_int), ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            assert len(seq) == M, f"{name} must have length M={M}"
        if hazard not in HAZARDS:
            raise ValueError(f"hazard must be one of {sorted(HAZARDS)}, got {hazard!r}")

        dtype = torch.get_default_dtype()
        self.M, self.dt, self.device = M, dt, device
        self.hazard = hazard
        self.dt0 = dt if dt0 is None else dt0
        self.lam0 = 1.0 / self.dt0
        if HAZARDS[hazard].bounded_step and self.lam0 * dt > 1.0 + 1e-12:
            # Phi = lam0*dt*sigma would exceed 1: an update probability above one
            raise ValueError(
                f"hazard {hazard!r} requires lam0*dt <= 1, got lam0*dt = {self.lam0 * dt:.4g} "
                f"(dt={dt}, dt0={self.dt0}). Refine dt, or raise dt0.")
        self.N = N
        self.w = w.to(device=device, dtype=dtype)

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
            Qm = [krn.q_from_tau(max(self.tau_int[i].item(), self.tau_ref[i].item()), dt, eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        self.R = [krn.refractory_kernel(Qm[i], self.K_ref[i].item(), self.tau_ref[i].item(),
                                         dt=dt, device=device) for i in range(M)]

        # all age mass in the oldest bin, zero integrated field -- as RDMNetwork, so the two
        # start from the same state and a comparison is not reading a difference in the IC
        self.p = [torch.zeros(Qm[i], device=device) for i in range(M)]
        for pi in self.p: pi[-1] = 1.0
        self.S = [torch.zeros(Qm[i], device=device) for i in range(M)]

        self.m = torch.zeros(M, device=device)

    # Observables
    # ~~~~~~~~~~~

    @torch.inference_mode()
    def rates(self) -> torch.Tensor:
        """ Population firing rates r = m/dt, in 1/ms. The dt-invariant observable: compare
        these across resolutions, never the per-bin overlaps. """
        return self.m / self.dt

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Per-bin firing probabilities m, shape (M,). Provided because this is what
        RDMNetwork reports; it is O(dt) and so not comparable across resolutions. """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ N-weighted mean network rate (scalar, 1/ms). """
        return (self.N_ratios * self.rates()).sum()

    @torch.inference_mode()
    def age_density(self) -> list[torch.Tensor]:
        """ q = p/dt, the age density of the transport equations. Integrates to 1 over age;
        unlike the mass p, it has a dt -> 0 limit. """
        return [p_i / self.dt for p_i in self.p]

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I^a = sum_b w^{ab} r^b + E^a, shape (M,). Built from the RATE, which is why w
        carries no 1/dt here. """
        return compute_total_input(self.rates(), self.w, self.E)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ h^a(u) = S^a(u) + R^a(u) - theta^a, list of (Qm[i],). """
        return [self.S[i] + self.R[i] - self.theta[i] for i in range(self.M)]

    @torch.inference_mode()
    def hazard_rate(self) -> list[torch.Tensor]:
        """ The firing rate lambda(h) the hazard represents, in 1/ms, list of (Qm[i],).

        The quantity the three hazards actually differ by in the dt -> 0 limit; the per-bin
        Phi below is its rendering on this grid. For "synchronous" this is sigma(beta*h)/dt,
        which has no limit -- see sync_rate. """
        fn, h = HAZARDS[self.hazard].rate, self.field()
        return [fn(h[i], self.beta[i], self.lam0, self.dt) for i in range(self.M)]

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Per-bin hazard Phi, list of (Qm[i],). """
        fn, h = HAZARDS[self.hazard].phi, self.field()
        return [fn(h[i], self.beta[i], self.lam0, self.dt) for i in range(self.M)]

    @torch.inference_mode()
    def d_firing_prob(self) -> list[torch.Tensor]:
        """ dPhi/dh at the current state, list of (Qm[i],). Needed by the tangent map. """
        fn, h = HAZARDS[self.hazard].dphi, self.field()
        return [fn(h[i], self.beta[i], self.lam0, self.dt) for i in range(self.M)]

    # Fixed points
    # ~~~~~~~~~~~~

    def _fp_args(self) -> tuple:
        return (self.w, self.E, self.alpha_int, self.beta, self.theta, self.R,
                HAZARDS[self.hazard], self.lam0, self.dt)

    def fp_residual(self, m: torch.Tensor) -> torch.Tensor:
        """ G(m) = m - F(m), in OVERLAPS (per-bin), shape (M,).

        In overlaps rather than rates so the residual sits on the same [0, 1] scale the generic
        Newton routines in rdme.fixed_points clamp to. Divide a root by dt for the rate. """
        return m - fp_overlap_map(m, *self._fp_args())

    def fp_state(self, m: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """ The stationary (p, S) implied by the overlaps m. """
        return fp_state(m, *self._fp_args())

    def set_state(self, m: torch.Tensor) -> None:
        """ Place the network on the stationary state implied by m. """
        p, S = self.fp_state(m)
        self.p = [t.clone() for t in p]
        self.S = [t.clone() for t in S]
        self.m = torch.as_tensor(m, device=self.device).clone().reshape(self.M)

    def fixed_points(self,
                     method: str = 'auto',
                     guesses: torch.Tensor | None = None,
                     n_grid: int = 201,
                     tol: float = 1e-10,
                     merge_tol: float = 1e-6,
                     **kwargs) -> torch.Tensor:     # (R, M) roots, one row per fixed point
        """ All fixed points, as overlap vectors.

        method='bracket' scans [0, 1] for sign changes of G and bisects them; it is exhaustive
        up to the grid resolution and is the default for M=1, where it reliably picks up the
        unstable middle branch inside a bistable region. method='newton' runs Newton from a
        lattice of starting points and is the default for M>1, where no scan is affordable; it
        finds whatever the basins reach, so a root missing from its output is not proof that
        none exists. 'auto' picks by M.

        Roots come back unclassified -- pass each to fp_eigenvalues (or is_stable) to sort the
        stable branches from the unstable ones. In overlaps, not rates: the scan and the Newton
        clamp both live on [0, 1], and at small dt that is a very coarse net over a root sitting
        at O(dt), so pass `guesses` explicitly when refining. """
        if method == 'auto':
            method = 'bracket' if self.M == 1 else 'newton'

        if method == 'bracket':
            assert self.M == 1, "bracketing is a scalar method; use method='newton' for M > 1"
            return fpts.bracket_roots(self.fp_residual, n_grid=n_grid,
                                      merge_tol=merge_tol, device=self.device, **kwargs)
        if method == 'newton':
            if guesses is None:
                guesses = fp_guess_lattice(self.M, n_per_axis=5, device=self.device)
            return fpts.newton_roots(self.fp_residual, guesses, tol=tol,
                                     merge_tol=merge_tol, **kwargs)
        raise ValueError(f"unknown method {method!r}, expected 'auto', 'bracket' or 'newton'")

    # State vector, for the Jacobian and the tangent dynamics
    # ~~~~~~~~~~~~

    def flatten_state(self, p: list[torch.Tensor], S: list[torch.Tensor]) -> torch.Tensor:
        """ Stack the per-population (p, S) into one flat vector, interleaved as
        [p_0, S_0, p_1, S_1, ...] -- the same layout as RDMNetwork, so the two are directly
        comparable and lyapunov's block convention carries over. """
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
        out-of-place, so torch.func can trace it.

        m is read back as p[i][..., 0] rather than carried in the state, so the map has no
        spurious zero mode from a redundant variable. """
        p, S = self.unflatten_state(x)
        m = torch.stack([p[i][..., 0] for i in range(self.M)], dim=-1)
        I = compute_total_input(m / self.dt, self.w, self.E)
        fn = HAZARDS[self.hazard].phi
        fprobs = [fn(S[i] + self.R[i] - self.theta[i], self.beta[i], self.lam0, self.dt)
                  for i in range(self.M)]
        p_new = [update_age_distribution(p[i], fprobs[i]) for i in range(self.M)]
        S_new = [update_synaptic_term(S[i], self.alpha_int[i], I[..., i])
                 for i in range(self.M)]
        return self.flatten_state(p_new, S_new)

    def fp_eigenvalues(self, m: torch.Tensor) -> torch.Tensor:
        """ Eigenvalues of the linearized map at the fixed point with overlaps m. """
        x = self.flatten_state(*self.fp_state(m))
        return fpts.jacobian_eigvals(self.state_map, x)

    def is_stable(self, m: torch.Tensor, tol: float = 1e-8) -> bool:
        """ Linear stability of the fixed point with overlaps m: spectral radius < 1. """
        return bool(fpts.spectral_radius(self.fp_eigenvalues(m)) < 1.0 - tol)

    # Tangent dynamics (Lyapunov exponents)
    # ~~~~~~~~~~~~

    def init_tangent(self, k: int,
                     generator: torch.Generator | None = None,
                     ) -> list[torch.Tensor]:   # 2M blocks, each (k, Qm[i])
        """ A random orthonormal, sum-zero tangent basis. Shares mean_field's builder: the
        layout and the sum-zero projection are properties of the state, not of the hazard. """
        return init_tangent_blocks(self.Qm, k, device=self.device, generator=generator)

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
        blocks[:] = update_tangent(blocks, self.p, fprobs, dphi, self.w,
                                   [self.alpha_int[i] for i in range(self.M)], self.dt)
        self._advance_from(I, fprobs)

    # Dynamics
    # ~~~~~~~~

    @torch.inference_mode()
    def _advance_from(self, I: torch.Tensor, fprobs: list[torch.Tensor]) -> None:
        """ Advance (m, p, S) one step given the input and hazards of the CURRENT state.

        Split out so step_tangent can reuse the same I and fprobs for state and tangent. This
        updates p in place, so a second evaluation afterwards would linearize about a state
        that had already moved. """
        # compute_firing_rate rather than (p*fprobs).sum(): same quantity, but reusing the
        # function RDMNetwork calls keeps the summation order identical, so hazard="sigmoid"
        # reproduces it bit for bit instead of to within roundoff
        self.m = torch.stack([compute_firing_rate(self.p[i], fprobs[i]) for i in range(self.M)])

        for i in range(self.M):
            update_age_distribution(self.p[i], fprobs[i], inplace=True)
            update_synaptic_term(self.S[i], self.alpha_int[i], I[i], inplace=True)

        for pi in self.p:
            check_age_normalization(pi)

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

    @torch.inference_mode()
    def trajectory(self, T: int,
                   r_tot: bool = False,
                   q: bool = False,
                   S: bool = False,
                   pb: bool = True,
                   ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """ Run for T steps. Always returns "r" (T, M), the per-population rates in 1/ms, and
        "m" (T, M), the per-bin overlaps, so a run can be compared with RDMNetwork either way.
        Optional: r_tot (T,) the N-weighted rate; q/S as lists of M tensors, each (T, Qm[i]).

        Time axis is in steps; multiply by dt for ms, which is what any comparison across
        resolutions has to do. """
        out: dict = {"r": torch.zeros(T, self.M, device=self.device),
                     "m": torch.zeros(T, self.M, device=self.device)}
        if r_tot: out["r_tot"] = torch.zeros(T, device=self.device)
        if q: out["q"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]
        if S: out["S"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]

        for t in tqdm.tqdm(range(T), disable=not pb):
            out["r"][t] = self.rates()
            out["m"][t] = self.m
            if r_tot: out["r_tot"][t] = self.activity()
            if q:
                for i in range(self.M): out["q"][i][t] = self.p[i] / self.dt
            if S:
                for i in range(self.M): out["S"][i][t] = self.S[i]
            self.update()

        def _cpu(v): return v.cpu() if torch.is_tensor(v) else [t.cpu() for t in v]
        return {k: _cpu(v) for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory(self, T: int,
                           method: str = "sliding",
                           pb: bool = True,
                           ) -> dict[str, torch.Tensor]:
        """ Per-population entropy production over T reported steps.

        method="sliding" (default) runs the O(Q) recursions of step_epr; method="joint" builds
        the explicit (Q, K) joint every step at O(Q^2) and reads the EPR off it. Both are kept:
        the joint path is the oracle -- it is the definition, with nothing to go wrong beyond
        the formula itself -- and the sliding path is what makes long runs and small dt
        affordable. They share epr_from_parts, so a disagreement can only come from the four
        marginals, which is exactly what the cross-check in the tests isolates.

        Both share everything else: the hazard, the ring buffers, the fill phase and the output
        assembly. The branch is three lines inside the loop.

        The reverse field needs the input Qm[i] steps into the FUTURE, so the run opens with a
        fill phase of Q_max = max(Qm) shared ticks that is not reported. Populations share one
        clock but have their own Qm, so a population with Qm[i] < Q_max accumulates its
        leftover snapshots in a FIFO that the main loop keeps pushing to and popping from,
        leaving its tail-writes a fixed Q_max - Qm[i] steps behind.

        Returns CPU tensors: r/m (T, M) rates and overlaps, r_tot/m_tot (T,); sigma/H_fwd/H_rev
        (T, M) and their N-weighted aggregates. sigma is per bin; divide by dt for the rate,
        which is the only one of the three with a continuum limit (see the module docstring).
        """
        if method not in ("sliding", "joint"):
            raise ValueError(f"method must be 'sliding' or 'joint', got {method!r}")

        M, Qm, device = self.M, self.Qm, self.device
        Q_max = max(Qm)
        hz, lam0, dt = HAZARDS[self.hazard], self.lam0, self.dt

        out = {k: torch.zeros(T, M, device=device)
               for k in ("r", "m", "sigma", "H_fwd", "H_rev")}
        out["r_tot"] = torch.zeros(T, device=device)
        out["m_tot"] = torch.zeros(T, device=device)

        p_cur   = [self.p[i].clone() for i in range(M)]
        buff_S  = [torch.zeros(Qm[i], Qm[i], device=device) for i in range(M)]
        buff_I  = [torch.zeros(Qm[i], device=device) for i in range(M)]
        pending = [deque() for _ in range(M)]

        for j in range(Q_max):                      # fill phase, off the reported clock
            I = self.total_input()
            for i in range(M):
                snap = (self.S[i].clone(), I[i].clone())
                if j < Qm[i]: buff_S[i][j], buff_I[i][j] = snap
                else:         pending[i].append(snap)
            self.update()

        # The buffers are rings from here on: logical row j lives at slot (head + j) % Qm[i],
        # so retiring a row is one slot write rather than a (Qm, Qm) shift.
        idx = [make_epr_indices(Qm[i], Qm[i], device) for i in range(M)]
        state = [init_epr_state(p_cur[i], buff_S[i], self.R[i], self.beta[i], self.theta[i],
                                hz, lam0, dt) if method == "sliding" else None
                 for i in range(M)]
        head = 0

        for t in tqdm.tqdm(range(T), disable=not pb):
            for i in range(M):
                # read the activity before either path advances the marginal, so m_t is
                # reported against the sigma of the t -> t+1 transition rather than m_{t+1}
                out["m"][t, i] = p_cur[i][0]

                if method == "sliding":
                    e, p_cur[i], state[i] = step_epr(
                        p_cur[i], state[i], buff_S[i], buff_I[i], idx[i], head,
                        self.R[i], self.alpha_int[i], self.beta[i], self.theta[i],
                        hz, lam0, dt)
                else:
                    sl = idx[i].slots[head % Qm[i]]
                    Sb = buff_S[i][sl]                         # logical order, O(Q^2) gather
                    J = joint_distribution(p_cur[i], Sb, self.R[i], self.beta[i],
                                           self.theta[i], hz, lam0, dt)
                    phi_f = hz.phi(Sb[0] + self.R[i] - self.theta[i],
                                   self.beta[i], lam0, dt)
                    S_rev = anchor_reverse_synaptic_term(buff_I[i][sl], self.alpha_int[i])
                    phi_r = hz.phi(S_rev + self.R[i] - self.theta[i],
                                   self.beta[i], lam0, dt)
                    e = epr_from_joint(J, phi_f, phi_r)
                    p_cur[i] = update_age_distribution(p_cur[i], phi_f)

                out["sigma"][t, i], out["H_fwd"][t, i], out["H_rev"][t, i] = e

            out["r"][t] = out["m"][t] / dt
            out["m_tot"][t] = (self.N_ratios * out["m"][t]).sum()
            out["r_tot"][t] = out["m_tot"][t] / dt

            I = self.total_input()
            for i in range(M):
                pending[i].append((self.S[i].clone(), I[i].clone()))
                S_in, I_in = pending[i].popleft()
                retired = idx[i].slots[head % Qm[i]][0]      # logical row 0 becomes t+Qm[i]
                buff_S[i][retired], buff_I[i][retired] = S_in, I_in
            head += 1
            self.update()

        # rewind the live state to the last reported step. The fill phase and the loop both
        # ran the model past T to keep the look-ahead buffers fed, and that overshoot is not
        # simulation the caller asked for: without this, stepping the model afterwards would
        # continue from T + Q_max and a p(n) read off it would be at the wrong phase.
        for i in range(M):
            self.p[i] = p_cur[i]
            self.S[i] = buff_S[i][idx[i].slots[head % Qm[i]][0]].clone()
        self.m = torch.stack([self.p[i][0] for i in range(M)])

        for k in ("sigma", "H_fwd", "H_rev"):
            out[k + "_tot"] = (out[k] * self.N_ratios).sum(dim=-1)
        return {k: v.cpu() for k, v in out.items()}

    # Stationary reference
    # ~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def stationary_rate(self) -> torch.Tensor:
        """ r* = [int_0^inf Survival(u) du]^{-1} evaluated at the CURRENT field, per population.

        The appendix's fixed-point form of the renewal equation. At a genuine fixed point this
        equals rates(); away from one it does not, so it is a consistency check on a settled
        run rather than a solver. Computed from the same per-bin survival the dynamics uses, so
        at a fixed point the two agree to roundoff rather than to a quadrature error.

        The integral runs to infinity but the grid stops at U_max = Qm*dt, and the difference is
        not negligible: the last bin is absorbing, so neurons that reach it stay there and keep
        firing at the hazard of that bin. That makes the tail an exact geometric series,
        S[Q-1] * sum_j (1 - Phi[Q-1])^j = S[Q-1]/Phi[Q-1], which is added here. Dropping it
        biases r* high by the fraction of mass past the truncation -- at the default Qm that was
        0.9% for a 29 Hz population whose kernels put U_max at 46 ms, since Qm is sized from the
        kernels and not from the firing rate. """
        fprobs = self.firing_prob()
        out = []
        for i in range(self.M):
            # survival to the START of each bin: S[0] = 1, S[n] = prod_{k<n}(1 - Phi_k). The
            # stationary mass of bin n is r*dt*S[n]. Using the end-of-bin survival instead
            # shifts the quadrature by one bin.
            keep = torch.cumprod(1 - fprobs[i], dim=0)
            surv = torch.cat([torch.ones(1, device=keep.device, dtype=keep.dtype), keep[:-1]])
            tail = surv[-1] / fprobs[i][-1].clamp_min(1e-300)   # absorbing bin, geometric
            out.append(1.0 / ((surv[:-1].sum() + tail) * self.dt))
        return torch.stack(out)

# Model factories
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def RDMIsingModel(J: float,
                 E: float,
                 beta: float,
                 theta: float,
                 tau_int: float,
                 tau_ref: float,
                 K_ref: float,
                 dt: float = 1.0,
                 dt0: float | None = None,
                 hazard: str = "escape",
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 device: str = 'cpu',
                 ) -> RDMNetwork:
    """ Single-population (M=1) continuous-time network. J is in physical units: unlike
    RDMIsingModel, do not pass J/dt. """
    return RDMNetwork(M=1, w=torch.tensor([[J]]), N=[1000], E=[E], beta=[beta], theta=[theta],
                     tau_int=[tau_int], tau_ref=[tau_ref], K_ref=[K_ref],
                     dt=dt, dt0=dt0, hazard=hazard, Qm=Qm, eps=eps, device=device)


def RDMWilsonCowan(E_ratio: float,
                  w_EE: float,
                  w_EI: float,
                  w_IE: float,
                  w_II: float,
                  E_exc: float,
                  E_inh: float,
                  beta_E: float,
                  beta_I: float,
                  theta_E: float,
                  theta_I: float,
                  tau_int_E: float,
                  tau_int_I: float,
                  tau_ref_E: float,
                  tau_ref_I: float,
                  K_ref_E: float,
                  K_ref_I: float,
                  dt: float = 1.0,
                  dt0: float | None = None,
                  hazard: str = "escape",
                  Qm: list[int] | None = None,
                  eps: float = 0.01,
                  device: str = 'cpu',
                  ) -> RDMNetwork:
    """ Wilson-Cowan-like two-population (M=2, order [E, I]) continuous-time network.

    Weights are in physical units: pass w_EE, not w_EE/dt. This is the one call-site difference
    from RDMWilsonCowan and the easiest thing to get wrong when porting a parameter set over. """
    N_E = int(E_ratio * 1000)
    N_I = 1000 - N_E
    # row = target, column = source, matching w^{ab} ("from b onto a"), as in RDMWilsonCowan
    w = torch.tensor([[w_EE, w_IE],
                      [w_EI, w_II]])
    return RDMNetwork(M=2, w=w, N=[N_E, N_I], E=[E_exc, E_inh], beta=[beta_E, beta_I],
                     theta=[theta_E, theta_I], tau_int=[tau_int_E, tau_int_I],
                     tau_ref=[tau_ref_E, tau_ref_I], K_ref=[K_ref_E, K_ref_I],
                     dt=dt, dt0=dt0, hazard=hazard, Qm=Qm, eps=eps, device=device)
