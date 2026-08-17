from __future__ import annotations

from collections import deque
import torch
import tqdm

import rdme.shared as shrd


# Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_population_input(
        m: torch.Tensor, # (M,) overlaps
        w: torch.Tensor, # (M, M) pop connection weights
        I: torch.Tensor, # (M,) pop external inputs
        ) -> torch.Tensor:
    """ Compute the input current to population m. """
    return w@m+I

def update_integration_variable(
        y: torch.Tensor,               # (Q) integration variable
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        pop_input: float | torch.Tensor  # input current to population
        ) -> torch.Tensor:
    """ Update the integration variable y. """
    y_new = torch.zeros_like(y)
    y_new[1:] = alpha * pop_input + (1 - alpha) * y[:-1]
    return y_new

def update_integration_variable_inplace(
        y: torch.Tensor,                 # (Q) integration variable
        alpha: float | torch.Tensor,     # integration kernel scaling factor
        pop_input: float | torch.Tensor  # input current to population
        ) -> None:
    """ Update the integration variable y in place. """
    # y[:-1] and y[1:] overlap, so the shifted values must land in one fresh
    # buffer before being written back — torch refuses to alias src/dst here.
    # mul()+add_() reuses that one buffer instead of allocating a second one.
    shifted = y[:-1].mul(1 - alpha)
    shifted.add_(alpha * pop_input)
    y[1:] = shifted
    y[0] = 0

def compute_firing_rate(P: torch.Tensor, fprobs: torch.Tensor) -> torch.Tensor:
    """ Firing rate for one population: sum(P * fprobs) over the age axis. """
    return torch.dot(P, fprobs)  # fused multiply-reduce, no elementwise product materialized

def update_age_distribution(P: torch.Tensor, fprobs: torch.Tensor) -> torch.Tensor:
    """ Renewal update: births into bin 0, survival-decay into interior bins,
    remainder absorbed into the last (boundary) bin to keep P normalized. """
    P_new = torch.zeros_like(P)
    P_new[0] = compute_firing_rate(P, fprobs)
    P_new[1:-1] = P[:-2] * (1 - fprobs[:-2])
    P_new[-1] = 1 - P_new[:-1].sum()
    return P_new

def update_age_distribution_inplace(P: torch.Tensor, fprobs: torch.Tensor) -> None:
    """ Renewal update in place: births into bin 0, survival-decay into interior bins,
    remainder absorbed into the last (boundary) bin to keep P normalized. """
    frate = compute_firing_rate(P, fprobs)
    interior = fprobs[:-2].neg().add_(1)  # 1 - fprobs[:-2], reusing one buffer
    interior.mul_(P[:-2])                 # fuse in the multiply, no second buffer
    P[0] = frate
    P[1:-1] = interior
    P[-1] = 1 - frate - interior.sum()

def check_age_normalization(P: torch.Tensor, tol: float = 1e-6) -> None:
    """ Check that the population distribution P is normalized. """
    if not torch.allclose(P.sum(), torch.tensor(1.0, device=P.device), atol=tol):
        raise ValueError(f"Population distribution not normalized: sum={P.sum().item()}")


# Auxiliary functions for entropy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def compute_backward_field(
        pop_input: torch.Tensor,       # (Qm[i],) buffered pop_input[j] = input at time t+j
        alpha: float | torch.Tensor,   # integration kernel scaling factor
        ) -> torch.Tensor:             # (Qm[i],) backward field y^dagger_t(n)
    """ Age-dependent backward (time-reversed) field from a buffered trajectory of future
    population inputs, via a kernel-weighted cumulative sum. """
    device = pop_input.device
    Q = pop_input.shape[-1]
    n_idx = torch.arange(1, Q, device=device).float()
    kernel = alpha * (1.0 - alpha) ** (n_idx - 1)
    return torch.cat([pop_input.new_zeros(1), torch.cumsum(kernel * pop_input[1:], dim=0)])

def compute_joint_distribution(
        p:      torch.Tensor,          # (Q,)   p_t(n) age distribution at time t
        buff_p: torch.Tensor,          # (Q, Q) buff_p[j, n] = p_{t+j}(n), j=0 is time t
        buff_y: torch.Tensor,          # (K, Q) buff_y[j, n] = y_{t+j}(n), j=0 is time t
        eta:    torch.Tensor,          # (Q,)   refractory kernel
        beta:   float | torch.Tensor,
        theta:  float | torch.Tensor,
        ) -> torch.Tensor:             # (Q, K) P_joint[n, k] = p_t(n, n_dagger=k)
    """ Joint p_t(n, n_dagger=k) built from hazards, not density ratios.

    S_t(k|n) = prod_{j<k} (1 - Phi_{t+j}(n+j)) tracks a single cohort even
    after it enters the lumped bin, because it multiplies hazards instead of
    reading merged densities (buff_p is unused). Rows sum to 1 by
    telescoping -- no normalisation needed. """
    Q, device = buff_y.shape[1], buff_y.device
    K = buff_y.shape[0]                       # look-ahead horizon (time axis)
    n_idx = torch.arange(Q, device=device)
    j_idx = torch.arange(K, device=device)

    future_age  = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)
    future_time = j_idx

    y_future = buff_y[future_time.unsqueeze(0).expand(Q, K), future_age]
    phi = torch.sigmoid(beta * (y_future + eta[future_age] - theta))    # (Q, K)

    log_surv = torch.cumsum(torch.log1p(-phi.clamp(max=1 - 1e-12)), dim=1)
    S = torch.cat([torch.ones(Q, 1, device=device, dtype=phi.dtype),
                   torch.exp(log_surv[:, :-1])], dim=1)                 # (Q, K)

    boundary = torch.zeros(K, dtype=torch.bool, device=device)
    boundary[-1] = True
    fpt = torch.where(boundary.unsqueeze(0), S, S * phi)

    return p.unsqueeze(1) * fpt

def compute_epr(
        p:       torch.Tensor,         # (Q,)   forward marginal p_t(n)
        p_joint: torch.Tensor,         # (Q, K) P_joint[n, k] = p_t(n, n_dagger=k)
        y_fwd:   torch.Tensor,         # (Q,)   y_t(n)              forward integrated field
        y_rev:   torch.Tensor,         # (K,)   y^dagger_t(n_dagger) backward integrated field
        eta:     torch.Tensor,         # (Q,)
        beta:    float | torch.Tensor,
        theta:   float | torch.Tensor,
        ) -> torch.Tensor:             # (3,) [sigma, S_fwd, S_bw]; sigma = S_bw - S_fwd
    """ sigma = S_bw - S_fwd, with

        S_fwd = sum_n p(n) softplus(hf(n))   -  sum_n P_joint[n,0] hf(n)
        S_bw  = sum_k p_dagger(k) softplus(hr(k)) - sum_k P_joint[0,k] hr(k)

    The spike weights are indicators, not firing probabilities: the forward
    transition emits s(t+1), which happens iff k == 0; the reverse transition
    emits s(t), which happens iff n == 0.

    Index convention: hf[i] and hr[i] are both evaluated at age i, with
    y(0) = y_dagger(0) = 0. y_rev must be anchored at t+1. """
    softplus = torch.nn.functional.softplus

    hf = beta * (y_fwd + eta - theta)
    hr = beta * (y_rev + eta - theta)

    p_fwd = p_joint.sum(dim=1)   # p_t(n)
    p_rev = p_joint.sum(dim=0)   # p_t(n_dagger)

    S_fwd = torch.sum(p_fwd * softplus(hf)) - torch.sum(p_joint[:, 0] * hf)
    S_bw  = torch.sum(p_rev * softplus(hr)) - torch.sum(p_joint[0, :] * hr)

    return torch.stack([S_bw - S_fwd, S_fwd, S_bw])


# Auxiliary functions for fixed points
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#TODO



# Class for single systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RDMNetwork:

    """ Class representing the RDM network. """

    M: int              # number of populations
    deltaT: float       # time step size, in ms
    Qm: list[int]       # (M,) number of age bins for each population
    N: list[int]        # (M,) number of neurons in each population
    w: torch.Tensor     # (M, M) synaptic weights

    p: list[torch.Tensor]     # age distribution, one (Qm[i],) tensor per population
    y: list[torch.Tensor]     # integration variable, one (Qm[i],) tensor per population
    eta: list[torch.Tensor]   # refractory kernel, one (Qm[i],) tensor per population

    I: torch.Tensor           # (M,) external input current for each population
    beta: torch.Tensor        # (M,) inverse temperature
    theta: torch.Tensor       # (M,) firing threshold
    tau_int: torch.Tensor     # (M,) integration kernel time constant (ms)
    tau_ref: torch.Tensor     # (M,) refractory kernel time constant (ms)
    K_ref: torch.Tensor       # (M,) refractory kernel strength
    alpha_int: torch.Tensor   # (M,) integration kernel scaling factor

    device: str         # device for tensors

    # precomputed quantities
    N_ratios: torch.Tensor      # (M,) population size ratios

    # Construction and initialization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, M: int,
                 w: torch.Tensor,
                 N: list[int],
                 I: torch.Tensor | list[float],
                 beta: torch.Tensor | list[float],
                 theta: torch.Tensor | list[float],
                 tau_int: torch.Tensor | list[float],
                 tau_ref: torch.Tensor | list[float],
                 K_ref: torch.Tensor | list[float],
                 deltaT: float = 1.0,
                 Qm: list[int] | None = None,
                 eps: float = 0.01,
                 device: str = 'cpu') -> None:
        """ Initialize the model with given parameters and default initial conditions. """

        assert w.shape == (M, M), f"w must be ({M},{M})"
        for name, seq in [("N", N), ("I", I), ("beta", beta), ("theta", theta),
                          ("tau_int", tau_int), ("tau_ref", tau_ref), ("K_ref", K_ref)]:
            assert len(seq) == M, f"{name} must have length M={M}"
        assert all(isinstance(n, int) for n in N), "N must be a list of integers."

        # Pin every tensor in this model to the dtype active at construction time
        # (respects torch.set_default_dtype), instead of hardcoding float32 for
        # some tensors while others silently follow the ambient default -- that
        # split made w/I/beta/... float32 and p/y/m float64 under a float64
        # default, and w @ m in population_input() doesn't auto-promote across
        # dtypes like elementwise ops do, so it crashed.
        dtype = torch.get_default_dtype()

        self.M, self.deltaT, self.device = M, deltaT, device
        self.N = N
        self.w = w.to(device=device, dtype=dtype)

        self.I       = torch.as_tensor(I,       dtype=dtype, device=device)
        self.beta    = torch.as_tensor(beta,    dtype=dtype, device=device)
        self.theta   = torch.as_tensor(theta,   dtype=dtype, device=device)
        self.tau_int = torch.as_tensor(tau_int, dtype=dtype, device=device)
        self.tau_ref = torch.as_tensor(tau_ref, dtype=dtype, device=device)
        self.K_ref   = torch.as_tensor(K_ref,   dtype=dtype, device=device)

        self.N_ratios = torch.as_tensor(N, dtype=dtype, device=device)
        self.N_ratios = self.N_ratios / self.N_ratios.sum()

        self.alpha_int = shrd.tau2alpha(self.tau_int, deltaT)

        if Qm is None:
            Qm = [shrd.q_from_tau(max(self.tau_int[i].item(), self.tau_ref[i].item()), eps)
                  for i in range(M)]
        assert len(Qm) == M and all(isinstance(q, int) and q > 0 for q in Qm)
        self.Qm = Qm

        # per-population refractory kernel, precomputed once
        self.eta = [shrd.simple_refractory_kernel(Qm[i], self.K_ref[i].item(), self.tau_ref[i].item(), device=device)
                    for i in range(M)]

        # initial conditions: all age-mass in the oldest bin, zero integrated field
        self.p = [torch.zeros(Qm[i], device=device) for i in range(M)]
        for pi in self.p: pi[-1] = 1.0
        self.y = [torch.zeros(Qm[i], device=device) for i in range(M)]

        self.m = torch.zeros(M, device=device)  # m[i] == p[i][0], starts at 0

    # Observables and derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def overlaps(self) -> torch.Tensor:
        """ Returns the current per-population activity vector m of shape (M,). """
        return self.m

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ Returns the N-weighted mean network activity (scalar). """
        return (self.N_ratios * self.m).sum()

    @torch.inference_mode()
    def population_input(self) -> torch.Tensor:
        """ Computes the cross-population input current, shape (M,). """
        return compute_population_input(self.m, self.w, self.I)

    @torch.inference_mode()
    def field(self) -> list[torch.Tensor]:
        """ Returns the total local field y_i + eta_i per population, list of (Qm[i],) tensors. """
        return [self.y[i] + self.eta[i] for i in range(self.M)]

    @torch.inference_mode()
    def firing_prob(self) -> list[torch.Tensor]:
        """ Computes firing probabilities Phi(beta*(field - theta)) per population, list of (Qm[i],) tensors. """
        fields = self.field()
        return [torch.sigmoid(self.beta[i] * (fields[i] - self.theta[i])) for i in range(self.M)]

    # Dynamics and trajectories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def update(self) -> None:
        """ Perform a single time-step update. """
        pop_input = self.population_input()         # (M,) — from OLD self.m
        fprobs    = self.firing_prob()              # list of (Qm[i],) — from OLD y/eta

        self.m = torch.stack([compute_firing_rate(self.p[i], fprobs[i]) for i in range(self.M)])

        for i in range(self.M):
            update_age_distribution_inplace(self.p[i], fprobs[i])
            update_integration_variable_inplace(self.y[i], self.alpha_int[i], pop_input[i])

        for pi in self.p:
            check_age_normalization(pi)

    @torch.inference_mode()
    def forward(self, T: int) -> None:
        for _ in range(T):
            self.update()

    @torch.inference_mode()
    def trajectory(self, T: int, act: bool = False, p: bool = False, y: bool = False) -> dict:
        """ Run for T steps, returning only the requested quantities.
        Always returns "m" (T, M). Optional keys: act (T,);
        p/y as lists of M tensors, each (T, Qm[i]) — ragged across populations, dense across time. """
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {"m": torch.zeros(T, self.M, device=self.device)}
        if act: out["act"] = torch.zeros(T, device=self.device)
        if p:   out["p"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]
        if y:   out["y"] = [torch.zeros(T, self.Qm[i], device=self.device) for i in range(self.M)]

        for t in tqdm.tqdm(range(T)):
            out["m"][t] = self.m
            if act: out["act"][t] = self.activity()
            if p:
                for i in range(self.M): out["p"][i][t] = self.p[i]
            if y:
                for i in range(self.M): out["y"][i][t] = self.y[i]
            self.update()

        def _cpu(v): return v.cpu() if torch.is_tensor(v) else [t.cpu() for t in v]
        return {k: _cpu(v) for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory(self, T: int) -> dict[str, torch.Tensor]:
        """ Online per-population EPR using O(Qm[i]^2) rolling buffers per population.

        All populations advance on ONE shared clock (self.update() has no per-population
        granularity), but each has its own Qm[i]. The fill phase therefore runs Q_max=max(Qm)
        shared ticks, writing into population i's (Qm[i], Qm[i]) buffer for only the first
        Qm[i] of them; the (Q_max-Qm[i]) leftover snapshots computed for smaller populations
        during this phase are stashed in a small per-population FIFO and drained into the
        main loop's first few tail-writes, so no population ever loses or re-reads a timestep.

        Returns dict of CPU tensors: act (T,) N-weighted network activity;
        sigma/S_fwd/S_rev (T, M) per-population entropy-production decomposition
        (sigma = S_bw - S_fwd, per population); sigma_tot/S_fwd_tot/S_rev_tot (T,) the
        same quantities aggregated across populations via the N_ratios weighting already
        used for activity() — i.e. sigma_tot = sum_i N_ratios[i] * sigma[:, i]. """
        M, Qm, device = self.M, self.Qm, self.device
        Q_max = max(Qm)

        out_act   = torch.zeros(T, device=device)
        out_sigma = torch.zeros(T, M, device=device)
        out_S_fwd = torch.zeros(T, M, device=device)
        out_S_rev = torch.zeros(T, M, device=device)

        buff_p  = [torch.zeros(Qm[i], Qm[i], device=device) for i in range(M)]
        buff_y  = [torch.zeros(Qm[i], Qm[i], device=device) for i in range(M)]
        buff_in = [torch.zeros(Qm[i],        device=device) for i in range(M)]
        pending = [deque() for _ in range(M)]  # leftover fill-phase snapshots, Qm[i] < Q_max

        for j in range(Q_max):
            pop_input = self.population_input()
            for i in range(M):
                snap = (self.p[i].clone(), self.y[i].clone(), pop_input[i].clone())
                if j < Qm[i]:
                    buff_p[i][j], buff_y[i][j], buff_in[i][j] = snap
                else:
                    pending[i].append(snap)
            self.update()

        for t in tqdm.tqdm(range(T)):
            for i in range(M):
                H_rev = compute_backward_field(buff_in[i], self.alpha_int[i])
                P_joint = compute_joint_distribution(
                    buff_p[i][0], buff_p[i], buff_y[i], self.eta[i], self.beta[i], self.theta[i])
                epr = compute_epr(
                    buff_p[i][0], P_joint, buff_y[i][0], H_rev, self.eta[i], self.beta[i], self.theta[i])
                out_sigma[t, i] = epr[0]
                out_S_fwd[t, i] = epr[1]
                out_S_rev[t, i] = epr[2]

                buff_p[i]  = torch.roll(buff_p[i],  shifts=-1, dims=0)
                buff_y[i]  = torch.roll(buff_y[i],  shifts=-1, dims=0)
                buff_in[i] = torch.roll(buff_in[i], shifts=-1, dims=0)

            m_t = torch.stack([buff_p[i][0, 0] for i in range(M)])
            out_act[t] = (self.N_ratios * m_t).sum()

            # read the tail from the CURRENT (pre-update) live state, THEN advance —
            # this ordering (vs. update-then-read) is what avoids a timestep skip
            pop_input = self.population_input()
            for i in range(M):
                if pending[i]:
                    p_val, y_val, in_val = pending[i].popleft()
                else:
                    p_val, y_val, in_val = self.p[i].clone(), self.y[i].clone(), pop_input[i].clone()
                buff_p[i][-1]  = p_val
                buff_y[i][-1]  = y_val
                buff_in[i][-1] = in_val
            self.update()

        # rewind live state to match the last reported timestep, discarding the
        # extra Q_max-step lookahead accumulated in self.p/self.y during buffering
        for i in range(M):
            self.p[i] = buff_p[i][0].clone()
            self.y[i] = buff_y[i][0].clone()
        self.m = torch.stack([self.p[i][0] for i in range(M)])

        return {
            "a_pop":       out_act.cpu(),
            "sigma":     out_sigma.cpu(),
            "S_fwd":     out_S_fwd.cpu(),
            "S_rev":     out_S_rev.cpu(),
            "sigma_tot": (out_sigma * self.N_ratios).sum(dim=1).cpu(),
            "S_fwd_tot": (out_S_fwd * self.N_ratios).sum(dim=1).cpu(),
            "S_rev_tot": (out_S_rev * self.N_ratios).sum(dim=1).cpu(),
        }






def RDMIsingModel(
    J: float,
    I: float,
    beta: float,
    theta: float,
    tau_int: float,
    tau_ref: float,
    K_ref: float,
    dt: float = 1.0,
    eps: float = 0.01,
    device: str = 'cpu',
    ) -> RDMNetwork:
        """ Ising RDMNetwork. """
        w = torch.tensor([[J]])
        return RDMNetwork(
            M=1, w=w, N=[1000], I=[I], beta=[beta], theta=[theta],
            tau_int=[tau_int], tau_ref=[tau_ref], K_ref=[K_ref],
            deltaT=dt, eps=eps, device=device,
        )


def RDMWilsonCowan(
    w_EE: float,
    w_EI: float,
    w_IE: float,
    w_II: float,
    I_E: float,
    I_I: float,
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
    N_E: int,
    N_I: int,
    dt: float = 1.0,
    eps: float = 0.01,
    device: str = 'cpu',
) -> RDMNetwork:
    """ Wilson-Cowan-*like* two-population (M=2, order [E, I]) RDMNetwork. """
    w = torch.tensor([[w_EE, w_EI],
                      [w_IE, w_II]], dtype=torch.float32)
    return RDMNetwork(
        M=2, w=w, N=[N_E, N_I], I=[I_E, I_I], beta=[beta_E, beta_I],
        theta=[theta_E, theta_I], tau_int=[tau_int_E, tau_int_I],
        tau_ref=[tau_ref_E, tau_ref_I], K_ref=[K_ref_E, K_ref_I],
        deltaT=dt, eps=eps, device=device,
    )