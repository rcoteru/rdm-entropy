import torch
import tqdm

import rdme.equations as eq
from rdme.hazards import HAZARDS, resolve_timesteps

# Class for single systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SpinModel:
    """ A non-Markovian spin network, or a batch of B independent replicas of it.

    The replicas share every parameter and differ only in their noise: the batch is for
    seeds, not sweeps. State tensors are (N,) for one network and (B, N) for a batch, and
    every output gains the same leading axis -- the layout RDMNetwork uses for its batch. """

    N: int # number of neurons
    M: int # number of populations
    Nm: torch.Tensor   # (M,) number of neurons in each population, sum(Nm) = N
    dt: float # solver step size, in ms
    nu0: float # rate unit of the Poisson reference, in 1/ms (fixed at 1)
    t_abs: float # absolute refractory period, in ms -- a physical dead time
    hazard: str # key into rdme.hazards.HAZARDS

    s: torch.Tensor # (*b, N) current state
    n: torch.Tensor # (*b, N) neuron ages, dtype long
    S: torch.Tensor # (*b, N) local field state variable, integrated from [D_t-1,...,D_t-Q]
    R: torch.Tensor # (*b, N) refractory state variable

    w: torch.Tensor # (M,M), synaptic weigths between populations
    E: torch.Tensor # (M,), external input current for each population

    beta: torch.Tensor # (M,), inverse temperature for each population
    theta: torch.Tensor # (M,), firing threshold for each population
    tau_int: torch.Tensor # (M,), integration time constant for each population
    tau_ref: torch.Tensor # (M,), refractory time constant for each population
    K_ref: torch.Tensor # (M,), refractory strength for each population

    device: torch.device    # device to store tensors on

    # helper tensor built in constructor
    pop_idx: torch.Tensor   # (N,) population of each neuron; populations are contiguous blocks
    pop_sizes: list[int]    # (M,) Nm as plain ints, the block lengths pop_mean splits on

    # Construction and initialization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, s: torch.Tensor, n: torch.Tensor, S: torch.Tensor, R: torch.Tensor,
             w: torch.Tensor, E: torch.Tensor, beta: torch.Tensor, theta: torch.Tensor,
             tau_int: torch.Tensor, K_ref: torch.Tensor, tau_ref: torch.Tensor, 
             Nm: torch.Tensor, *,
             dt: float, hazard: str, nu0: float = 1.0, t_abs: float = 1.0):

        """ dt is the solver step and hazard the kernel, exactly as in RDMNetwork; both are
        required and keyword-only, and the convention and its checks live in
        hazards.resolve_timesteps. nu0 is the rate unit of the Poisson reference (1 ms^-1) and
        t_abs the physical dead time in ms, which is what bounds the realised rate under the
        exponential route-B kernels.

        w is in PHYSICAL units and is NOT pre-divided by dt, matching RDMNetwork: the input
        is built from the rate m/dt, so the scaling lives in total_input rather than at the
        call site. (It used to be the caller's job, which meant every script wrote J/dt and a
        comparison against the mean field had to carry two weight sets for one network.)

        s, n, S, R are (N,) for one network or (B, N) for a batch of replicas; the batch
        shape is read off them. """

        assert s.ndim in (1, 2), f"s must be (N,) or (B, N), got {tuple(s.shape)}"
        N = s.shape[-1]
        M = Nm.shape[0]
        
        # sanity checks on tensor shapes
        assert n.shape == s.shape and S.shape == s.shape and R.shape == s.shape, \
            "State tensors s, n, S, R must all have the same shape, (N,) or (B, N)."
        assert w.shape == (M, M), f"Synaptic weights w must have shape ({M}, {M}), got {w.shape}."
        assert E.shape == (M,), f"Input vector E must have shape ({M},), got {E.shape}."
        assert beta.shape == (M,), f"beta must have shape ({M},).)"
        assert theta.shape == (M,), f"theta must have shape ({M},)."
        assert tau_int.shape == (M,), f"tau_int must have shape ({M},)."
        assert tau_ref.shape == (M,), f"tau_ref must have shape ({M},)."
        assert K_ref.shape == (M,), f"K_ref must have shape ({M},)."
        assert Nm.sum().item() == N, f"Sum of Nm must equal N={N}, got {Nm.sum().item()}"
        
        dt, t_abs, nu0 = resolve_timesteps(dt, t_abs, nu0, hazard)
        self.nu0, self.t_abs = nu0, t_abs

        self.N, self.M, self.dt = N, M, dt
        self.hazard = hazard
        self.s, self.n, self.S, self.R = s, n, S, R
        self.w, self.E = w, E
        self.theta, self.beta = theta, beta
        self.Nm = Nm
        
        a_int = eq.tau2alpha(tau_int, dt)
        self.tau_int, self.a_int = tau_int, a_int
        a_ref = eq.tau2alpha(tau_ref, dt)
        self.K_ref, self.tau_ref, self.a_ref = K_ref, tau_ref, a_ref
        
        self.device = w.device

        # neuron i belongs to population pop_idx[i]: x[..., pop_idx] expands a population
        # vector to the network, and pop_mean reduces back over the contiguous blocks
        self.pop_idx = torch.repeat_interleave(torch.arange(M, device=self.device),
                                               Nm.to(self.device))
        self.pop_sizes = [int(k) for k in Nm.tolist()]

        # Expand population-level parameters to network level, (N,)
        self.a_int_net = self.a_int[self.pop_idx]
        self.a_ref_net = self.a_ref[self.pop_idx]
        # R just after a spike (age 0). Same convention as equations.refractory_kernel, which
        # evaluates age n at u = (n+1)*dt: R holds at -K_ref through the dead time and decays
        # only once age exceeds n_abs, so the first eligible age carries exactly -K_ref. With
        # t_abs = 0 there is no hold, and age 0 already sits one step into the decay.
        # K_ref = 0 or tau_ref = 0 is no relative term at all.
        K_eff = torch.where(self.tau_ref > 0, self.K_ref, 0.0)
        R_reset = -K_eff * (1 - self.a_ref) if t_abs == 0 else -K_eff
        self.R_reset_net = R_reset[self.pop_idx]
        self.theta_net = self.theta[self.pop_idx]
        self.beta_net = self.beta[self.pop_idx]

        # Absolute refractory window: a mask on the age n, n < n_abs, the per-neuron
        # counterpart of the mean field's RDMNetwork.blocked. Phi = 0 there exactly, and the
        # EPR's log branches take their exact values (Hazard.neg_log_prob). The two must
        # agree, which is what the spin-vs-mean-field tests pin. t_abs is a physical dead
        # time; see equations.refractory_kernel.
        self.n_abs = eq.n_abs_bins(self.t_abs, self.dt)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """ The leading axes: () for one network, (B,) for a batch of replicas. """
        return tuple(self.s.shape[:-1])

    @property
    def B(self) -> int | None:
        """ Batch size; None for one network (no leading axis). """
        return self.batch_shape[0] if self.batch_shape else None

    @classmethod
    def random_start(cls, Nm: torch.Tensor, w: torch.Tensor, E: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    *, dt: float, hazard: str,
                    nu0: float = 1.0, t_abs: float = 1.0,
                    B: int | None = None) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        shape = (N,) if B is None else (B, N)
        #TODO: start this more reasonably, e.g. by sampling from the fixed point distribution for the given parameters
        s = torch.randint(0, 2, shape, device=device).int()
        n = torch.randint(1, 50, shape, device=device).long()
        n = torch.where(s==1, torch.zeros_like(n), n)
        S = torch.zeros(shape, device=device, dtype=torch.float32)
        R = torch.zeros(shape, device=device, dtype=torch.float32)
        return cls(s, n, S, R, w, E, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs)

    @classmethod
    def silent_start(cls, Nm: torch.Tensor, w: torch.Tensor, E: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    *, dt: float, hazard: str,
                    nu0: float = 1.0, t_abs: float = 1.0,
                    B: int | None = None) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        shape = (N,) if B is None else (B, N)
        s = torch.zeros(shape, device=device, dtype=torch.int32)   # update() keeps it int
        n = torch.full(shape, 200, device=device).long()
        S = torch.zeros(shape, device=device, dtype=torch.float32)
        R = torch.zeros(shape, device=device, dtype=torch.float32)
        return cls(s, n, S, R, w, E, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs)
    
    
    # Observables and derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ Computes the activity at current time time, i.e. a_t = (1/N) sum_i s_i(t), (*b,). """
        return torch.mean(self.s.float(), dim=-1, dtype=torch.float32)

    @torch.inference_mode()
    def population_activity(self) -> torch.Tensor:
        """ Computes the activity for each population at current time, i.e. a_t^m = (1/N_m) sum_{i in pop m} s_i(t), (*b, M). """
        return self.pop_mean(self.s.float())

    def pop_mean(self, x: torch.Tensor) -> torch.Tensor:
        """ Average a per-neuron quantity over each population: (..., N) -> (..., M). The
        N-weighted sum of the result is the plain all-neuron mean, which is how the
        per-population and _tot entries of an entropy_trajectory agree.

        A split over the contiguous blocks rather than an index_add_ on pop_idx: with M this
        small, every neuron would add into one of one or two slots, and the contention made
        that ~100x slower than a plain mean at N = 1e5 on both CPU and CUDA. """
        return torch.stack([blk.mean(dim=-1) for blk in x.split(self.pop_sizes, dim=-1)],
                           dim=-1)
                           
    @torch.inference_mode()
    def rates(self) -> torch.Tensor:
        """ Population firing rates r = m/dt, in 1/ms, shape (*b, M). The dt-invariant
        observable: compare these across resolutions, never the per-step activities. """
        return self.population_activity() / self.dt

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I at the current time, I^a_t = sum_b w^{ab} r^b_t + E^a, built from the RATE
        r = m/dt -- which is why w carries no 1/dt here, exactly as in RDMNetwork.

        Row = target, column = source, matching w^{ab} ("from b onto a") in the theory
        and compute_total_input in mean_field. The source-first w_XY names the model
        constructors take are transposed into this layout there, not here. """
        pop_I = self.rates() @ self.w.T + self.E      # (*b, M)
        return pop_I[..., self.pop_idx]               # (*b, N)

    def blocked(self, n: torch.Tensor) -> torch.Tensor:
        """ The absolute window at ages n: True where the neuron cannot fire.

        Takes n as an ARGUMENT rather than reading self, because the entropy production needs
        the same mask on the reverse trajectory's own age -- reconstructed by backward
        recursion, not the live state. Routing both directions through one function is what
        keeps them from drifting apart. """
        return n < self.n_abs

    @torch.inference_mode()
    def field(self) -> torch.Tensor:
        """ S + R. Inside the absolute window R holds -K_ref, which the hazard never reads:
        the window is the mask, not a field value (see blocked). """
        return self.S + self.R

    @torch.inference_mode()
    def firing_prob(self) -> torch.Tensor:
        """ Per-step firing probability P(s_i(t+1)=1) = Phi(beta*(H_i(t)+X_i(t)-theta)), with
        Phi the chosen hazard; exactly 0 inside the absolute window. """
        return self._phi(self.field() - self.theta_net).masked_fill(self.blocked(self.n), 0.0)

    @torch.inference_mode()
    def hazard_rate(self) -> torch.Tensor:
        """ The firing rate lambda(h) the hazard represents, in 1/ms, shape (*b, N).

        The quantity the hazards actually differ by as dt -> 0; firing_prob is its rendering
        on this step. "poisson" and "poisson_exact" share it exactly -- they are one model
        under two discretization maps -- while "bernoulli" has its own, saturating at nu0. """
        return HAZARDS[self.hazard].rate(self.field() - self.theta_net, self.beta_net,
                                         self.nu0, self.dt).masked_fill(self.blocked(self.n), 0.0)

    def _phi(self, h: torch.Tensor) -> torch.Tensor:
        """ The hazard applied to a raw field h = S + R - theta (beta not yet folded in). """
        return HAZARDS[self.hazard].phi(h, self.beta_net, self.nu0, self.dt)

    def _neg_log_prob(self, s: torch.Tensor, hb: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """ -ln p(s | h) for the current hazard, given a field hb that ALREADY carries beta and
        the age n it is evaluated at, for the absolute window: 0 for a silent blocked neuron,
        +inf for a spike from one -- which a realised trajectory never contains, forward or
        reversed, since reversal keeps every ISI >= t_abs.

        beta=1 is passed on because the EPR passes expand the gain into the stored fields once
        rather than per bond; the hazard only ever sees beta*h, so this is the same number. """
        return HAZARDS[self.hazard].neg_log_prob(s, hb, 1.0, self.nu0, self.dt,
                                                 blocked=self.blocked(n))

    @torch.inference_mode()
    def fdist(self, Q: int) -> torch.Tensor:
        """ Firing distribution over neuron ages for each population, (*b, M, Q), each row
        summing to 1 -- the counterpart of the mean field's age mass p. Ages >= Q are
        counted in the last bin. """
        # one flat (population, age) bin per neuron, counted by a scatter: O(N) memory
        idx = self.pop_idx * Q + self.n.clamp(max=Q - 1)                    # (*b, N)
        counts = torch.zeros(self.batch_shape + (self.M * Q,), device=self.device,
                             dtype=torch.float32)
        counts.scatter_add_(-1, idx, torch.ones_like(idx, dtype=torch.float32))
        return counts.view(self.batch_shape + (self.M, Q)) / self.Nm[:, None]

    # Dynamics and trajectories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def update(self) -> None:
        """ Update the state of the system based on firing probabilities. """
        probs = self.firing_prob()
        I = self.total_input()

        # sample new spikes based on probabilities
        # float64 draws: what every run so far has used, through the scripts' global default
        fired = torch.rand(self.s.shape, device=self.device, dtype=torch.float64) < probs
        if self.n_abs:
            # The absolute window, on the draw. probs is already exactly 0 there, and
            # `rand < 0` is never True, so this restates the mask rather than adding to it.
            fired &= ~self.blocked(self.n)
        
        self.s = fired.int() # set new state based on fired neurons
        
        # update neuron ages: if fired, age is 0, else increment age by 1
        self.n = torch.where(fired, torch.zeros_like(self.n), self.n + 1).long()
        
        # update field for next time step: integrate I and reset neurons that just fired
        self.S = (self.a_int_net * I + (1 - self.a_int_net) * self.S).masked_fill_(fired, 0)

        # update refractory state for next time step: reset if fired, else hold through the
        # dead time and decay after it (self.n is already the new age here)
        self.R = torch.where(fired, self.R_reset_net,
                             torch.where(self.n > self.n_abs, (1 - self.a_ref_net) * self.R, self.R))

    @torch.inference_mode()
    def _snapshot(self) -> tuple[torch.Tensor, ...]:
        """ The full evolving state: update() reads and writes exactly these four. """
        return (self.s.clone(), self.n.clone(), self.S.clone(), self.R.clone())

    @torch.inference_mode()
    def _restore(self, snap: tuple[torch.Tensor, ...]) -> None:
        """ Rewind to a snapshot, so a method that had to run past its reported window
        leaves the model where the window ends rather than where the lookahead does. """
        self.s, self.n, self.S, self.R = snap

    @torch.inference_mode()
    def forward(self, T: int) -> None:
        """ Thermalize the system for T steps. """
        for _ in range(T):
            self.update()

    @torch.inference_mode()
    def trajectory(self, T: int, s: bool = False, pot: bool = False, fdist: bool = False,
                   Q: int = 100, pb: bool = True) -> dict[str, torch.Tensor]:
        """Run for T steps, returning only the requested quantities.

        Always returns "m"/"r" (*b, T, M), the per-population activity and rate, and
        "m_tot"/"r_tot" (*b, T) their network aggregates -- the keys RDMNetwork.trajectory
        uses. The leading axes are none for one network and (B,) for a batch. Optional keys:
          s    — full spin state, shape (*b, T, N)
          pot  — membrane potential h = S + R, shape (*b, T, N)
          fdist — firing distribution over ages, shape (*b, T, M, Q)
        All tensors are moved to CPU.
        """
        lead, dev = self.batch_shape, self.device
        out = {"m_tot": torch.zeros(lead + (T,), device=dev, dtype=torch.float64),
               "m": torch.zeros(lead + (T, self.M), device=dev, dtype=torch.float64)}
        if s:   out["s"]   = torch.zeros(lead + (T, self.N), device=dev, dtype=torch.int8)
        if pot: out["pot"] = torch.zeros(lead + (T, self.N), device=dev, dtype=torch.float64)
        if fdist: out["fdist"] = torch.zeros(lead + (T, self.M, Q), device=dev,
                                             dtype=torch.float64)
        for t in tqdm.tqdm(range(T), disable=not pb):
            out["m_tot"][..., t] = self.activity()
            out["m"][..., t, :] = self.population_activity()
            if s:   out["s"][..., t, :]   = self.s
            if pot: out["pot"][..., t, :] = self.field()
            if fdist: out["fdist"][..., t, :, :] = self.fdist(Q)
            self.update()
        out["r"], out["r_tot"] = out["m"] / self.dt, out["m_tot"] / self.dt
        return {k: v.cpu() for k, v in out.items()}
    
    @torch.inference_mode()
    def entropy_trajectory(self, T: int, buffer: int = 100,
                               s: bool = False, pot: bool = False,
                               fdist: bool = False, fields: bool = False,
                                Q: int = 100, pb: bool = True) -> dict[str, torch.Tensor]:
        """Forward/backward EPR trajectory.

        Timing: sigma[t] is a property of the TRANSITION t -> t+1, so both halves
        describe the same bond (t, t+1), following eq. (total-epr) in the model
        write-up. Expanding ln p(Gamma)/p(Gamma^dagger), the reverse path factorises
        as p(s_T) prod_t p(s_t | hr[t+1]): the forward traversal emits s[t+1] from
        hf[t], and the reverse traversal of that same bond emits s[t] from hr[t+1].

        Pairing hr[t+2] with s[t+1] instead -- matching on the realized spike rather
        than on the bond -- sums to the same total EPR up to boundary terms, but
        shifts H_rev one step against H_fwd, which shows up as a one-step offset in
        per-step plots and against RDMNetwork.entropy_trajectory.


        Returns the same dict contract as RDMNetwork.entropy_trajectory: m/r (*b, T, M)
        per-population activity and rate and m_tot/r_tot (*b, T) their network aggregates;
        sigma/H_fwd/H_rev (*b, T, M) the per-population entropy-production decomposition
        (sigma = H_rev - H_fwd) and sigma_tot/H_fwd_tot/H_rev_tot (*b, T) the same
        quantities over the whole network. The leading axes are none for one network and
        (B,) for a batch.

        sigma[t] is the *sampled* log-ratio ln p(Gamma)/p(Gamma^dagger) per
        neuron per step. It fluctuates in sign; only its average is the EP
        rate. Do not substitute sigmoid(hf) for the realized spike -- hr
        depends on future spikes and is correlated with s[t+1], so that
        substitution turns the estimator into a pointwise KL that is
        non-negative by construction and cannot detect reversibility.

        Time origin: index 0 is the state this is called on, so the trajectory lines
        up step-for-step with RDMNetwork.entropy_trajectory called on an equivalent
        state. `buffer` is the TAIL the reverse recursion needs, not a lead-in --
        nothing is discarded at the front, so equilibrate with forward() beforehand.

        Runs T + buffer steps: analysis window | tail. The model is rewound onto the
        end of the window afterwards, so it is left at index T rather than out on the
        tail -- fdist() is then comparable against a mean-field p(n) run for the same T.
        """
        if buffer < 2:
            raise ValueError("buffer must be >= 2; in practice use several "
                             "times max(tau_int, tau_ref).")
        if buffer <= self.n_abs:
            # the reverse age starts at 0 at the tail's end (the reversed path's initial
            # condition is not fixed by the forward one), so a neuron silent through the
            # whole tail is inside the window there; past n_abs steps it never reaches a
            # reported bond
            raise ValueError(f"buffer must exceed the absolute window, n_abs = {self.n_abs}")

        dev = self.device
        L = T + buffer
        lo, hi = 0, T
        # buffers are time-leading internally, (L, *b, N); the time axis is moved behind the
        # batch axes on output
        shp = (L,) + self.batch_shape + (self.N,)

        s_trj = torch.zeros(shp, device=dev, dtype=torch.int8)
        S_fwd = torch.zeros(shp, device=dev, dtype=torch.float32)
        R_fwd = torch.zeros(shp, device=dev, dtype=torch.float32)
        n_fwd = torch.zeros(shp, device=dev, dtype=torch.long)
        I = torch.zeros(shp, device=dev, dtype=torch.float32)
        P_fwd = (torch.zeros((L,) + self.batch_shape + (self.M, Q), device=dev,
                             dtype=torch.float32) if fdist else None)
        end_state = None

        for t in tqdm.tqdm(range(L), desc="Forward pass", disable=not pb):
            s_trj[t] = self.s
            S_fwd[t] = self.S
            R_fwd[t] = self.R
            n_fwd[t] = self.n
            I[t] = self.total_input()
            if P_fwd is not None: P_fwd[t] = self.fdist(Q)
            self.update()
            # the tail past the reported window is lookahead for the reverse recursion,
            # not simulation the caller asked for -- remember where the window ends
            if t + 1 == T: end_state = self._snapshot()

        S_rev = torch.zeros(shp, device=dev, dtype=torch.float32)
        R_rev = torch.zeros(shp, device=dev, dtype=torch.float32)
        # the reverse trajectory's AGE, by the same backward recursion as R_rev: in reversed
        # time a neuron's last spike is its next spike forward, so the age resets on the same
        # mask and increments the other way. Needed only for the absolute refractory window,
        # which is a function of age rather than of R.
        n_rev = torch.zeros(shp, device=dev, dtype=torch.long)

        for t in tqdm.tqdm(range(L - 2, -1, -1), desc="Reverse pass", disable=not pb):
            fired_t = s_trj[t] == 1
            S_rev[t] = ((1 - self.a_int_net) * S_rev[t + 1]
                        + self.a_int_net * I[t + 1]).masked_fill_(fired_t, 0)
            n_rev[t] = (n_rev[t + 1] + 1).masked_fill_(fired_t, 0)
            R_rev[t] = torch.where(fired_t, self.R_reset_net,
                                   torch.where(n_rev[t] > self.n_abs,
                                               (1 - self.a_ref_net) * R_rev[t + 1], R_rev[t + 1]))

        hf = self.beta_net * (S_fwd + R_fwd - self.theta_net)
        hr = self.beta_net * (S_rev + R_rev - self.theta_net)

        # both halves of the bond (t, t+1): forward emits s[t+1] from hf[t], reverse
        # emits s[t] from hr[t+1]
        hf_a = hf[lo:hi]                          # field that generated s[t+1]
        hr_a = hr[lo + 1:hi + 1]                  # reverse field that generated s[t]
        s_next = s_trj[lo + 1:hi + 1].float()     # the realized spike s[t+1]
        s_cur  = s_trj[lo:hi].float()             # the realized spike s[t]

        # -ln p(s | h) under the chosen hazard. For a logistic kernel ("poisson", or
        # "bernoulli" at nu0*dt = 1) this is the -s*h + softplus(h) this used to compute.
        ent_f = self._neg_log_prob(s_next, hf_a, n_fwd[lo:hi])
        ent_r = self._neg_log_prob(s_cur,  hr_a, n_rev[lo + 1:hi + 1])

        # same dict contract as RDMNetwork.entropy_trajectory: per-population (*b, T, M)
        # arrays, plus the N-weighted network aggregates under the _tot suffix
        H_fwd_pop, H_rev_pop = self.pop_mean(ent_f), self.pop_mean(ent_r)

        out = {
            "m":         self.pop_mean(s_cur),
            "m_tot":     s_cur.mean(dim=-1),
            "sigma":     H_rev_pop - H_fwd_pop,
            "H_fwd":     H_fwd_pop,
            "H_rev":     H_rev_pop,
            "sigma_tot": (ent_r - ent_f).mean(dim=-1),
            "H_fwd_tot": ent_f.mean(dim=-1),
            "H_rev_tot": ent_r.mean(dim=-1),
        }
        if s:   out["s"]   = s_trj[lo:hi]
        if pot: out["pot"] = (S_fwd + R_fwd)[lo:hi]
        if fields:
            out["hf"] = hf_a
            out["hr"] = hr_a
        # time leads everything above; move it behind the batch axes, which sit right
        # after it -- a no-op for one network
        nb = len(self.batch_shape)
        out = {k: v.movedim(0, nb) for k, v in out.items()}
        if P_fwd is not None: out["fdist"] = P_fwd[lo:hi].movedim(0, nb)
        out["r"], out["r_tot"] = out["m"] / self.dt, out["m_tot"] / self.dt

        # rewind off the tail, so the live state matches reported index T -- the same
        # contract RDMNetwork.entropy_trajectory honours, and what makes a post-hoc
        # fdist() comparable against the mean field's p(n)
        assert end_state is not None
        self._restore(end_state)

        return {k: v.cpu() for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory_chunked(self, T: int, chunk: int = 2048, overlap: int = 512,
                               store_dtype: torch.dtype = torch.float32,
                               s: bool = False, pot: bool = False,
                               fields: bool = False, fdist: bool = False,
                               Q: int = 100, check_overlap: bool = True,
                               pb: bool = True) -> dict[str, torch.Tensor]:
        """Forward/backward EPR trajectory, computed in overlapping windows.

        Memory is O((chunk + overlap) * N) instead of O(T * N): only a sliding
        window of the trajectory is held, and each window is reduced to
        per-timestep scalars before the next is read in.

        `overlap` must exceed the longest inter-spike interval, not merely a few
        tau_int: the reverse recursion resets exactly at spikes, so a neuron
        that has not fired within the window tail still carries the zero
        boundary condition. Set check_overlap=True to be warned when this bites.

        Timing: as in SpinModel.entropy_trajectory -- sigma[t] describes the
        transition t -> t+1, with hf[t] emitting s[t+1] and hr[t+1] emitting s[t].
        Both halves use a realized spike, so sigma == H_rev - H_fwd identically.

        Time origin: index 0 is the state this is called on, matching
        SpinModel.entropy_trajectory and RDMNetwork.entropy_trajectory, so the three
        line up step-for-step. Nothing is discarded at the front -- equilibrate with
        forward() before the call -- and the model is rewound onto the end of the window
        afterwards, so it is left at index T rather than out past the lookahead.

        Trajectories are stored in `store_dtype` (float32 is ample -- the
        estimator is sampling-noise dominated) while all reductions accumulate
        in float64.

        Returns the same dict contract as RDMNetwork.entropy_trajectory: m/r (*b, T, M)
        per-population activity and rate and m_tot/r_tot (*b, T) their network aggregates;
        sigma/H_fwd/H_rev (*b, T, M) the per-population entropy-production decomposition
        (sigma = H_rev - H_fwd) and sigma_tot/H_fwd_tot/H_rev_tot (*b, T) the same
        quantities over the whole network. The leading axes are none for one network and
        (B,) for a batch.
        """
        if overlap < 4:
            raise ValueError("overlap must be >= 4")
        if overlap <= self.n_abs:
            # see entropy_trajectory: the zero reverse-age boundary must not reach a
            # reported bond
            raise ValueError(f"overlap must exceed the absolute window, n_abs = {self.n_abs}")
        if chunk < overlap:
            raise ValueError("chunk must be >= overlap (the slide would self-overlap)")

        dev, N, M, lead = self.device, self.N, self.M, self.batch_shape
        W = chunk + overlap
        acc = torch.float64

        # windows are time-leading, (W, *b, N); outputs put time behind the batch axes
        s_win = torch.zeros((W,) + lead + (N,), device=dev, dtype=torch.int8)
        S_win = torch.zeros((W,) + lead + (N,), device=dev, dtype=store_dtype)
        R_win = torch.zeros((W,) + lead + (N,), device=dev, dtype=store_dtype)
        n_win = torch.zeros((W,) + lead + (N,), device=dev, dtype=torch.long)
        I_win = torch.zeros((W,) + lead + (N,), device=dev, dtype=store_dtype)
        fd_win = (torch.zeros((W,) + lead + (M, Q), device=dev, dtype=store_dtype) if fdist
                  else None)

        out = {k: torch.zeros(lead + (T,), device=dev, dtype=acc)
               for k in ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")}
        out.update({k: torch.zeros(lead + (T, M), device=dev, dtype=acc)
                    for k in ("m", "sigma", "H_fwd", "H_rev")})
        if fdist: out["fdist"] = torch.zeros(lead + (T, M, Q), device=dev, dtype=store_dtype)
        if s:     out["s"]   = torch.zeros(lead + (T, N), device=dev, dtype=torch.int8)
        if pot:   out["pot"] = torch.zeros(lead + (T, N), device=dev, dtype=store_dtype)
        if fields:
            out["hf"] = torch.zeros(lead + (T, N), device=dev, dtype=store_dtype)
            out["hr"] = torch.zeros(lead + (T, N), device=dev, dtype=store_dtype)

        steps_done, end_state = 0, None

        def advance():
            """ One update, remembering the state at reported time T: the windows run past
            the requested horizon (a full W on the first, then chunk at a time), and that
            overshoot is lookahead, not simulation the caller asked for. """
            nonlocal steps_done, end_state
            self.update()
            steps_done += 1
            if steps_done == T: end_state = self._snapshot()

        def record(j):
            s_win[j] = self.s
            S_win[j].copy_(self.S)
            R_win[j].copy_(self.R)
            n_win[j].copy_(self.n)
            I_win[j].copy_(self.total_input())
            if fd_win is not None: fd_win[j].copy_(self.fdist(Q))

        h_rev = torch.zeros(lead + (N,), device=dev, dtype=store_dtype)
        x_rev = torch.zeros(lead + (N,), device=dev, dtype=store_dtype)
        a_rev = torch.zeros(lead + (N,), device=dev, dtype=torch.long)   # reverse age, for the window

        emitted, first = 0, True
        pbar = tqdm.tqdm(total=T, desc="EPR", disable=not pb)
        while emitted < T:

            if first:
                for j in range(W):
                    record(j); advance()
                first = False
            else:
                for buf in (s_win, S_win, R_win, n_win, I_win):
                    buf[:overlap].copy_(buf[chunk:])
                if fd_win is not None: fd_win[:overlap].copy_(fd_win[chunk:])
                for j in range(overlap, W):
                    record(j); advance()

            n_emit = min(chunk, T - emitted)

            if check_overlap:
                never = s_win[chunk:].sum(dim=0) == 0
                silent = never.sum().item()
                if silent:
                    print(f"[warn] {silent}/{never.numel()} neurons never fired in the "
                          f"{overlap}-step tail; increase overlap.")

            h_rev.zero_(); x_rev.zero_(); a_rev.zero_()
            for t in range(W - 2, 0, -1):
                fired = s_win[t].bool()
                h_rev.mul_(1.0 - self.a_int_net).addcmul_(I_win[t + 1], self.a_int_net)
                h_rev.masked_fill_(fired, 0.0)
                a_rev.add_(1)
                a_rev.masked_fill_(fired, 0)
                x_rev = torch.where(fired, self.R_reset_net,
                                    torch.where(a_rev > self.n_abs,
                                                (1.0 - self.a_ref_net) * x_rev, x_rev))

                # h_rev/x_rev now hold the reverse field at window index t, which is
                # the one that emits s[t-1] -- so it pairs with the bond (t-1, t)
                k = t - 1
                if k < n_emit:
                    i  = emitted + k
                    hf = self.beta_net * (S_win[k] + R_win[k] - self.theta_net)
                    hr = self.beta_net * (h_rev + x_rev - self.theta_net)
                    sn = s_win[k + 1].to(store_dtype)   # s[t+1], forward emission
                    sc = s_win[k].to(store_dtype)       # s[t],   reverse emission

                    lp_f = -self._neg_log_prob(sn, hf, n_win[k])
                    lp_r = -self._neg_log_prob(sc, hr, a_rev)

                    out["sigma_tot"][..., i] = (lp_f - lp_r).mean(dim=-1, dtype=acc)
                    out["H_fwd_tot"][..., i] = (-lp_f).mean(dim=-1, dtype=acc)
                    out["H_rev_tot"][..., i] = (-lp_r).mean(dim=-1, dtype=acc)
                    out["m_tot"][..., i]     = s_win[k].mean(dim=-1, dtype=acc)

                    # one stacked reduction for the three per-population means
                    pops = self.pop_mean(torch.stack([sc, -lp_f, -lp_r]).to(acc))  # (3, *b, M)
                    out["m"][..., i, :]     = pops[0]
                    out["H_fwd"][..., i, :] = pops[1]
                    out["H_rev"][..., i, :] = pops[2]
                    out["sigma"][..., i, :] = pops[2] - pops[1]

                    if s:      out["s"][..., i, :]   = s_win[k]
                    if pot:    out["pot"][..., i, :] = S_win[k] + R_win[k]
                    if fields: out["hf"][..., i, :]  = hf; out["hr"][..., i, :] = hr
                    if fd_win is not None: out["fdist"][..., i, :, :] = fd_win[k]

            emitted += n_emit
            pbar.update(n_emit)
        pbar.close()

        out["r"], out["r_tot"] = out["m"] / self.dt, out["m_tot"] / self.dt

        # rewind off the lookahead, so the live state matches reported index T
        assert end_state is not None
        self._restore(end_state)

        return {k: v.cpu() for k, v in out.items()}


# Convenience constructors for common spin models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SpinIsingModel(N: int, J: float, E: float, beta: float, theta: float,
               tau_int: float, tau_ref: float, K_ref: float,
               *, dt: float, hazard: str, nu0: float = 1.0,
               t_abs: float = 1.0,
               device: str = "cpu", ic: str = "random",
               B: int | None = None) -> SpinModel:
    
    """ Construct an Ising SpinModel. J is in physical units: pass J, never J/dt -- the
    same convention as the mean field, stated in equations.compute_total_input. B gives a
    batch of B independent replicas (seeds) of the same network; None gives one. """

    M = 1
    Nm = torch.tensor([N], device=device)
    w = torch.full((M,M), J, device=device, dtype=torch.float32)
    E_vec = torch.full((M,), E, device=device, dtype=torch.float32)
    beta_vec = torch.full((M,), beta, device=device, dtype=torch.float32)
    theta_vec = torch.full((M,), theta, device=device, dtype=torch.float32)
    tau_int_vec = torch.full((M,), tau_int, device=device, dtype=torch.float32)
    tau_ref_vec = torch.full((M,), tau_ref, device=device, dtype=torch.float32)
    K_ref_vec = torch.full((M,), K_ref, device=device, dtype=torch.float32)

    if ic == "random":
        return SpinModel.random_start(Nm=Nm, w=w, E=E_vec, beta=beta_vec, theta=theta_vec,
                                      tau_int=tau_int_vec, tau_ref=tau_ref_vec, K_ref=K_ref_vec,
                                      dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, B=B)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, E=E_vec, beta=beta_vec, theta=theta_vec,
                                  tau_int=tau_int_vec, tau_ref=tau_ref_vec, K_ref=K_ref_vec,
                                  dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, B=B)
    else:
        raise ValueError(f"Unknown initial condition '{ic}'")


def SpinWilsonCowan(
        N: int, 
        E_ratio: float,
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
 
        K_ref_E: float, K_ref_I: float,
        *, dt: float, hazard: str, nu0: float = 1.0,
        t_abs: float = 1.0,
        device: str = "cpu", ic: str = "random",
        B: int | None = None
        ) -> SpinModel:
    
    """ Construct a Wilson-Cowan SpinModel. Weights are in physical units: pass w_EE, never
    w_EE/dt -- the same convention as the mean field, in equations.compute_total_input. B gives
    a batch of B independent replicas (seeds) of the same network; None gives one. """

    # split on E_ratio directly, as RDMWilsonCowan does: going through (1 - E_ratio)
    # rounds the wrong way for ratios with no exact binary form (1 - 0.8 = 0.19999...,
    # so N=4000 lands on 3201/799 instead of 3200/800)
    N_E = int(E_ratio * N)
    N_I = N - N_E
    Nm = torch.tensor([N_E, N_I], device=device)
    # w_XY is source-first (X -> Y); the stored matrix is target-first (row = target)
    w = torch.tensor([[w_EE, w_IE], [w_EI, w_II]], device=device, dtype=torch.float32)
    E_vec = torch.tensor([E_exc, E_inh], device=device, dtype=torch.float32)
    beta_vec = torch.tensor([beta_E, beta_I], device=device, dtype=torch.float32)
    theta_vec = torch.tensor([theta_E, theta_I], device=device, dtype=torch.float32)
    tau_int_vec = torch.tensor([tau_int_E, tau_int_I], device=device, dtype=torch.float32)
    tau_ref_vec = torch.tensor([tau_ref_E, tau_ref_I], device=device, dtype=torch.float32)
    K_ref_vec = torch.tensor([K_ref_E, K_ref_I], device=device, dtype=torch.float32)

    if ic == "random":
        return SpinModel.random_start(Nm=Nm, w=w, E=E_vec, beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, B=B)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, E=E_vec,
                                      beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      dt=dt, nu0=nu0, hazard=hazard, t_abs=t_abs, B=B)
    else:
        raise ValueError(f"Unknown initial condition '{ic}'")