import torch
import tqdm

import rdme.kernels as krn
from rdme.hazards import HAZARDS, resolve_timesteps

# Auxiliary functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def age_dist(n: torch.Tensor, Q: int) -> torch.Tensor:
    """ Calculate the firing distribution over neuron ages. """
    counts = torch.bincount(n, minlength=Q)
    if len(counts) > Q:
        counts[Q-1] += counts[Q:].sum() # add all ages >= Q into the last bin
    return counts[:Q] / n.shape[0]

# Class for single systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SpinModel:
    """ Base class for non-Markovian spin models. """

    N: int # number of neurons
    M: int # number of populations
    Nm: torch.Tensor   # (M,) number of neurons in each population, sum(Nm) = N
    dt: float # time step size, in ms
    dt0: float # the model's time quantum; lam0 = 1/dt0
    lam0: float # rate calibration
    hazard: str # key into rdme.hazards.HAZARDS

    s: torch.Tensor # (N,) current state
    n: torch.Tensor # (N,) neuron ages dtype int8
    S: torch.Tensor # (N,) local field state variable, integrated from [D_t-1,...,D_t-Q]
    R: torch.Tensor # (N,) refractory state variable

    w: torch.Tensor # (M,M), synaptic weigths between populations
    E: torch.Tensor # (M,), external input current for each population

    beta: torch.Tensor # (M,), inverse temperature for each population
    theta: torch.Tensor # (M,), firing threshold for each population
    tau_int: torch.Tensor # (M,), integration time constant for each population
    tau_ref: torch.Tensor # (M,), refractory time constant for each population
    K_ref: torch.Tensor # (M,), refractory strength for each population

    device: torch.device    # device to store tensors on

    # helper / intermediate tensor built in constructor¡
    n_obs: int                      # number of observables to track
    pop_map: list[tuple[int, int]]  # list of (start, end) indices for each population
    pop_expand: torch.Tensor        # (M, N) matrix to expand population-level vector to network-level vector

    # Construction and initialization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, s: torch.Tensor, n: torch.Tensor, S: torch.Tensor, R: torch.Tensor,
             w: torch.Tensor, E: torch.Tensor, beta: torch.Tensor, theta: torch.Tensor,
             tau_int: torch.Tensor, K_ref: torch.Tensor, tau_ref: torch.Tensor, 
             Nm: torch.Tensor, n_obs: int = 3, *,
             dt0: float, dt: float | None = None, hazard: str = "escape",
             t_abs: float | None = None):

        """ dt0 is the model and dt the solver, exactly as in RDMNetwork; the convention and
        its checks live in hazards.resolve_timesteps. dt0 is required and keyword-only, dt
        defaults to it, and at that default lam0*dt = 1 puts both hazards on the sigmoid
        this class used to hardcode.

        w is in PHYSICAL units and is NOT pre-divided by dt, matching RDMNetwork: the input
        is built from the rate m/dt, so the scaling lives in total_input rather than at the
        call site. (It used to be the caller's job, which meant every script wrote J/dt and a
        comparison against the mean field had to carry two weight sets for one network.) """

        N = s.shape[0]
        M = Nm.shape[0]
        
        # sanity checks on tensor shapes
        assert s.shape == (N,) and n.shape == (N,) and S.shape == (N,) and R.shape == (N,), \
            "State tensors s, n, S, R must all have shape (N,)."
        assert w.shape == (M, M), f"Synaptic weights w must have shape ({M}, {M}), got {w.shape}."
        assert E.shape == (M,), f"Input vector E must have shape ({M},), got {E.shape}."
        assert beta.shape == (M,), f"beta must have shape ({M},).)"
        assert theta.shape == (M,), f"theta must have shape ({M},)."
        assert tau_int.shape == (M,), f"tau_int must have shape ({M},)."
        assert tau_ref.shape == (M,), f"tau_ref must have shape ({M},)."
        assert K_ref.shape == (M,), f"K_ref must have shape ({M},)."
        assert Nm.sum().item() == N, f"Sum of Nm must equal N={N}, got {Nm.sum().item()}"
        
        self.dt0, dt, self.lam0 = resolve_timesteps(dt0, dt, hazard)

        self.N, self.M, self.dt = N, M, dt
        self.hazard = hazard
        self.s, self.n, self.S, self.R = s, n, S, R
        self.w, self.E = w, E
        self.theta, self.beta = theta, beta
        self.Nm = Nm
        
        a_int = krn.tau2alpha(tau_int, dt)
        self.tau_int, self.a_int = tau_int, a_int
        a_ref = krn.tau2alpha(tau_ref, dt)
        self.K_ref, self.tau_ref, self.a_ref = K_ref, tau_ref, a_ref
        
        self.device = w.device
        self.n_obs = n_obs
        
        # Derive pop_map from Nm
        self.pop_map = []
        start = 0
        for m in range(M):
            end = start + int(Nm[m].item())
            self.pop_map.append((start, end))
            start = end
        
        # Build expansion matrix
        self.pop_expand = torch.zeros((M, N), device=self.device, dtype=torch.float32)
        for m, (start, end) in enumerate(self.pop_map):
            self.pop_expand[m, start:end] = 1.0
        
        self.pop_sizes = self.pop_expand.sum(dim=1, keepdim=True)  # (M, 1)

        # Expand population-level parameters to network level
        self.a_int_net = self.pop_expand.t().float() @ self.a_int  # (N, M) @ (M,) -> (N,)
        self.a_ref_net = self.pop_expand.t().float() @ self.a_ref  # (N, M) @ (M,) -> (N,)
        self.K_ref_net = self.pop_expand.t().float() @ self.K_ref  # (N, M) @ (M,) -> (N,)
        self.theta_net = self.pop_expand.t().float() @ self.theta  # (N, M) @ (M,) -> (N,)
        self.beta_net = self.pop_expand.t().float() @ self.beta    # (N, M) @ (M,) -> (N,)

        # Absolute refractory window. The mean field carries this inside its R(u) kernel; here
        # R is a per-NEURON state with no age axis, so the window has to be applied through the
        # age n instead. The two must agree, which is what the spin-vs-mean-field tests pin.
        # t_abs defaults to dt0 for the refinement-consistency reason in kernels.refractory_kernel.
        self.t_abs = self.dt0 if t_abs is None else t_abs
        self.n_abs = krn.n_abs_bins(self.t_abs, self.dt)
        R_abs = (krn.abs_refractory_floor(self.w, self.E, self.theta, self.beta,
                                          r_max=1.0 / self.t_abs)
                 if self.t_abs > 0 else torch.zeros_like(self.theta))
        self.R_abs_net = self.pop_expand.t().float() @ R_abs.float()   # (N,)

    @classmethod
    def random_start(cls, Nm: torch.Tensor, w: torch.Tensor, E: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    n_obs: int = 1, *, dt0: float, dt: float | None = None,
                    hazard: str = "escape", t_abs: float | None = None) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        #TODO: start this more reasonably, e.g. by sampling from the fixed point distribution for the given parameters
        s = torch.randint(0, 2, (N,), device=device).int()
        n = torch.randint(1, 50, (N,), device=device).long()
        n = torch.where(s==1, torch.zeros_like(n), n)
        S = torch.zeros((N,), device=device, dtype=torch.float32)
        R = torch.zeros((N,), device=device, dtype=torch.float32)
        return cls(s, n, S, R, w, E, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, n_obs=n_obs, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)

    @classmethod
    def silent_start(cls, Nm: torch.Tensor, w: torch.Tensor, E: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    n_obs: int = 1, *, dt0: float, dt: float | None = None,
                    hazard: str = "escape", t_abs: float | None = None) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        s = torch.zeros((N,), device=device)
        n = torch.full((N,), 200, device=device).long()
        S = torch.zeros((N,), device=device, dtype=torch.float32)
        R = torch.zeros((N,), device=device, dtype=torch.float32)
        return cls(s, n, S, R, w, E, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, n_obs=n_obs, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)
    
    
    # Observables and derived quantities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def activity(self) -> torch.Tensor:
        """ Computes the activity at current time time, i.e. a_t = (1/N) sum_i s_i(t). """
        return torch.mean(self.s.float(), dtype=torch.float32)

    @torch.inference_mode()
    def population_activity(self) -> torch.Tensor:
        """ Computes the activity for each population at current time, i.e. a_t^m = (1/N_m) sum_{i in pop m} s_i(t). """
        pop_counts = self.pop_expand @ self.s.float()
        return pop_counts / self.Nm

    # @torch.inference_mode()
    # def kuramoto_order(self) -> float:
    #     """ Calculate the Kuramoto order parameter for the current state. """
    #     val = torch.clamp(self.field(), min=-self.K_ref, max=self.theta)
    #     phase = (val + self.K_ref) / (self.theta + self.K_ref) * 2 * torch.pi
    #     order_parameter = torch.mean(torch.exp(1j * phase))
    #     return torch.abs(order_parameter).item()

    # @torch.inference_mode()
    # def age_entropy(self) -> float:
    #     """ Calculate the entropy of the age distribution. """
    #     p = self.fdist(self.n.max().item()+1) + 1e-10
    #     ent = -torch.sum(torch.where(p > 0.0, p * torch.log(p), torch.zeros_like(p))).item()
    #     return ent

    @torch.inference_mode()
    def pop_to_network(self, a_pop: torch.Tensor) -> torch.Tensor:
        """Convert population-level vector to network-level vector via matrix multiply."""
        return self.pop_expand.t() @ a_pop  # (N, M) @ (M,) -> (N,)

    @torch.inference_mode()
    def rates(self) -> torch.Tensor:
        """ Population firing rates r = m/dt, in 1/ms, shape (M,). The dt-invariant
        observable: compare these across resolutions, never the per-step activities. """
        return self.population_activity() / self.dt

    @torch.inference_mode()
    def total_input(self) -> torch.Tensor:
        """ I at the current time, I^a_t = sum_b w^{ab} r^b_t + E^a, built from the RATE
        r = m/dt -- which is why w carries no 1/dt here, exactly as in RDMNetwork.

        Row = target, column = source, matching w^{ab} ("from b onto a") in the theory
        and compute_total_input in mean_field. The source-first w_XY names the model
        constructors take are transposed into this layout there, not here. """
        pop_I = self.w @ self.rates() + self.E
        return self.pop_to_network(pop_I)

    def R_eff(self, R: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """ The refractory term the hazard actually sees: the exponential state R, floored at
        -R_abs for neurons younger than n_abs bins.

        Takes R and n as ARGUMENTS rather than reading self, because the entropy production
        needs exactly this transformation applied to the reverse trajectory's own (R, n) --
        which are reconstructed by backward recursion and are not the live state. Routing both
        directions through one function is what keeps them from drifting apart. """
        if self.n_abs == 0:
            return R
        return torch.where(n < self.n_abs, -self.R_abs_net.to(R.dtype), R)

    @torch.inference_mode()
    def field(self) -> torch.Tensor:
        return self.S + self.R_eff(self.R, self.n)

    @torch.inference_mode()
    def firing_prob(self) -> torch.Tensor:
        """ Per-step firing probability P(s_i(t+1)=1) = Phi(beta*(H_i(t)+X_i(t)-theta)), with
        Phi the chosen hazard. At the default dt = dt0 this is the sigmoid for both. """
        return self._phi(self.field() - self.theta_net)

    @torch.inference_mode()
    def hazard_rate(self) -> torch.Tensor:
        """ The firing rate lambda(h) the hazard represents, in 1/ms, shape (N,).

        The quantity the hazards actually differ by as dt -> 0; firing_prob is its rendering on
        this step. For hazard="synchronous" it is Phi/dt, which has no limit. """
        return HAZARDS[self.hazard].rate(self.field() - self.theta_net, self.beta_net,
                                         self.lam0, self.dt)

    def _phi(self, h: torch.Tensor) -> torch.Tensor:
        """ The hazard applied to a raw field h = S + R - theta (beta not yet folded in). """
        return HAZARDS[self.hazard].phi(h, self.beta_net, self.lam0, self.dt)

    def _neg_log_prob(self, s: torch.Tensor, hb: torch.Tensor) -> torch.Tensor:
        """ -ln p(s | h) for the current hazard, given a field hb that ALREADY carries beta.

        beta=1 is passed on because the EPR passes expand the gain into the stored fields once
        rather than per bond; the hazard only ever sees beta*h, so this is the same number. """
        return HAZARDS[self.hazard].neg_log_prob(s, hb, 1.0, self.lam0, self.dt)

    @torch.inference_mode()
    def fdist(self, Q: int) -> torch.Tensor:
        """ Calculate the firing distribution over neuron ages for the whole network. """
        return age_dist(self.n, Q)

    @torch.inference_mode()
    def fdists(self, Q: int) -> torch.Tensor:
        """ Calculate the firing distribution over neuron ages for all populations. """
        # Clamp ages to [0, Q-1], treating ages >= Q as Q-1
        n_clamped = torch.clamp(self.n, max=Q-1)
        # One-hot encode ages: (N, Q)
        n_one_hot = torch.nn.functional.one_hot(n_clamped.long(), num_classes=Q).float()
        # Aggregate by population: (M, N) @ (N, Q) -> (M, Q)
        P = self.pop_expand @ n_one_hot
        # Normalize by population size: (M, Q) / (M, 1) -> (M, Q)
        pop_sizes = self.pop_expand.sum(dim=1, keepdim=True)  # (M, 1)
        return P / pop_sizes

    # Dynamics and trajectories
    # ~~~~~~~~~~~~~~~~~~~~~~~~~

    @torch.inference_mode()
    def update(self) -> None:
        """ Update the state of the system based on firing probabilities. """
        probs = self.firing_prob()
        I = self.total_input()

        # sample new spikes based on probabilities
        fired = torch.rand(self.N, device=self.device) < probs
        if self.n_abs:
            # The absolute window must be enforced on the DRAW, not only through the field.
            # R_abs is finite on purpose (-inf gives 0*-inf = nan in the log-probability
            # branches), so a blocked neuron has Phi ~ 1e-34 rather than 0 -- and
            # torch.rand returns exactly 0.0 with probability ~2^-24 in float32, for which
            # `0.0 < Phi` is True at ANY positive Phi. That leaks one spike per ~6e-8
            # neuron-steps through a window that is supposed to be absolute, and no increase
            # in R_abs removes it. Measured before this mask: 1 escape in 6e7 draws, at an
            # age of 1 bin against n_abs = 4. The mean field has no counterpart, being
            # deterministic -- 1e-34 of age mass is simply negligible there.
            fired &= self.n >= self.n_abs
        
        # update state tensor by rolling it down and inserting new spikes at the top
        self.s = torch.roll(self.s, shifts=1, dims=0) 
        self.s = fired.int() # set new state based on fired neurons
        
        # update neuron ages: if fired, age is 0, else increment age by 1
        self.n = torch.where(fired, torch.zeros_like(self.n), self.n + 1).long()
        
        # update field for next time step: integrate I and reset
        self.S = self.a_int_net * I + (1 - self.a_int_net) * self.S # integrate I
        self.S[self.s==1] = 0 # reset local field for neurons that just fired

        # update refractory state for next time step: decay and set to K_ref
        self.R = (1-self.a_ref_net)*self.R # refractory state decays
        self.R[fired] = -self.K_ref_net[fired] # set refractory state to K_ref if fired

    @torch.inference_mode()
    def pop_mean_matrix(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """ (N, M) matrix whose columns are normalized population indicators, so
        x @ pop_mean_matrix() averages a per-neuron quantity over each population.
        The N-weighted sum of those columns is the plain all-neuron mean, which is how
        the per-population and _tot entries of an entropy_trajectory agree. """
        return (self.pop_expand / self.pop_sizes).t().to(dtype)

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
    def trajectory(self, T: int, kur: bool = False, ent: bool = False,
                   s: bool = False, pot: bool = False, fdist: bool = False, 
                   Q: int = 100) -> dict[str, torch.Tensor]:
        """Run for T steps, returning only the requested quantities.

        Always returns "obs" (observables, shape (T, n_obs)). Optional keys:
          kur  — Kuramoto order parameter, shape (T,)
          ent  — age-distribution entropy, shape (T,)
          s    — full spin state, shape (T, N)
          pot  — membrane potential h = S+R, shape (T, N)
          fdist — firing distribution over ages, shape (T, M, Q)
        All tensors are moved to CPU.
        """
        out = {"m_tot": torch.zeros(T, self.n_obs, device=self.device),
               "m": torch.zeros(T, self.M, device=self.device)}
        if kur: out["kur"] = torch.zeros(T, device=self.device)
        if ent: out["ent"] = torch.zeros(T, device=self.device)
        if s:   out["s"]   = torch.zeros(T, self.N, device=self.device, dtype=torch.int8)
        if pot: out["pot"] = torch.zeros(T, self.N, device=self.device)
        if fdist: out["fdist"] = torch.zeros(T, self.M, Q, device=self.device)
        for t in tqdm.tqdm(range(T)):
            out["m_tot"][t] = self.activity()
            out["m"][t] = self.population_activity()
            # if kur: out["kur"][t] = self.kuramoto_order()
            # if ent: out["ent"][t] = self.age_entropy()
            if s:   out["s"][t]   = self.s
            if pot: out["pot"][t] = self.field()
            if fdist: out["fdist"][t] = self.fdists(Q)
            self.update()
        return {k: v.cpu() for k, v in out.items()}
    
    @torch.inference_mode()
    def entropy_trajectory(self, T: int, buffer: int = 100,
                               kur: bool = False, ent: bool = False,
                               s: bool = False, pot: bool = False,
                               fdist: bool = False, fields: bool = False,
                                Q: int = 100) -> dict[str, torch.Tensor]:
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


        Returns the same dict contract as RDMNetwork.entropy_trajectory: m (T, M)
        per-population activity and m_tot (T,) its network aggregate;
        sigma/H_fwd/H_rev (T, M) the per-population entropy-production decomposition
        (sigma = H_rev - H_fwd) and sigma_tot/H_fwd_tot/H_rev_tot (T,) the same
        quantities over the whole network.

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

        dev = self.device
        L = T + buffer
        lo, hi = 0, T

        s_trj = torch.zeros((L, self.N), device=dev, dtype=torch.int8)
        S_fwd = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        R_fwd = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        n_fwd = torch.zeros((L, self.N), device=dev, dtype=torch.long)
        I = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        if kur: kur_buf = torch.zeros(L, device=dev)
        if ent: ent_buf = torch.zeros(L, device=dev)
        if fdist: P_fwd = torch.zeros((L, self.M, Q), device=dev, dtype=torch.float32)

        for t in tqdm.tqdm(range(L), desc="Forward pass"):
            s_trj[t] = self.s
            S_fwd[t] = self.S
            R_fwd[t] = self.R
            n_fwd[t] = self.n
            I[t] = self.total_input()
            if fdist: P_fwd[t] = self.fdist(Q)
            self.update()
            # the tail past the reported window is lookahead for the reverse recursion,
            # not simulation the caller asked for -- remember where the window ends
            if t + 1 == T: end_state = self._snapshot()

        S_rev = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        R_rev = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        # the reverse trajectory's AGE, by the same backward recursion as R_rev: in reversed
        # time a neuron's last spike is its next spike forward, so the age resets on the same
        # mask and increments the other way. Needed only for the absolute refractory window,
        # which is a function of age rather than of R.
        n_rev = torch.zeros((L, self.N), device=dev, dtype=torch.long)

        for t in tqdm.tqdm(range(L - 2, -1, -1), desc="Reverse pass"):
            fired_t = s_trj[t] == 1
            S_rev[t] = (1 - self.a_int_net) * S_rev[t + 1] + self.a_int_net * I[t + 1]
            S_rev[t, fired_t] = 0
            R_rev[t] = (1 - self.a_ref_net) * R_rev[t + 1]
            R_rev[t, fired_t] = -self.K_ref_net[fired_t]
            n_rev[t] = n_rev[t + 1] + 1
            n_rev[t, fired_t] = 0

        hf = self.beta_net * (S_fwd + self.R_eff(R_fwd, n_fwd) - self.theta_net)
        hr = self.beta_net * (S_rev + self.R_eff(R_rev, n_rev) - self.theta_net)

        # both halves of the bond (t, t+1): forward emits s[t+1] from hf[t], reverse
        # emits s[t] from hr[t+1]
        hf_a = hf[lo:hi]                          # field that generated s[t+1]
        hr_a = hr[lo + 1:hi + 1]                  # reverse field that generated s[t]
        s_next = s_trj[lo + 1:hi + 1].float()     # the realized spike s[t+1]
        s_cur  = s_trj[lo:hi].float()             # the realized spike s[t]

        # -ln p(s | h) under the chosen hazard. At hazard="synchronous" this is the
        # -s*h + softplus(h) of the logistic kernel, which is all this used to compute.
        ent_f = self._neg_log_prob(s_next, hf_a)
        ent_r = self._neg_log_prob(s_cur,  hr_a)

        # same dict contract as RDMNetwork.entropy_trajectory: per-population (T, M)
        # arrays, plus the N-weighted network aggregates under the _tot suffix
        pm = self.pop_mean_matrix(ent_f.dtype)        # (N, M)
        H_fwd_pop, H_rev_pop = ent_f @ pm, ent_r @ pm

        out = {
            "m":         s_cur @ pm,
            "m_tot":     s_cur.mean(dim=1),
            "sigma":     H_rev_pop - H_fwd_pop,
            "H_fwd":     H_fwd_pop,
            "H_rev":     H_rev_pop,
            "sigma_tot": (ent_r - ent_f).mean(dim=1),
            "H_fwd_tot": ent_f.mean(dim=1),
            "H_rev_tot": ent_r.mean(dim=1),
        }
        if kur: out["kur"] = kur_buf[lo:hi]
        if ent: out["ent"] = ent_buf[lo:hi]
        if s:   out["s"]   = s_trj[lo:hi]
        if pot: out["pot"] = (S_fwd + R_fwd)[lo:hi]
        if fdist: out["fdist"] = P_fwd[lo:hi]
        if fields:
            out["hf"] = hf_a
            out["hr"] = hr_a

        # rewind off the tail, so the live state matches reported index T -- the same
        # contract RDMNetwork.entropy_trajectory honours, and what makes a post-hoc
        # fdist() comparable against the mean field's p(n)
        self._restore(end_state)

        return {k: v.cpu() for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory_chunked(self, T: int, chunk: int = 2048, overlap: int = 512,
                               store_dtype: torch.dtype = torch.float32,
                               s: bool = False, pot: bool = False,
                               fields: bool = False, fdist: bool = False,
                               Q: int = 100, check_overlap: bool = True) -> dict[str, torch.Tensor]:
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

        Returns the same dict contract as RDMNetwork.entropy_trajectory: m (T, M)
        per-population activity and m_tot (T,) its network aggregate;
        sigma/H_fwd/H_rev (T, M) the per-population entropy-production decomposition
        (sigma = H_rev - H_fwd) and sigma_tot/H_fwd_tot/H_rev_tot (T,) the same
        quantities over the whole network.
        """
        if overlap < 4:
            raise ValueError("overlap must be >= 4")
        if chunk < overlap:
            raise ValueError("chunk must be >= overlap (the slide would self-overlap)")

        dev, N = self.device, self.N
        W = chunk + overlap
        acc = torch.float64

        s_win = torch.zeros((W, N), device=dev, dtype=torch.int8)
        S_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        R_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        n_win = torch.zeros((W, N), device=dev, dtype=torch.long)
        I_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        if fdist: fd_win  = torch.zeros((W, Q), device=dev, dtype=store_dtype)

        pm = self.pop_mean_matrix(acc)                          # (N, M)
        out = {k: torch.zeros(T, device=dev, dtype=acc)
               for k in ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")}
        out.update({k: torch.zeros(T, self.M, device=dev, dtype=acc)
                    for k in ("m", "sigma", "H_fwd", "H_rev")})
        if fdist: out["fdist"] = torch.zeros((T, Q), device=dev, dtype=store_dtype)
        if s:     out["s"]   = torch.zeros((T, N), device=dev, dtype=torch.int8)
        if pot:   out["pot"] = torch.zeros((T, N), device=dev, dtype=store_dtype)
        if fields:
            out["hf"] = torch.zeros((T, N), device=dev, dtype=store_dtype)
            out["hr"] = torch.zeros((T, N), device=dev, dtype=store_dtype)

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
            if fdist: fd_win[j].copy_(self.fdist(Q))

        h_rev = torch.zeros(N, device=dev, dtype=store_dtype)
        x_rev = torch.zeros(N, device=dev, dtype=store_dtype)
        a_rev = torch.zeros(N, device=dev, dtype=torch.long)   # reverse age, for the window

        emitted, first = 0, True
        pbar = tqdm.tqdm(total=T, desc="EPR")
        while emitted < T:

            if first:
                for j in range(W):
                    record(j); advance()
                first = False
            else:
                for buf in (s_win, S_win, R_win, n_win, I_win):
                    buf[:overlap].copy_(buf[chunk:])
                if fdist: fd_win[:overlap].copy_(fd_win[chunk:])
                for j in range(overlap, W):
                    record(j); advance()

            n_emit = min(chunk, T - emitted)

            if check_overlap:
                silent = (s_win[chunk:].sum(dim=0) == 0).sum().item()
                if silent:
                    print(f"[warn] {silent}/{N} neurons never fired in the "
                          f"{overlap}-step tail; increase overlap.")

            h_rev.zero_(); x_rev.zero_(); a_rev.zero_()
            for t in range(W - 2, 0, -1):
                fired = s_win[t].bool()
                h_rev.mul_(1.0 - self.a_int_net).addcmul_(I_win[t + 1], self.a_int_net)
                h_rev.masked_fill_(fired, 0.0)
                x_rev.mul_(1.0 - self.a_ref_net)
                x_rev[fired] = -self.K_ref_net[fired]
                a_rev.add_(1)
                a_rev.masked_fill_(fired, 0)

                # h_rev/x_rev now hold the reverse field at window index t, which is
                # the one that emits s[t-1] -- so it pairs with the bond (t-1, t)
                k = t - 1
                if k < n_emit:
                    i  = emitted + k
                    hf = self.beta_net * (S_win[k] + self.R_eff(R_win[k], n_win[k])
                                          - self.theta_net)
                    hr = self.beta_net * (h_rev + self.R_eff(x_rev, a_rev) - self.theta_net)
                    sn = s_win[k + 1].to(store_dtype)   # s[t+1], forward emission
                    sc = s_win[k].to(store_dtype)       # s[t],   reverse emission

                    lp_f = -self._neg_log_prob(sn, hf)
                    lp_r = -self._neg_log_prob(sc, hr)

                    out["sigma_tot"][i] = (lp_f - lp_r).mean(dtype=acc)
                    out["H_fwd_tot"][i] = (-lp_f).mean(dtype=acc)
                    out["H_rev_tot"][i] = (-lp_r).mean(dtype=acc)
                    out["m_tot"][i]     = s_win[k].mean(dtype=acc)

                    # one stacked matvec for the three per-population reductions
                    pops = torch.stack([sc, -lp_f, -lp_r]).to(acc) @ pm   # (3, M)
                    out["m"][i]     = pops[0]
                    out["H_fwd"][i] = pops[1]
                    out["H_rev"][i] = pops[2]
                    out["sigma"][i] = pops[2] - pops[1]

                    if s:      out["s"][i]     = s_win[k]
                    if pot:    out["pot"][i]   = S_win[k] + R_win[k]
                    if fields: out["hf"][i]    = hf; out["hr"][i] = hr
                    if fdist:  out["fdist"][i] = fd_win[k]

            emitted += n_emit
            pbar.update(n_emit)
        pbar.close()

        # rewind off the lookahead, so the live state matches reported index T
        self._restore(end_state)

        return {k: v.cpu() for k, v in out.items()}


# Convenience constructors for common spin models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SpinIsingModel(N: int, J: float, E: float, beta: float, theta: float,
               tau_int: float, tau_ref: float, K_ref: float,
               *, dt0: float, dt: float | None = None, hazard: str = "escape",
               t_abs: float | None = None,
               device: str = "cpu", ic: str = "random") -> SpinModel:
    
    """ Construct an Ising SpinModel. J is in physical units: pass J, never J/dt -- the
    same convention as the mean field, stated in rdm.compute_total_input. """

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
                                      n_obs=1, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, E=E_vec, beta=beta_vec, theta=theta_vec,
                                  tau_int=tau_int_vec, tau_ref=tau_ref_vec, K_ref=K_ref_vec,
                                  n_obs=1, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)
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
 
        K_ref1: float, K_ref2: float,
        *, dt0: float, dt: float | None = None, hazard: str = "escape",
        t_abs: float | None = None,
        n_obs: int = 1, device: str = "cpu", ic: str = "random"
        ) -> SpinModel:
    
    """ Construct a Wilson-Cowan SpinModel. Weights are in physical units: pass w_EE, never
    w_EE/dt -- the same convention as the mean field, in rdm.compute_total_input. """

    M = 2
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
    K_ref_vec = torch.tensor([K_ref1, K_ref2], device=device, dtype=torch.float32)

    if ic == "random":
        return SpinModel.random_start(Nm=Nm, w=w, E=E_vec, beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      n_obs=n_obs, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, E=E_vec,
                                      beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      n_obs=n_obs, dt=dt, dt0=dt0, hazard=hazard, t_abs=t_abs)
    else:
        raise ValueError(f"Unknown initial condition '{ic}'")