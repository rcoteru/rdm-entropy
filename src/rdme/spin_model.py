import torch
import tqdm

import rdme.shared as shrd


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

    s: torch.Tensor # (N,) current state
    n: torch.Tensor # (N,) neuron ages dtype int8
    H: torch.Tensor # (N,) local field state variable, integrated from [D_t-1,...,D_t-Q]
    X: torch.Tensor # (N,) refractory state variable

    w: torch.Tensor # (M,M), synaptic weigths between populations
    I: torch.Tensor # (M,), external input current for each population

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

    def __init__(self, s: torch.Tensor, n: torch.Tensor, H: torch.Tensor, X: torch.Tensor,
             w: torch.Tensor, I: torch.Tensor, beta: torch.Tensor, theta: torch.Tensor,
             tau_int: torch.Tensor, K_ref: torch.Tensor, tau_ref: torch.Tensor, 
             Nm: torch.Tensor, n_obs: int = 3, dt: float = 1.0):
    
        N = s.shape[0]
        M = Nm.shape[0]
        
        # sanity checks on tensor shapes
        assert s.shape == (N,) and n.shape == (N,) and H.shape == (N,) and X.shape == (N,), \
            "State tensors s, n, H, X must all have shape (N,)."
        assert w.shape == (M, M), f"Synaptic weights w must have shape ({M}, {M}), got {w.shape}."
        assert I.shape == (M,), f"Input vector I must have shape ({M},), got {I.shape}."
        assert beta.shape == (M,), f"beta must have shape ({M},).)"
        assert theta.shape == (M,), f"theta must have shape ({M},)."
        assert tau_int.shape == (M,), f"tau_int must have shape ({M},)."
        assert tau_ref.shape == (M,), f"tau_ref must have shape ({M},)."
        assert K_ref.shape == (M,), f"K_ref must have shape ({M},)."
        assert Nm.sum().item() == N, f"Sum of Nm must equal N={N}, got {Nm.sum().item()}"
        
        self.N, self.M, self.dt = N, M, dt
        self.s, self.n, self.H, self.X = s, n, H, X
        self.w, self.I = w, I
        self.theta, self.beta = theta, beta
        self.Nm = Nm
        
        a_int = shrd.tau2alpha(tau_int, dt)
        self.tau_int, self.a_int = tau_int, a_int
        a_ref = shrd.tau2alpha(tau_ref, dt)
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

    @classmethod
    def random_start(cls, Nm: torch.Tensor, w: torch.Tensor, I: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    n_obs: int = 1, dt: float = 1.0) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        #TODO: start this more reasonably, e.g. by sampling from the fixed point distribution for the given parameters
        s = torch.randint(0, 2, (N,), device=device).int()
        n = torch.randint(1, 50, (N,), device=device).long()
        n = torch.where(s==1, torch.zeros_like(n), n)
        H = torch.zeros((N,), device=device, dtype=torch.float32)
        X = torch.zeros((N,), device=device, dtype=torch.float32)
        return cls(s, n, H, X, w, I, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, n_obs=n_obs, dt=dt)

    @classmethod
    def silent_start(cls, Nm: torch.Tensor, w: torch.Tensor, I: torch.Tensor, 
                    beta: torch.Tensor, theta: torch.Tensor, 
                    tau_int: torch.Tensor, tau_ref: torch.Tensor, K_ref: torch.Tensor, 
                    n_obs: int = 1, dt: float = 1.0) -> SpinModel:
        device = w.device
        N = int(Nm.sum().item())
        s = torch.zeros((N,), device=device)
        n = torch.full((N,), 200, device=device).long()
        H = torch.zeros((N,), device=device, dtype=torch.float32)
        X = torch.zeros((N,), device=device, dtype=torch.float32)
        return cls(s, n, H, X, w, I, beta, theta, tau_int, K_ref, tau_ref, 
                Nm, n_obs=n_obs, dt=dt)
    
    
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
    def drive(self) -> torch.Tensor:
        """ Computes the drive at current time, i.e. D_t = J*s_t + I. """
        activities = self.population_activity()
        pop_drive = self.w.T @ activities + self.I
        return self.pop_to_network(pop_drive)

    @torch.inference_mode()
    def field(self) -> torch.Tensor:
        return self.H + self.X

    @torch.inference_mode()
    def firing_prob(self) -> torch.Tensor:
        """ Computes the firing probability at current time, i.e. P(s_i(t+1)=1) = sigmoid(beta*(H_i(t)+X_i(t)-theta)). """
        return torch.sigmoid(self.beta_net * (self.field() - self.theta_net))

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
        drive = self.drive()

        # sample new spikes based on probabilities
        fired = torch.rand(self.N, device=self.device) < probs
        
        # update state tensor by rolling it down and inserting new spikes at the top
        self.s = torch.roll(self.s, shifts=1, dims=0) 
        self.s = fired.int() # set new state based on fired neurons
        
        # update neuron ages: if fired, age is 0, else increment age by 1
        self.n = torch.where(fired, torch.zeros_like(self.n), self.n + 1).long()
        
        # update field for next time step: integrate drive and reset
        self.H = self.a_int_net * drive + (1 - self.a_int_net) * self.H # integrate drive
        self.H[self.s==1] = 0 # reset local field for neurons that just fired

        # update refractory state for next time step: decay and set to K_ref
        self.X = (1-self.a_ref_net)*self.X # refractory state decays
        self.X[fired] = -self.K_ref_net[fired] # set refractory state to K_ref if fired

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
          pot  — membrane potential H+X, shape (T, N)
          fdist — firing distribution over ages, shape (T, M, Q)
        All tensors are moved to CPU.
        """
        out = {"a_tot": torch.zeros(T, self.n_obs, device=self.device),
               "a_pop": torch.zeros(T, self.M, device=self.device)}
        if kur: out["kur"] = torch.zeros(T, device=self.device)
        if ent: out["ent"] = torch.zeros(T, device=self.device)
        if s:   out["s"]   = torch.zeros(T, self.N, device=self.device, dtype=torch.int8)
        if pot: out["pot"] = torch.zeros(T, self.N, device=self.device)
        if fdist: out["fdist"] = torch.zeros(T, self.M, Q, device=self.device)
        for t in tqdm.tqdm(range(T)):
            out["a_tot"][t] = self.activity()
            out["a_pop"][t] = self.population_activity()
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

        Timing: the field at index t generates the spike at index t+1, so
        P(s[t+1]=1) = sigmoid(hf[t]) and the reverse-process counterpart is
        hr[t+2].

        sigma[t] is the *sampled* log-ratio ln p(Gamma)/p(Gamma^dagger) per
        neuron per step. It fluctuates in sign; only its average is the EP
        rate. Do not substitute sigmoid(hf) for the realized spike -- hr
        depends on future spikes and is correlated with s[t+1], so that
        substitution turns the estimator into a pointwise KL that is
        non-negative by construction and cannot detect reversibility.

        Runs T + 2*buffer steps: burn-in | analysis window | tail.
        """
        if buffer < 2:
            raise ValueError("buffer must be >= 2; in practice use several "
                             "times max(tau_int, tau_ref).")

        dev = self.device
        F = torch.nn.functional
        L = T + 2 * buffer
        lo, hi = buffer, buffer + T

        s_trj = torch.zeros((L, self.N), device=dev, dtype=torch.int8)
        H_fwd = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        X_fwd = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        drive = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        if kur: kur_buf = torch.zeros(L, device=dev)
        if ent: ent_buf = torch.zeros(L, device=dev)
        if fdist: P_fwd = torch.zeros((L, self.M, Q), device=dev, dtype=torch.float32)

        for t in tqdm.tqdm(range(L), desc="Forward pass"):
            s_trj[t] = self.s
            H_fwd[t] = self.H
            X_fwd[t] = self.X
            drive[t] = self.drive()
            if fdist: P_fwd[t] = self.fdist(Q)
            self.update()

        H_rev = torch.zeros((L, self.N), device=dev, dtype=torch.float32)
        X_rev = torch.zeros((L, self.N), device=dev, dtype=torch.float32)

        for t in tqdm.tqdm(range(L - 2, -1, -1), desc="Reverse pass"):
            fired_t = s_trj[t] == 1
            H_rev[t] = (1 - self.a_int_net) * H_rev[t + 1] + self.a_int_net * drive[t + 1]
            H_rev[t, fired_t] = 0
            X_rev[t] = (1 - self.a_ref_net) * X_rev[t + 1]
            X_rev[t, fired_t] = -self.K_ref_net[fired_t]

        hf = self.beta_net * (H_fwd + X_fwd - self.theta_net)
        hr = self.beta_net * (H_rev + X_rev - self.theta_net)

        hf_a = hf[lo:hi]                          # field that generated s[t+1]
        hr_a = hr[lo + 2:hi + 2]                  # reverse field, same spike
        s_next = s_trj[lo + 1:hi + 1].float()     # the realized spike s[t+1]

        # log-likelihood of the realized spike under each field
        lp_f = s_next * hf_a - F.softplus(hf_a)
        lp_r = s_next * hr_a - F.softplus(hr_a)

        # conditional entropies (diagnostics only -- their difference is a KL
        # and is NOT the entropy production)
        # p_f = torch.sigmoid(hf_a)
        ent_f = -s_next * hf_a + F.softplus(hf_a)
        ent_r = -s_next * hr_a + F.softplus(hr_a)

        out = {
            "a_tot":  s_trj[lo:hi].mean(dim=1, dtype=torch.float32),
            "sigma": (lp_f - lp_r).mean(dim=1),
            "S_fwd": ent_f.mean(dim=1),
            "S_rev": ent_r.mean(dim=1),
        }
        if kur: out["kur"] = kur_buf[lo:hi]
        if ent: out["ent"] = ent_buf[lo:hi]
        if s:   out["s"]   = s_trj[lo:hi]
        if pot: out["pot"] = (H_fwd + X_fwd)[lo:hi]
        if fdist: out["fdist"] = P_fwd[lo:hi]
        if fields:
            out["hf"] = hf_a
            out["hr"] = hr_a

        return {k: v.cpu() for k, v in out.items()}

    @torch.inference_mode()
    def entropy_trajectory_chunked(self, T: int, chunk: int = 2048, overlap: int = 512,
                               burn_in: int | None = None,
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

        Timing (unchanged): P(s[t+1]=1) = sigmoid(hf[t]); the reverse-process
        counterpart of hf[t] is hr[t+2]. Both S_fwd and S_rev use the realized
        spike, so sigma == S_rev - S_fwd identically.

        Trajectories are stored in `store_dtype` (float32 is ample -- the
        estimator is sampling-noise dominated) while all reductions accumulate
        in float64.
        """
        if overlap < 4:
            raise ValueError("overlap must be >= 4")
        if chunk < overlap:
            raise ValueError("chunk must be >= overlap (the slide would self-overlap)")
        if burn_in is None:
            burn_in = overlap

        dev, N = self.device, self.N
        F = torch.nn.functional
        W = chunk + overlap
        acc = torch.float64

        s_win = torch.zeros((W, N), device=dev, dtype=torch.int8)
        H_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        X_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        D_win = torch.zeros((W, N), device=dev, dtype=store_dtype)
        if fdist: fd_win  = torch.zeros((W, Q), device=dev, dtype=store_dtype)

        out = {k: torch.zeros(T, device=dev, dtype=acc)
               for k in ("a_tot", "sigma", "S_fwd", "S_rev")}
        if fdist: out["fdist"] = torch.zeros((T, Q), device=dev, dtype=store_dtype)
        if s:     out["s"]   = torch.zeros((T, N), device=dev, dtype=torch.int8)
        if pot:   out["pot"] = torch.zeros((T, N), device=dev, dtype=store_dtype)
        if fields:
            out["hf"] = torch.zeros((T, N), device=dev, dtype=store_dtype)
            out["hr"] = torch.zeros((T, N), device=dev, dtype=store_dtype)

        def record(j):
            s_win[j] = self.s
            H_win[j].copy_(self.H)
            X_win[j].copy_(self.X)
            D_win[j].copy_(self.drive())
            if fdist: fd_win[j].copy_(self.fdist(Q))

        for _ in range(burn_in):
            self.update()

        h_rev = torch.zeros(N, device=dev, dtype=store_dtype)
        x_rev = torch.zeros(N, device=dev, dtype=store_dtype)

        emitted, first = 0, True
        pbar = tqdm.tqdm(total=T, desc="EPR")
        while emitted < T:

            if first:
                for j in range(W):
                    record(j); self.update()
                first = False
            else:
                for buf in (s_win, H_win, X_win, D_win):
                    buf[:overlap].copy_(buf[chunk:])
                if fdist: fd_win[:overlap].copy_(fd_win[chunk:])
                for j in range(overlap, W):
                    record(j); self.update()

            n_emit = min(chunk, T - emitted)

            if check_overlap:
                silent = (s_win[chunk:].sum(dim=0) == 0).sum().item()
                if silent:
                    print(f"[warn] {silent}/{N} neurons never fired in the "
                          f"{overlap}-step tail; increase overlap.")

            h_rev.zero_(); x_rev.zero_()
            for t in range(W - 2, 1, -1):
                fired = s_win[t].bool()
                h_rev.mul_(1.0 - self.a_int_net).addcmul_(D_win[t + 1], self.a_int_net)
                h_rev.masked_fill_(fired, 0.0)
                x_rev.mul_(1.0 - self.a_ref_net)
                x_rev[fired] = -self.K_ref_net[fired]

                k = t - 2
                if k < n_emit:
                    i  = emitted + k
                    hf = self.beta_net * (H_win[k] + X_win[k] - self.theta_net)
                    hr = self.beta_net * (h_rev + x_rev - self.theta_net)
                    sn = s_win[k + 1].to(store_dtype)

                    lp_f = sn * hf - F.softplus(hf)
                    lp_r = sn * hr - F.softplus(hr)

                    out["sigma"][i] = (lp_f - lp_r).mean(dtype=acc)
                    out["S_fwd"][i] = (-lp_f).mean(dtype=acc)
                    out["S_rev"][i] = (-lp_r).mean(dtype=acc)
                    out["a_tot"][i]     = s_win[k].mean(dtype=acc)

                    if s:      out["s"][i]     = s_win[k]
                    if pot:    out["pot"][i]   = H_win[k] + X_win[k]
                    if fields: out["hf"][i]    = hf; out["hr"][i] = hr
                    if fdist:  out["fdist"][i] = fd_win[k]

            emitted += n_emit
            pbar.update(n_emit)
        pbar.close()

        return {k: v.cpu() for k, v in out.items()}


# Convenience constructors for common spin models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def SpinIsingModel(N: int, J: float, I: float, beta: float, theta: float,
               tau_int: float, tau_ref: float, K_ref: float,
               dt: float = 1.0, device: str = "cpu",
               ic: str = "random") -> SpinModel:
    
    """ Construct an Ising SpinModel. """

    M = 1
    Nm = torch.tensor([N], device=device)
    w = torch.full((M,M), J, device=device, dtype=torch.float32)
    I_vec = torch.full((M,), I, device=device, dtype=torch.float32)
    beta_vec = torch.full((M,), beta, device=device, dtype=torch.float32)
    theta_vec = torch.full((M,), theta, device=device, dtype=torch.float32)
    tau_int_vec = torch.full((M,), tau_int, device=device, dtype=torch.float32)
    tau_ref_vec = torch.full((M,), tau_ref, device=device, dtype=torch.float32)
    K_ref_vec = torch.full((M,), K_ref, device=device, dtype=torch.float32)

    if ic == "random":
        return SpinModel.random_start(Nm=Nm, w=w, I=I_vec, beta=beta_vec, theta=theta_vec,
                                      tau_int=tau_int_vec, tau_ref=tau_ref_vec, K_ref=K_ref_vec,
                                      n_obs=1, dt=dt)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, I=I_vec, beta=beta_vec, theta=theta_vec,
                                  tau_int=tau_int_vec, tau_ref=tau_ref_vec, K_ref=K_ref_vec,
                                  n_obs=1, dt=dt)
    else:
        raise ValueError(f"Unknown initial condition '{ic}'")


def SpinWilsonCowan(
        N: int, 
        E_ratio: float,
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
 
        K_ref1: float, K_ref2: float,
        dt: float = 1.0, n_obs: int = 1, 
        device: str = "cpu", ic: str = "random"
        ) -> SpinModel:
    
    """ Construct a Wilson-Cowan SpinModel. """

    M = 2
    N_I = int(N * (1 - E_ratio))
    N_E = N - N_I
    Nm = torch.tensor([N_E, N_I], device=device)
    w = torch.tensor([[w_EE, w_EI], [w_IE, w_II]], device=device, dtype=torch.float32)
    I_vec = torch.tensor([I_E, I_I], device=device, dtype=torch.float32)
    beta_vec = torch.tensor([beta_E, beta_I], device=device, dtype=torch.float32)
    theta_vec = torch.tensor([theta_E, theta_I], device=device, dtype=torch.float32)
    tau_int_vec = torch.tensor([tau_int_E, tau_int_I], device=device, dtype=torch.float32)
    tau_ref_vec = torch.tensor([tau_ref_E, tau_ref_I], device=device, dtype=torch.float32)
    K_ref_vec = torch.tensor([K_ref1, K_ref2], device=device, dtype=torch.float32)

    if ic == "random":
        return SpinModel.random_start(Nm=Nm, w=w, I=I_vec, beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      n_obs=n_obs, dt=dt)
    elif ic == "silent":
        return SpinModel.silent_start(Nm=Nm, w=w, I=I_vec,
                                      beta=beta_vec,
                                      theta=theta_vec,
                                      tau_int=tau_int_vec,
                                      tau_ref=tau_ref_vec,
                                      K_ref=K_ref_vec,
                                      n_obs=n_obs, dt=dt)
    else:
        raise ValueError(f"Unknown initial condition '{ic}'")