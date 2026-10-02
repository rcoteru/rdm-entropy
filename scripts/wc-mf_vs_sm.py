import matplotlib.pyplot as plt
import torch
import time

import rdme.fixed_points as fpts
import rdme.lyapunov as lyap
from rdme.single import RDMWilsonCowan
from rdme.spin_model import SpinWilsonCowan


# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.set_default_dtype(torch.float64)

N = 50000

E_ratio = 0.8 # ratio of excitatory neurons

# Previous hand-picked values, kept so this is a one-line revert:
# w_EE = 5.0      # coupling strength; adjust to test different regimes
# w_EI = 3.0      # external field; adjust to test different regimes
# w_IE = -6.0       # external field; adjust to test different regimes
# w_II = -0.5       # external field; adjust to test different regimes
# E = 1
# tau_int = 20; tau_int_E = tau_int; tau_int_I = tau_int
# K_ref = 0; K_ref_E = K_ref; K_ref_I = K_ref

# The sharpest cell on the 28561-cell (E, w_EE, g, w_II) grid swept by wc-search.py and
# wc-lyapunov.py, where g = sqrt(w_EI*|w_IE|) is the E-I loop gain and the weights come from
# w_EI = g*rho, w_IE = -g/rho with rho = sqrt(2). This is E = 1.16, w_EE = 3.67, g = 11.25,
# w_II = -2.50.
#
# Why this one: it sits inside the re-entrant suppression tongue in g -- the oscillatory
# region is bounded by two bifurcation curves that pinch together around g ~ 8-12, and the
# leading Lyapunov exponent peaks right on that pinch. It carried the largest lambda_1 of the
# whole grid, +2.5e-3 /ms at 4.4 sigma.
#
# Do not read that as chaos. Re-run at dt = 0.1 the same cell gives lambda_1 = -6.5e-4 /ms,
# so the positive sign is a discretization artifact of dt = 0.2 and the attractor is an
# invariant circle (quasiperiodic). It is still the most strongly structured dynamics anywhere
# on the grid, which is what makes it worth looking at here.
w_EE = 3.67
w_EI = 15.910     #  g*rho,  g = 11.25, rho = sqrt(2)
w_IE = -7.955     # -g/rho
w_II = -2.50

E = 1.16;
E_exc = E 
E_inh = E
beta = 30; beta_E = beta; beta_I = beta
theta = 1; theta_E = theta; theta_I = theta
# Asymmetric, unlike the old symmetric tau_int = 20: slow inhibition is what the Hopf
# condition wants, and it is what the sweep ran. Note tau_int_I = 40 at dt = 0.2 pushes the
# inhibitory age grid to Qm ~ 922 bins, so both models get appreciably heavier than before.
tau_int_E, tau_int_I = 10.0, 40.0
tau_ref = 3; tau_ref_E = tau_ref; tau_ref_I = tau_ref
# Adaptation on the excitatory population only -- K_ref_I made no measurable difference.
K_ref_E, K_ref_I = 0.5, 0.0

dt = 0.3
steps1 = 50000
steps2 = 20000

# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)

sm = SpinWilsonCowan(N, E_ratio, w_EE/dt, w_EI/dt, w_IE/dt, w_II/dt, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I, 
                    dt=dt, device=device, ic="silent")

# srm takes the weights in physical units -- no /dt, since the input is built from the
# rate -- and hazard="synchronous" is the sigmoid of the original model.
mf = RDMWilsonCowan(E_ratio, w_EE, w_EI, w_IE, w_II, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I,
                    dt=dt, hazard="synchronous", eps=0.01, device=device)

print(mf.Qm)

# mf.P = sm.fdist(Q) # initialize mean-field distribution to match spin model

# Main simulation loop
# ~~~~~~~~~~~~~~~~~~~~
times = []
print(f"Running spin model and mean-field simulations for {steps1+steps2} steps on device: {device}...")
times.append(time.time())
print("Running mean-field simulation...")
mf_traj1 = mf.trajectory(T=steps1)
mf_traj2 = mf.entropy_trajectory(T=steps2)
times.append(time.time())
print("Running spin model simulation...")
sm_traj1 = sm.trajectory(T=steps1)
sm_traj2 = sm.entropy_trajectory_chunked(T=steps2)
times.append(time.time())
# show timings
print(f"Markovian mean-field simulation completed in {times[1] - times[0]:.2f} seconds.")
print(f"Markovian spin-model simulation completed in {times[2] - times[1]:.2f} seconds.")
print("All simulations completed successfully.")

# Plotting
# ~~~~~~~~

if True: # visualize final p(n) distribution

    plt.figure(figsize=(10, 4))
    plt.title("Firing age distribution (mean-field vs spin model)")


    fdists = sm.fdists(max(mf.Qm)).cpu().numpy()
    plt.plot(fdists[0], label='Spin Model [E]', linewidth=2)
    plt.plot(fdists[1], label='Spin Model [I]', linewidth=2)

    plt.plot(mf.p[0].cpu().numpy(), label='Mean Field [E]', linewidth=2)
    plt.plot(mf.p[1].cpu().numpy(), label='Mean Field [I]', linewidth=2)

    plt.xlabel('p(n)'); plt.ylabel('Probability'); plt.legend(); plt.grid()

if True: # equilibration trajectories

    plt.figure(figsize=(10, 4))
    plt.title('Equilibration Trajectories')
    plt.plot(sm_traj1["m"][:, 0], label='Spin Model [E]', linewidth=2)
    plt.plot(sm_traj1["m"][:, 1], label='Spin Model [I]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 0], label='Mean Field [E]', linewidth=2)
    plt.plot(mf_traj1["m"][:, 1], label='Mean Field [I]', linewidth=2)
    plt.xlabel('Time Steps'); plt.ylabel('Mean Activity'); 
    plt.legend(); plt.grid(); plt.tight_layout()


if True: # return map (m_t vs m_{t+1}) of non-transient trajectories

    # a return map is a scalar series against its own lag, so it wants the network
    # aggregate, not the (T, M) per-population overlaps
    sm_m = sm_traj2["m_tot"].cpu().numpy()
    mf_m = mf_traj2["m_tot"].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1: plt.Axes; ax2: plt.Axes

    fig.suptitle('Return Map - Transient Trajectories')

    ax1.set_title('Spin Model')
    ax1.scatter(sm_m[:-1], sm_m[1:], s=1, alpha=1)
    ax1.set_xlabel(r'$m_t$'); ax1.set_ylabel(r'$m_{t+1}$')
    ax1.grid()

    ax2.set_title('Mean Field')
    ax2.scatter(mf_m[:-1], mf_m[1:], s=1, alpha=1)
    ax2.set_xlabel(r'$m_t$'); ax2.set_ylabel(r'$m_{t+1}$')
    ax2.grid()

    fig.tight_layout()

if True: # phase plane trajectories

    # the phase plane is the E overlap against the I overlap, so it wants the
    # per-population (T, M) array
    sm_m = sm_traj2["m"].cpu().numpy()
    mf_m = mf_traj2["m"].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # ax1: plt.Axes; ax2: plt.Axes

    fig.suptitle('Phase Plane - Non-transient Trajectories')

    ax1.set_title('Spin Model')
    ax1.scatter(sm_m[:, 0], sm_m[:, 1], s=1, alpha=1)
    ax1.set_xlabel(r'$m_E$'); ax1.set_ylabel(r'$m_I$')
    ax1.grid()

    ax2.set_title('Mean Field')
    ax2.scatter(mf_m[:, 0], mf_m[:, 1], s=1, alpha=1)
    ax2.set_xlabel(r'$m_E$'); ax2.set_ylabel(r'$m_I$')
    ax2.grid()

    fig.tight_layout()


if True: # fixed-point objective |G(m)| over the (m_E, m_I) square

    # With M=2 the objective is a vector field on [0,1]^2, so what is plotted is its
    # magnitude, and the two zero contours -- the nullclines G_E = 0 and G_I = 0 -- are
    # drawn on top: the fixed points are exactly where they cross, which is also the check
    # that the Newton solver did not miss one. The orbit is overlaid to show what the
    # dynamics does with them.
    n_grid = 201

    roots = mf.fixed_points(method='newton', guesses=fpts.fp_guess_lattice(2, 15, device=device))
    rho   = torch.tensor([fpts.spectral_radius(mf.fp_eigenvalues(r)) for r in roots])
    orbit = mf_traj2["m"]                                                   # (T, 2)

    # everything interesting sits within a few percent of the origin, so the second panel
    # covers only the span the roots and the orbit occupy -- on its own grid, since
    # subsetting the full-range one would leave a handful of cells across the whole window
    hi     = max(roots.max().item(), orbit.max().item()) * 1.4
    windows = [((0.0, 1.0), 'Full range'), ((-0.05 * hi, hi), 'Around the roots')]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Fixed-point objective  |G(m)| = |m - F(m)|')

    for ax, ((lo, up), title) in zip(axes, windows):
        axis   = torch.linspace(lo, up, n_grid, device=device)
        mE, mI = torch.meshgrid(axis, axis, indexing='ij')
        pts    = torch.stack([mE.reshape(-1), mI.reshape(-1)], dim=-1)      # (K, 2)
        # chunked: each residual evaluation materialises a (chunk, Q) field per population
        G  = torch.cat([mf.fp_residual(c) for c in pts.split(4096)]).cpu()  # (K, 2)
        # .T throughout because G is indexed [m_E, m_I] and matplotlib wants [row=y, col=x]
        GE = G[:, 0].reshape(n_grid, n_grid).T
        GI = G[:, 1].reshape(n_grid, n_grid).T
        Gn = G.norm(dim=-1).reshape(n_grid, n_grid).T
        a  = axis.cpu()

        ax.set_title(title)
        pcm = ax.pcolormesh(a, a, torch.log10(Gn + 1e-12), shading='auto', cmap='viridis')
        ax.contour(a, a, GE, levels=[0.0], colors='w', linewidths=1.5)
        ax.contour(a, a, GI, levels=[0.0], colors='w', linewidths=1.5, linestyles='--')
        ax.plot(orbit[:, 0], orbit[:, 1], color='C1', linewidth=0.8, alpha=0.8)
        for r, rh in zip(roots, rho):
            ax.plot(r[0].item(), r[1].item(), marker='o', markersize=9, zorder=3, color='r',
                    markerfacecolor='r' if rh < 1.0 else 'none',
                    label=f"m*=({r[0]:.4f}, {r[1]:.4f})  |λ|={rh:.4f}")
        ax.set_xlim(lo, up); ax.set_ylim(lo, up)
        ax.set_xlabel(r'$m_E$'); ax.set_ylabel(r'$m_I$')
        # one colorbar per panel: the two windows span very different ranges of |G|, and a
        # single shared bar would be labelled with only one of the two norms
        fig.colorbar(pcm, ax=ax, label=r'$\log_{10}|G(m)|$')

    # empty handles so the legend can name the contours and the orbit too; filled marker =
    # stable root, open = unstable, same convention as everywhere else
    for ax in axes:
        ax.plot([], [], color='w', linewidth=1.5, label=r'$G_E = 0$')
        ax.plot([], [], color='w', linewidth=1.5, linestyle='--', label=r'$G_I = 0$')
        ax.plot([], [], color='C1', linewidth=0.8, label='mean-field orbit')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)

    fig.tight_layout()


if True: # entropy trajectories

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
    # ax1: plt.Axes; ax2: plt.Axes; ax3: plt.Axes; ax4: plt.Axes;

    # activity trajectories
    ax1.set_title('Activity')
    ax1.plot(sm_traj2["m_tot"], label='Spin Model', linewidth=2)
    ax1.plot(mf_traj2["m_tot"], label='Mean Field', linewidth=2)
    ax1.set_xlabel('Time Steps')
    ax1.legend(); ax1.grid()

    # sigma trajectories
    ax2.set_title('Entropy Production Rate')
    ax2.plot(sm_traj2["sigma_tot"], label='Spin Model', linewidth=2)
    ax2.plot(mf_traj2["sigma_tot"], label='Mean Field', linewidth=2)
    ax2.legend(); ax2.grid()

    # H_fwd trajectories
    ax3.set_title('Forward Entropy')
    ax3.plot(sm_traj2["H_fwd_tot"], label='Spin Model', linewidth=2)
    ax3.plot(mf_traj2["H_fwd_tot"], label='Mean Field', linewidth=2)
    ax3.legend(); ax3.grid()

    # H_rev trajectories
    ax4.set_title('Backward Entropy')
    ax4.plot(sm_traj2["H_rev_tot"], label='Spin Model', linewidth=2)
    ax4.plot(mf_traj2["H_rev_tot"], label='Mean Field', linewidth=2)
    ax4.legend(); ax4.grid()

    # print trajetcory averages
    skip = 100 # skip initial and final transient
    print(f"Spin Model: <sigma> = {sm_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <H_fwd> = {sm_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {sm_traj2['H_rev'][skip:-skip].mean():.8f}")
    print(f"Mean Field: <sigma> = {mf_traj2['sigma'][skip:-skip].mean():.8f} nats/ms, <H_fwd> = {mf_traj2['H_fwd'][skip:-skip].mean():.8f}, <H_rev> = {mf_traj2['H_rev'][skip:-skip].mean():.8f}")

    plt.tight_layout()


if True: # entropy trajectories, decomposed by population

    # The same four quantities as above, but per population rather than N-weighted totals.
    # The "_tot" series hide the thing that is actually interesting at these parameters: E and
    # I run at very different rates and their entropy production need not even have the same
    # sign of trend, and a population-size-weighted sum of the two is dominated by E (N_E/N is
    # 0.8 here) and so says little about I.
    #
    # Column is the population, colour is the model. The comparison this figure exists for is
    # spin-model against mean-field within one panel, so the model keeps one colour everywhere
    # and the population is read off position. Rows share a y-axis so E and I are on the same
    # scale; that is the point of splitting them.
    POPS = ["Excitatory", "Inhibitory"]
    ROWS = [("m",     "Activity"),
            ("sigma", "Entropy Production Rate"),
            ("H_fwd", "Forward Entropy"),
            ("H_rev", "Backward Entropy")]

    fig, axs = plt.subplots(len(ROWS), 2, sharex=True, sharey='row', figsize=(14, 12))

    for r, (key, title) in enumerate(ROWS):
        for c in range(len(POPS)):
            ax = axs[r, c]
            ax.plot(sm_traj2[key][:, c], label='Spin Model', linewidth=2)
            ax.plot(mf_traj2[key][:, c], label='Mean Field', linewidth=2)
            ax.set_title(f'{title} - {POPS[c]}')
            ax.grid()
            if r == len(ROWS) - 1:
                ax.set_xlabel('Time Steps')
    axs[0, 0].legend()   # identity is the same in every panel, so one legend covers the figure

    # per-population averages, with the same transient trim the totals use above
    skip = 100
    for c, pop in enumerate(POPS):
        print(f"{pop} (column {c}):")
        for nm, tr in (("  Spin Model", sm_traj2), ("  Mean Field", mf_traj2)):
            print(f"{nm}: <m> = {tr['m'][skip:-skip, c].mean():.8f}, "
                  f"<sigma> = {tr['sigma'][skip:-skip, c].mean():.8f} nats/ms, "
                  f"<H_fwd> = {tr['H_fwd'][skip:-skip, c].mean():.8f}, "
                  f"<H_rev> = {tr['H_rev'][skip:-skip, c].mean():.8f}")

    plt.tight_layout()


if True: # Lyapunov spectrum of the mean field

    # The exponents are a property of the deterministic mean-field map, so only mf is used
    # here; the spin model is a finite-N stochastic process and has no Lyapunov spectrum in
    # this sense. Also note this is a MAP, not a flow: an attracting periodic orbit has every
    # exponent strictly negative, and lambda_1 ~ 0 means an invariant circle (quasiperiodic),
    # not a limit cycle. Reading the flow table here would call every locked cycle a fixed
    # point.
    k_lyap   = 2       # lambda_1 decides chaos; lambda_2 separates a circle from a 2-torus
    eq_lyap  = 15000   # settle the trajectory
    wu_lyap  = 2000    # settle the tangent BASIS: a random one is not yet aligned
    T_lyap   = 20000   # accumulation window; sets the ~1e-4 /ms noise floor

    lmf = RDMWilsonCowan(E_ratio, w_EE, w_EI, w_IE, w_II,
                         E_exc, E_inh, beta_E, beta_I, theta_E, theta_I,
                         tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I,
                         dt=dt, hazard="synchronous", eps=0.01, device=device)

    print(f"\nLyapunov: equilibrating {eq_lyap} steps, then {wu_lyap} warmup + {T_lyap} "
          f"accumulation, k={k_lyap} ...")
    t0 = time.time()
    lmf.forward(eq_lyap)
    blocks = lmf.init_tangent(k_lyap, generator=torch.Generator().manual_seed(0))
    lmf.step_tangent(blocks)        # one step annihilates the 2M unphysical directions
    lyap.orthonormalize(blocks)
    spec = lyap.benettin(lmf.step_tangent, blocks, n_steps=T_lyap, dt=dt,
                         cadence=100, n_blocks=20, warmup=wu_lyap)
    print(f"  done in {time.time() - t0:.1f}s")

    for i in range(k_lyap):
        print(f"  lambda_{i+1} = {spec.lam[i]:+.4e} +/- {spec.se[i]:.1e} /ms"
              f"   ({spec.lam[i]/spec.se[i]:+.1f} sigma)"
              f"   drift {spec.drift[i]:+.1e}"
              f"{'   [degenerate with its neighbour]' if spec.degenerate[i] else ''}")
    print(f"  partial sums: {[f'{v:+.4e}' for v in spec.partial.tolist()]}")

    # Classification needs one bit the spectrum cannot supply: whether the state actually
    # moves. In a map a dead cell and a locked cycle both have lambda_1 < 0.
    m_E = mf_traj2["m"][:, 0]
    cv = (m_E.std() / m_E.mean().clamp_min(1e-300))
    moving = torch.tensor(bool(cv > 1e-4))
    tol = max(5e-4, 3 * spec.se[0].item())      # floor, or 3 sigma, whichever binds
    cls = lyap.classify_spectrum(spec.lam, tol=tol, moving=moving,
                                 se=spec.se, n_sigma=3.0)
    print(f"  CV(m_E) = {cv:.3e} -> moving={bool(moving)};  band = +/-{tol:.1e} /ms")
    print(f"  => {lyap.CLASS_NAMES[int(cls)]}")
    if int(cls) == lyap.CHAOS:
        print("  NOTE: confirm at dt <= 0.1 before believing this. On the parameter sweep every"
              "\n        chaotic candidate at dt = 0.2 flipped sign at dt = 0.1.")

plt.show()