"""
wc-mf_vs_sm.py at one interesting parameter point, so it can be eyeballed.

The cell is the strongest aperiodic candidate of the wc-painchaud.py sweep: at dt = 0.1 its
leading Lyapunov exponent is +1.5e-2 /ms, stable to within its error bar over 60 s of
accumulation (first half +1.429e-2, last half +1.560e-2), with lambda_2 ~ -1e-5. The rate
trace is stationary over the same 60 s -- four 2 s epochs spread across it give 34.5, 34.2,
33.7, 34.7 Hz at CV 2.57, 2.49, 2.52, 2.54 -- so whatever this is, it is not a transient.

Two caveats worth holding while looking at the plots:

  The same cell gives lambda_1 = +1.0e-5 at dt = 0.05, while a NEIGHBOURING cell
  (tau_scale 1.8, cross 2.236 instead of 2.1, 1.495) goes the other way: zero at dt = 0.1 and
  +9.7e-3 at dt = 0.05. Both measurements are well resolved, so each is a true property of its
  own discretization, and the natural reading is a fine-grained chaotic set -- periodic windows
  interleaved with chaotic bands -- in which an O(dt) shift moves the cell between them. That
  neighbour is the commented-out parameter line below.

  The mean field drops covariances. An aperiodicity that is an artifact of that closure has no
  reason to appear in the microscopic process, which is why the spin model is run alongside and
  is the panel to look at first.

Everything else -- the panels, the plotting, the entropy comparison -- is wc-mf_vs_sm.py
unchanged.
"""

# ── NOT RUNNABLE (2026-11-07) ─────────────────────────────────────────────────
#
# Ported to the route-B API mechanically and nothing more: `dt0` is gone, the dead hazard keys
# are mapped, and the model builds. It is NOT wired to a preset and has NOT been re-run, because
# the two-population presets it needs (`EI`, `EE-P`) are still `PENDING` in rdme.presets and
# their values are a vault decision that has not been made. Running it produces numbers that
# belong to no named parameter set, so do not.
#
# What the port changed, and why it is a change of MODEL and not a rename:
#   `dt0`         removed. There is no time quantum under route B; `nu0` is the rate unit of
#                 the Poisson reference and `t_abs` the physical dead time.
#   "async"    -> "poisson_exact". The route-B kernel with async's per-bin FORM is "bernoulli"
#                 (nu0*dt*sigma), not this one; the 2026-11-07 brief asks for the solver here
#                 instead, so the rate function changes from saturating to exponential.
#   "sync"/"synchronous"
#              -> "bernoulli", which at nu0*dt = 1 is the sigmoid bit for bit.
#
#   the TIME-UNIT ANCHOR is unresolved. This script fixes its physical time unit by
#                 T_u = alpha_1 * dt0, i.e. by identifying Painchaud's maximal rate with the
#                 old lam0 = 1/dt0 RATE CEILING. Route B has no rate ceiling: the hazard is
#                 exponential and the only cap is 1/t_abs. The arithmetic below is left
#                 EXACTLY as it was and the old dt0 survives only as DT0_ANCHOR, a number
#                 feeding T_u; it is no longer passed to any model. Choosing the new anchor
#                 is a vault decision and is what blocks the EE-P preset.
# ──────────────────────────────────────────────────────────────────────────────

import math

import matplotlib.pyplot as plt
import torch
import time

import rdme.fixed_points as fpts
import rdme.kernels as krn
import rdme.lyapunov as lyap
import rdme.plots as rp
from rdme.rdm_single import RDMWilsonCowan
from rdme.spin_model import SpinWilsonCowan


# Simulation parameters
# ~~~~~~~~~~~~~~~~~~~~~

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_default_dtype(torch.float64)

N = 50000

# The cell. Translated parameters, already evaluated -- see latex/app_painchaud.tex for where
# the numbers come from. Both populations are excitatory, so all four weights are positive.
E_ratio = 0.5

w_EE = 16.667
w_IE =  0.701     # I -> E
w_EI =  0.0311    # E -> I
w_II = 10.938

# The dt = 0.05 neighbour, kept so this is a one-line swap:
# w_EE, w_IE, w_EI, w_II = 16.667, 1.048, 0.0465, 10.938   # cross = 2.236
# tau_ref_E, tau_ref_I   = 3.474, 3.087                    # tau_scale = 1.8

E_exc = 0.0
E_inh = 0.0
beta_E, beta_I = 2.5, 5.0        # 1/s_theta
theta_E, theta_I = 2.0, 0.84
tau_int_E, tau_int_I = 1.0, 1.0
tau_ref_E, tau_ref_I = 4.053, 3.602
K_ref_E, K_ref_I = 30.0, 30.0

# dt is the solver step. DT0_ANCHOR is the old time quantum, kept ONLY as the number that
# fixes this port's physical time unit through T_u = alpha_1 * dt0 (see wc-painchaud.py): the
# anchor identified Painchaud's maximal rate with the route-A ceiling lam0 = 1/dt0 = 2/ms,
# which route B does not have. UNRESOLVED -- a vault decision, not something to invent here.
# It is no longer passed to any model.
dt  = 0.1
DT0_ANCHOR = 0.5
# nu0 is the rate unit of the Poisson reference and is NOT 1/DT0_ANCHOR; it is the repo's
# 1 /ms. Stated so that the two numbers are not confused back together.
nu0 = 1.0
hazard = "poisson_exact"

# Absolute refractory period, as a hard floor on the first round(T_ABS/dt) age bins. Without
# it the exponential kernel is overwhelmed at this coupling and the population runs at >1 kHz.
T_ABS, M_SUPPRESS = 1.0, 40.0

U_MAX  = 100.0                   # ms; age grid held fixed in TIME, so Qm tracks dt
Qm     = [math.ceil(U_MAX / dt)] * 2

steps1 = int(3000 / dt)          # equilibration
steps2 = int(3000 / dt)          # recorded


# The absolute refractory window is now a model parameter, t_abs, rather than something this
# script patches in afterwards. Two things that cost us before:
#
#   The mean field carried it in R(u) and the spin model in a lambda over sm.field, and the
#   spin model's EPR does NOT read field() -- it rebuilds S + R - theta in its forward and
#   reverse passes. So the patched version had the dynamics on a 1.0 ms window and the entropy
#   production on whatever the kernel gave, silently.
#
#   kernels.refractory_kernel_abs blocked round(t_abs/dt) bins, which is one too many: a
#   neuron entering bin 0 fires at the NEXT step, so blocking n bins buys a dead time of
#   (n+1)*dt. The realised dead time here was 1.1 ms, not the 1.0 ms T_ABS asks for, and it
#   moved with dt. kernels.n_abs_bins now gets this right, so these runs differ from the
#   cached ones by that one bin.
#
# R_abs is sized by kernels.abs_refractory_floor, which reproduces the hand-rolled
# `max(inc - theta + M_SUPPRESS/beta, M_SUPPRESS/beta)` this script used, to the last bit.


# Model initialization
# ~~~~~~~~~~~~~~~~~~~~
torch.set_default_dtype(torch.float64)

sm = SpinWilsonCowan(N, E_ratio, w_EE, w_EI, w_IE, w_II, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I, 
                    dt=dt, nu0=nu0, hazard=hazard, t_abs=T_ABS,
                    device=device, ic="silent")

# Both models take the weights in physical units -- no /dt, since both build the input
# from the rate. "synchronous" is "bernoulli" at nu0*dt = 1 under route B.
mf = RDMWilsonCowan(E_ratio, w_EE, w_EI, w_IE, w_II, 
                    E_exc, E_inh, beta_E, beta_I, theta_E, theta_I, 
                    tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I,
                    dt=dt, nu0=nu0, hazard=hazard, t_abs=T_ABS, Qm=Qm, device=device)

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
    fig.set_label('entropy-trajectories')   # the figure's filename slug
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
    fig.set_label('per-population-trajectories')   # the figure's filename slug

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
    k_lyap   = 2                    # lambda_1 decides chaos; lambda_2 a circle from a torus
    eq_lyap  = int(8000 / dt)       # settle the trajectory
    wu_lyap  = int(1000 / dt)       # settle the tangent BASIS: a random one is not aligned
    T_lyap   = int(30000 / dt)      # accumulation; windows in ms, so dt can be refined

    # Same model as above -- same hazard, same nu0, same age grid, same window. The original
    # script built this one with hazard="synchronous" and no dt0, which here would measure a
    # different model from the one the panels show.
    lmf = RDMWilsonCowan(E_ratio, w_EE, w_EI, w_IE, w_II,
                         E_exc, E_inh, beta_E, beta_I, theta_E, theta_I,
                         tau_int_E, tau_int_I, tau_ref_E, tau_ref_I, K_ref_E, K_ref_I,
                         dt=dt, nu0=nu0, hazard=hazard, t_abs=T_ABS, Qm=Qm, device=device)

    print(f"\nLyapunov: equilibrating {eq_lyap} steps, then {wu_lyap} warmup + {T_lyap} "
          f"accumulation, k={k_lyap} ...")
    t0 = time.time()
    lmf.forward(eq_lyap)
    blocks = lmf.init_tangent(k_lyap, generator=torch.Generator(device=device).manual_seed(0))
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
    # `lmf.is_flow`, so these exponents describe a FLOW sampled every dt: a limit cycle keeps its
    # phase direction and sits at lam_1 ~ 0, which the map table would file as quasiperiodic.
    cls = lyap.classify_spectrum(spec.lam, tol=tol, moving=moving, continuous=lmf.is_flow,
                                 se=spec.se, n_sigma=3.0)
    print(f"  CV(m_E) = {cv:.3e} -> moving={bool(moving)};  band = +/-{tol:.1e} /ms")
    print(f"  => {lyap.CLASS_NAMES[int(cls)]}")
    if int(cls) == lyap.CHAOS:
        print("  NOTE: confirm at dt <= 0.1 before believing this. On the parameter sweep every"
              "\n        chaotic candidate at dt = 0.2 flipped sign at dt = 0.1.")

rp.show()