from pathlib import Path
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

from rdme.batch import RDMIsingModelBatch, RDMNetworkBatch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[1] / "cache"
CACHE_TRAJ_FILE = CACHE_DIR / f"{bname}_traj.pt"
CACHE_SIM_FILE  = CACHE_DIR / f"{bname}_sim.pt"
CACHE_FP_FILE   = CACHE_DIR / f"{bname}_fp.pt"

run_sim   = True
run_fp    = True
run_plot  = True
overwrite = True

torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    J       = 10
    E       = torch.linspace(0.7, 2.0, 300)
    beta    = 40.0
    theta   = 1.0
    tau_int = 20.0
    tau_ref = 3.0
    K_ref   = 0.0
    dt      = 0.2

    equi  = 20000
    steps = 30000

    if CACHE_TRAJ_FILE.exists() and not overwrite:
        print(f"Simulation already exists at {CACHE_TRAJ_FILE}. Skipping.")
    else:
        # J in physical units (srm builds the input from the rate), and the sigmoid kernel
        # of the original model
        mf = RDMIsingModelBatch(J=J, E=E, beta=beta, theta=theta,
                                tau_int=tau_int, tau_ref=tau_ref, K_ref=K_ref,
                                device=device, dt=dt, hazard="synchronous", eps=0.01)
        print(f"Running {mf.B} mean-fields for {equi+steps} steps on {device}...")
        t0 = time.time()
        mf.forward(equi, pb=True)
        traj = mf.entropy_trajectory(steps, pb=True)
        traj["dt"] = torch.tensor(dt)
        print(f"Done in {time.time() - t0:.2f}s")

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(traj, CACHE_TRAJ_FILE)
        print(f"Trajectory saved to {CACHE_TRAJ_FILE}.")
        mf.save(CACHE_SIM_FILE)
        print(f"Simulation state saved to {CACHE_SIM_FILE}.")


# ── Fixed points ──────────────────────────────────────────────────────────────
#
# Independent of the trajectory above: the fixed points come from the self-consistency
# condition m = F(m), one O(Q) pass per evaluation and no time integration, so the whole
# branch diagram costs less than a single mean-field trajectory. The spectrum does not --
# it is a dense eigendecomposition of the (2Q, 2Q) one-step Jacobian, a few seconds per
# point at this Q -- so it is computed on a strided subset of the E axis and cached.

if run_fp:

    if not CACHE_SIM_FILE.exists():
        raise FileNotFoundError(f"{CACHE_SIM_FILE} not found. Run the simulation first.")

    n_lead   = 8    # leading eigenvalues kept per fixed point
    n_E_fp   = 50   # E values to analyse; the cost is dominated by the eigendecomposition
    n_scan   = 401  # resolution of the sign scan that brackets the roots

    if CACHE_FP_FILE.exists() and not overwrite:
        print(f"Fixed points already exist at {CACHE_FP_FILE}. Skipping.")
    else:
        # on CPU: the fixed-point pass is tiny and LAPACK's eig has no GPU equivalent here
        mf_fp = RDMNetworkBatch.load(CACHE_SIM_FILE, device="cpu")
        stride = max(1, mf_fp.B // n_E_fp)
        idx    = torch.arange(0, mf_fp.B, stride)
        E_axis = mf_fp.grid_axes["E"]

        print(f"Finding fixed points at {len(idx)} of {mf_fp.B} E values (Q={mf_fp.Qm[0]})...")
        t0 = time.time()
        E_of_root, m_star, rho, lead = [], [], [], []
        for b in idx:
            sys = mf_fp.select(int(b))
            # M=1, so the scan over [0,1] is exhaustive up to its resolution: it picks up
            # unstable branches that Newton would only reach from a narrow basin
            for r in sys.fixed_points(method='bracket', n_grid=n_scan):
                ev = sys.fp_eigenvalues(r)
                ev = ev[ev.abs().argsort(descending=True)][:n_lead]
                E_of_root.append(E_axis[b])
                m_star.append(r[0])
                rho.append(ev.abs().max())
                lead.append(ev)
        print(f"Done in {time.time() - t0:.2f}s, {len(m_star)} fixed points.")

        fp = {"E": torch.stack(E_of_root), "m_star": torch.stack(m_star),
              "rho": torch.stack(rho), "lead": torch.stack(lead),
              "dt": torch.tensor(mf_fp.dt)}
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(fp, CACHE_FP_FILE)
        print(f"Fixed points saved to {CACHE_FP_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    for f in (CACHE_TRAJ_FILE, CACHE_SIM_FILE):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run simulation first.")

    traj = torch.load(CACHE_TRAJ_FILE, weights_only=True)
    mf   = RDMNetworkBatch.load(CACHE_SIM_FILE, device="cpu")
    E    = mf.grid_axes["E"]   # the only swept axis, so every output stays (n_E, T)
    fp   = torch.load(CACHE_FP_FILE, weights_only=True) if CACHE_FP_FILE.exists() else None

    n_points = 1000  # tail points per E shown in the bifurcation-diagram scatter
    # (1/ms) y-limit for the spectrogram panel; None = full Nyquist range. The arg(λ)
    # overlay lives at ~0.05/ms here, so it needs a small value to be legible at all.
    max_freq = 0.2

    dt_    = traj["dt"].item()
    dt_fp  = fp["dt"].item() if fp is not None else dt_
    stable = (fp["rho"] < 1.0) if fp is not None else None
    # last E at which every fixed point is still an attractor: with a single branch this is
    # the instability threshold, and the bifurcation scatter should fan out there
    E_crit = (fp["E"][stable].max() if stable.any() and not stable.all() else None) \
             if fp is not None else None

    def new_figure(title: str, n_rows: int, height: float):
        """ One figure per topic, all sharing the E axis and the parameter banner. The
        parameters are read off the loaded model rather than the simulation block above, so
        the plot half stands alone with run_sim = False. """
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, height), sharex=True)
        axes = [axes] if n_rows == 1 else list(axes)
        fig.suptitle(f'{title}    —    RDM Ising mean-field  '
                     f'J={mf.w[0, 0, 0].item()*dt_:g},  β={mf.beta[0, 0].item():g},  '
                     f'τ_int={mf.tau_int[0, 0].item():g},  τ_ref={mf.tau_ref[0, 0].item():g},  '
                     f'K_ref={mf.K_ref[0, 0].item():g}', fontsize=13)
        for ax in axes:
            ax.grid()
            if E_crit is not None:
                ax.axvline(E_crit, color='0.4', linewidth=1, linestyle=':')
        axes[-1].set_xlabel('External input  E')
        return fig, axes


    # ── Figure 1: bifurcation ─────────────────────────────────────────────────
    # The simulated attractor and the fixed points it is built from, on a common axis: in
    # the stable region the scatter collapses onto the branch, past the threshold the
    # branch stays at the centre of the oscillation the scatter spreads over.

    fig_bif, (ax_scatter, ax_branch) = new_figure('Bifurcation', 2, 8)

    ax_scatter.set_title(f'Simulated activity (last {n_points} points per E)')
    tail   = traj["m_tot"][:, -n_points:]                     # (B, n_points)
    E_tail = E.unsqueeze(1).expand_as(tail)
    ax_scatter.scatter(E_tail.reshape(-1), tail.reshape(-1)/dt_, s=1, alpha=0.2,
                       color='k', linewidths=0)
    ax_scatter.plot(E, traj["m_tot"].mean(dim=1)/dt_, linewidth=1, color='C1',
                    label='time average')
    ax_scatter.set_ylabel('m/dt  (1/ms)')
    ax_scatter.legend()

    ax_branch.set_title('Mean-field fixed points  (filled = stable, open = unstable)')
    if fp is not None:
        ax_branch.plot(E, traj["m_tot"].mean(dim=1)/dt_, color='C1', linewidth=1,
                       label='simulated time average')
        for mask, style, lab in [
                (stable,  dict(marker='o', color='C0'), 'stable'),
                (~stable, dict(marker='o', facecolors='none', edgecolors='C3'), 'unstable')]:
            if mask.any():
                ax_branch.scatter(fp["E"][mask], fp["m_star"][mask]/dt_fp, s=22,
                                  label=lab, **style)
        ax_branch.legend()
    else:
        ax_branch.set_title('No fixed points: run with run_fp = True')
    ax_branch.set_ylabel('m*/dt  (1/ms)')


    # ── Figure 2: stability ───────────────────────────────────────────────────
    # The spectrum of the one-step Jacobian at each fixed point, against the oscillation the
    # simulation actually shows: |λ| says whether the fixed point is an attractor, arg(λ)
    # says at what frequency it fails to be one, and the spectrogram is the check on both.

    fig_stab, (ax_mod, ax_spec) = new_figure('Stability', 2, 8)

    ax_mod.set_title('Leading eigenvalues of the one-step Jacobian at each fixed point')
    if fp is not None:
        # the subleading moduli give the margin: a gap means the leading pair really is the
        # mode that goes unstable, no gap means the crossing is not a clean Hopf
        for k in range(1, fp["lead"].shape[1]):
            ax_mod.scatter(fp["E"], fp["lead"][:, k].abs(), s=4, color='0.75',
                           label='subleading' if k == 1 else None)
        ax_mod.scatter(fp["E"], fp["lead"][:, 0].abs(), s=22,
                       c=['C0' if s else 'C3' for s in stable], label='leading |λ|', zorder=3)
        ax_mod.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label='unit circle')
        ax_mod.legend()
    else:
        ax_mod.set_title('No eigenvalues: run with run_fp = True')
    ax_mod.set_ylabel('|λ|')

    ax_spec.set_title('Activity spectrogram')
    ax_spec.grid(False)                       # the mesh is the data; a grid over it is noise
    sig   = traj["m_tot"] - traj["m_tot"].mean(dim=1, keepdim=True)  # (B, T), DC-removed per E
    power = torch.fft.rfft(sig, dim=1).abs() ** 2                    # (B, F)
    freqs = torch.fft.rfftfreq(sig.shape[1], d=dt_)                  # (F,) cycles/ms
    db    = 10 * torch.log10(power.T + 1e-12)                        # (F, B)
    pcm = ax_spec.pcolormesh(E.numpy(), freqs.numpy(), db.numpy(), shading='auto', cmap='magma')
    # the colorbar gets its own slot carved out of ax_spec, and ax_mod gets an identical
    # slot left blank -- attaching it to ax_spec alone would narrow only that panel and
    # slide the two E axes out of register despite the sharex
    cax = make_axes_locatable(ax_spec).append_axes("right", size="2%", pad=0.15)
    fig_stab.colorbar(pcm, cax=cax, label='Power (dB)')
    make_axes_locatable(ax_mod).append_axes("right", size="2%", pad=0.15).set_axis_off()
    if max_freq is not None:
        ax_spec.set_ylim(0, max_freq)
    if fp is not None:
        # arg(λ) of the leading mode is the oscillation the linearization predicts; it should
        # sit on the spectrogram's ridge wherever that mode is the one driving the dynamics
        osc = fp["lead"][:, 0].angle().abs() / (2 * torch.pi) / dt_fp
        ax_spec.plot(fp["E"], osc, color='c', linewidth=1.5, linestyle='--',
                     label='arg(λ) of the leading mode')
        ax_spec.legend(loc='upper left')
    ax_spec.set_ylabel('Frequency (1/ms)')


    # ── Figure 3: entropy production ──────────────────────────────────────────

    fig_ent, (ax_sigma, ax_H) = new_figure('Entropy production', 2, 8)

    ax_sigma.set_title('Entropy production rate  Σ')
    ax_sigma.plot(E, traj["sigma_tot"].mean(dim=1)/dt_, linewidth=2)
    ax_sigma.set_ylabel('Σ  (nats/ms)')

    ax_H.set_title('Conditional entropies')
    ax_H.plot(E, traj["H_fwd_tot"].mean(dim=1)/dt_, linewidth=2, label='Forward')
    ax_H.plot(E, traj["H_rev_tot"].mean(dim=1)/dt_, linewidth=2, label='Backward')
    ax_H.set_ylabel('H  (nats/ms)')
    ax_H.legend()

    for fig in (fig_bif, fig_stab, fig_ent):
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
