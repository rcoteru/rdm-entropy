from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch

from rdme.mean_field import RDMIsingModelBatch

# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[2] / "cache"
CACHE_TRAJ_FILE = CACHE_DIR / f"{bname}_traj.pt"

run_sim   = True
run_plot  = True
overwrite = False

torch.set_default_dtype(torch.float64)


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    J       = 1.5
    I       = torch.linspace(0.7, 1.4, 200)
    beta    = 30.0
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
        mf = RDMIsingModelBatch(J=J/dt, I=I, beta=beta, theta=theta,
                                 tau_int=tau_int, tau_ref=tau_ref, K_ref=K_ref,
                                 device=device)
        print(f"Running {mf.B} mean-fields for {equi+steps} steps on {device}...")
        t0 = time.time()
        mf.forward(equi, pb=True)
        traj = mf.entropy_trajectory(steps, pb=True)
        traj["I"]  = I.cpu()
        traj["dt"] = torch.tensor(dt)
        print(f"Done in {time.time() - t0:.2f}s")

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(traj, CACHE_TRAJ_FILE)
        print(f"Trajectory saved to {CACHE_TRAJ_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_TRAJ_FILE.exists():
        raise FileNotFoundError(f"Trajectory not found at {CACHE_TRAJ_FILE}. Run simulation first.")

    traj = torch.load(CACHE_TRAJ_FILE, weights_only=True)
    I = traj["I"]

    n_points = 1000  # tail points per I shown in the bifurcation-diagram scatter
    max_freq = None   # (1/ms) y-limit for the spectrogram panel; None = full Nyquist range

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 16))
    for ax in (ax1, ax2, ax3, ax4):
        ax.sharex(ax5)

    ax1.set_title('Activity (bifurcation diagram: last %d points per I)' % n_points)
    traj_avg = traj["a_pop"].mean(dim=1)
    tail = traj["a_pop"][:, -n_points:]                       # (B, n_points)
    I_tail = I.unsqueeze(1).expand_as(tail)                   # (B, n_points)
    ax1.scatter(I_tail.reshape(-1), tail.reshape(-1), s=1, alpha=0.2, color='k', linewidths=0)
    ax1.plot(I, traj_avg, linewidth=1, color='C1')
    ax1.grid()

    ax2.set_title('Sigma')
    traj_avg, traj_std = traj["sigma_tot"].mean(dim=1), traj["sigma_tot"].std(dim=1)
    ax2.plot(I, traj_avg, linewidth=2)
    # ax2.fill_between(I, traj_avg - traj_std, traj_avg + traj_std, alpha=0.3)
    ax2.grid()

    ax3.set_title('Conditional Entropies')
    ax3.plot(I, traj["S_fwd_tot"].mean(dim=1), linewidth=2, label="Forward Entropy")
    ax3.plot(I, traj["S_rev_tot"].mean(dim=1), linewidth=2, label="Backward Entropy")
    ax3.grid(); ax3.legend()

    dI    = torch.diff(I)
    I_mid = (I[:-1] + I[1:]) / 2
    ax4.set_title('Differences vs I')
    sigma_avg = traj["sigma_tot"].mean(dim=1)
    ax4.plot(I_mid, torch.diff(traj["sigma_tot"].mean(dim=1)) / dI, linewidth=2, label="dΣ/dI")
    ax4.plot(I_mid, torch.diff(traj["a_pop"].mean(dim=1)) / dI, linewidth=2, label="dm/dI")
    ax4.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax4.grid(); ax4.legend()


    ax5.set_title('Activity spectrogram vs I')
    dt_sim = traj.get("dt", torch.tensor(1.0)).item()
    sig   = traj["a_pop"] - traj["a_pop"].mean(dim=1, keepdim=True)   # (B, T), DC-removed per I
    power = torch.fft.rfft(sig, dim=1).abs() ** 2                     # (B, F)
    freqs = torch.fft.rfftfreq(sig.shape[1], d=dt_sim)                # (F,) cycles/ms
    db    = 10 * torch.log10(power.T + 1e-12)                         # (F, B)
    pcm = ax5.pcolormesh(I.numpy(), freqs.numpy(), db.numpy(), shading='auto', cmap='magma')
    fig.colorbar(pcm, ax=ax5, label='Power (dB)')
    if max_freq is not None:
        ax5.set_ylim(0, max_freq)
    ax5.set_ylabel('Frequency (1/ms)')
    ax5.set_xlabel('External input  I')
    ax5.plot(I, traj["sigma_tot"].mean(dim=1), linewidth=2)

    fig.suptitle(f'RDM Ising mean-field  J={J},  β={beta},  τ_int={tau_int},  τ_ref={tau_ref},  K_ref={K_ref}', fontsize=14)
    # plt.tight_layout()
    plt.show()
