# FROZEN (2026-11-07). This file reproduces stale route-A `D1` figures and is deliberately
# NOT ported to the route-B API: run it from `git checkout 378bf2a^`, the last commit whose
# `rdme` it matches. Against the current library it will not even import the names it wants.
# The `S1` maps replace these figures; `D1` stays in rdme.presets as the record of their
# parameters and has no live caller.

""" One-dimensional cuts through the (E, J) plane of fig_ising_EJ_maps: the drive sweep at
fixed coupling and the coupling sweep at fixed drive, side by side.

Two figures, each 2x2 with one column per sweep:

  fig_ising_sweeps_bifurcation -- (row 1) the simulated attractor, its time average and the
      mean-field fixed points marked by linear stability; (row 2) the entropy production
      against the two conditional entropies, on twin axes.
  fig_ising_sweeps_stability   -- (row 1) the leading eigenvalues of the one-step Jacobian
      at each fixed point; (row 2) the activity spectrogram, with the frequencies predicted
      by arg(lambda) of the first and third modes drawn over it.
  fig_ising_sweeps_spectrum    -- that second row on its own, for use where the eigenvalue
      panels are not wanted.

Self-contained in the same sense as fig_ising_EJ_maps: all parameters live here, the cache
is this directory's, and only the `rdme` API is shared. The cache holds reductions -- time
averages, the scatter tail, the cropped power spectrum and the fixed-point branches -- and
not the trajectories, which are several hundred MB per sweep.

    python papers/scipost/fig_ising_sweeps.py
"""

from pathlib import Path
import time

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy
import torch

import rdme.helpers as hlp
from rdme.rdm_batch import RDMIsingModelBatch


# ── Paths & switches ──────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
CACHE_DIR = HERE / "cache"
FIG_DIR   = HERE / "figures"

run_sim   = True
run_plot  = True
overwrite = False
torch.set_default_dtype(torch.float64)


# ── Parameters ────────────────────────────────────────────────────────────────

# Shared with fig_ising_EJ_maps; the two cuts differ only in which of (E, J) is the axis.
BASE = dict(beta=30.0, theta=1.0, tau_int=20.0, tau_ref=3.0, K_ref=0.0,
            dt=0.5, dt0=0.5, hazard="async", eps=0.01)

SWEEPS = {
    "E": dict(axis="E", label=r"external drive $E$", fixed="J",
              J=6.0, E=torch.linspace(0.6, 1.8, 300)),
    "J": dict(axis="J", label=r"recurrent coupling $w$", fixed="E",
              J=torch.linspace(0.0, 17.0, 300), E=0.9),
}

EQUI, STEPS = 20000, 30000
N_TAIL   = 1000    # trajectory points per parameter value kept for the bifurcation scatter
N_FP     = 100     # parameter values at which fixed points and spectra are computed
N_SCAN   = 401     # resolution of the sign scan that brackets the roots
N_LEAD   = 4       # eigenvalues kept per fixed point, largest modulus first
MAX_FREQ = 100.0   # (Hz) y-limit of the spectrogram panels
KHZ_TO_HZ = 1e3    # dt is in ms, so every frequency derived from it comes out in kHz
N_WELCH  = 16      # segments the record is split into for the averaged periodogram
DB_RANGE = 60      # dynamic range of the spectrogram colour scale, below each panel's peak
N_FBINS  = 400     # frequency bins the spectrogram is max-pooled into before drawing


def cache_file(key: str) -> Path:
    return CACHE_DIR / f"ising_{key}_sweep.pt"


# ── Simulation ────────────────────────────────────────────────────────────────

def run_sweep(key: str) -> dict:
    """ One sweep: trajectories for the attractor and the spectrum, then the fixed-point
    branches and their spectra. The two halves are independent -- the fixed points come from
    the self-consistency condition m = F(m) and never integrate in time -- which is what lets
    the unstable branches appear at all. """
    spec   = SWEEPS[key]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dt     = BASE["dt"]

    mf = RDMIsingModelBatch(J=spec["J"], E=spec["E"], device=device, **BASE)
    x  = mf.grid_axes[spec["axis"]]
    print(f"[{key}] {mf.B} mean-fields, {EQUI+STEPS} steps on {device} (Q={mf.Qm[0]})...")

    t0 = time.time()
    mf.forward(EQUI, pb=True)
    traj = mf.entropy_trajectory(STEPS, pb=True)
    print(f"[{key}] trajectories done in {time.time() - t0:.1f}s")

    m = traj["m_tot"]                                       # (B, T)
    # Welch estimate of the activity spectrum, over the whole band up to Nyquist. A plain
    # periodogram is not consistent -- its variance stays put as the record grows, giving a
    # map streaked with notches that look like structure and are not -- so the record is
    # split into half-overlapping Hann-windowed segments and their periodograms averaged.
    sig    = m - m.mean(dim=1, keepdim=True)
    L      = sig.shape[1] // N_WELCH
    win    = torch.hann_window(L, periodic=False, dtype=sig.dtype)
    starts = range(0, sig.shape[1] - L + 1, L // 2)
    power  = torch.stack([torch.fft.rfft(sig[:, i:i+L] * win, dim=1).abs() ** 2
                          for i in starts]).mean(dim=0)
    freqs  = torch.fft.rfftfreq(L, d=dt)                    # cycles/ms

    out = {
        "x": x, "dt": torch.tensor(dt),
        "m_mean":     m.mean(dim=1),
        "sigma_mean": traj["sigma_tot"].mean(dim=1),
        "H_fwd_mean": traj["H_fwd_tot"].mean(dim=1),
        "H_rev_mean": traj["H_rev_tot"].mean(dim=1),
        "tail":   m[:, -N_TAIL:].clone(),
        "freqs":  freqs,
        # dB, and single precision: this is a colour scale, not a number anyone reads off
        "power_db": (10 * torch.log10(power + 1e-12)).T.float(),
    }
    del traj, m, sig, power

    # Fixed points on CPU: the pass itself is tiny, and the dense eigendecomposition of the
    # (2Q, 2Q) one-step Jacobian is the cost here, so it runs on a strided subset. The model
    # is rebuilt rather than moved -- the fixed points are a property of the parameters, not
    # of the state the trajectory above left behind, and Qm is derived identically.
    mf_cpu = RDMIsingModelBatch(J=spec["J"], E=spec["E"], device="cpu", **BASE)
    idx    = torch.arange(0, mf.B, max(1, mf.B // N_FP))
    print(f"[{key}] fixed points at {len(idx)} of {mf.B} values...")

    t0 = time.time()
    fp_x, fp_m, fp_rho, fp_lead = [], [], [], []
    for b in idx:
        sys = mf_cpu.select(int(b))
        # M=1, so the scan over [0, 1] is exhaustive up to its resolution: it picks up the
        # unstable branches that Newton would only reach from a narrow basin
        for r in sys.fixed_points(method='bracket', n_grid=N_SCAN):
            ev = sys.fp_eigenvalues(r)
            ev = ev[ev.abs().argsort(descending=True)][:N_LEAD]
            fp_x.append(x[b]); fp_m.append(r[0])
            fp_rho.append(ev.abs().max()); fp_lead.append(ev)
    print(f"[{key}] {len(fp_m)} fixed points in {time.time() - t0:.1f}s")

    out |= {"fp_x": torch.stack(fp_x), "fp_m": torch.stack(fp_m),
            "fp_rho": torch.stack(fp_rho), "fp_lead": torch.stack(fp_lead)}
    # the grid axes come back on whatever device the batch ran on; the cache is device-free
    return {k: v.cpu() for k, v in out.items()}


if run_sim:
    for key in SWEEPS:
        if cache_file(key).exists() and not overwrite:
            print(f"{cache_file(key)} exists and overwrite is off. Skipping.")
            continue
        data = run_sweep(key)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache_file(key))
        print(f"Saved {cache_file(key)}.")


# ── Figures ───────────────────────────────────────────────────────────────────

if run_plot:

    missing = [str(cache_file(k)) for k in SWEEPS if not cache_file(k).exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} not found. Run with run_sim = True.")

    hlp.pub_style()
    data = {k: torch.load(cache_file(k), weights_only=True, map_location="cpu") for k in SWEEPS}

    C_AVG, C_STABLE, C_UNSTABLE = "k", "C0", "C3"
    C_SIGMA, C_FWD, C_REV       = "k", "C0", "C3"

    def pool_spectrum(db: torch.Tensor, f: torch.Tensor, fmax: float):
        """ Max-pool the spectrum onto N_FBINS frequency bins below fmax. The rfft has tens
        of thousands of bins against a few hundred pixels of panel, and pcolormesh samples
        rather than reduces, so a one-bin ridge -- which is exactly what a periodic orbit
        gives -- lands on screen only if it happens to fall on a sampled row. Pooling by the
        maximum keeps every peak at the cost of widening it to one pooled bin. """
        keep = f <= fmax
        db, f = db[keep], f[keep]
        if len(f) <= N_FBINS:                 # already coarser than the target
            return db, f
        # pad up to a whole number of pooled bins rather than truncating, so the top of the
        # band -- where a period-2 orbit lives -- is not the part that gets dropped
        w   = -(-len(f) // N_FBINS)
        pad = w * N_FBINS - len(f)
        dbp = torch.cat([db, db.new_full((pad, db.shape[1]), -torch.inf)])
        # the rfft grid is uniform, so the pooled centres are just the padded band split
        # into N_FBINS -- taking them off f would land on the NaN padding in the last bin
        df = (f[1] - f[0]) * w
        centres = f[0] + (torch.arange(N_FBINS) + 0.5) * df
        return dbp.reshape(N_FBINS, w, db.shape[1]).amax(dim=1), centres

    def grid(height: float = 0.62):
        """ 2x2, one column per sweep, columns sharing their parameter axis. """
        fig, axes = plt.subplots(2, 2, constrained_layout=True, sharex="col", sharey="row",
                                 figsize=(hlp.SCIPOST_TEXTWIDTH,
                                          height * hlp.SCIPOST_TEXTWIDTH))
        for ax in axes.flat:
            ax.tick_params(direction="out", top=False, right=False)
        return fig, axes

    def label_panels(axes, titles):
        for ax, t in zip(axes.flat, titles):
            ax.set_title(t, loc="left", pad=4)

    def save(fig, name):
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(FIG_DIR / f"{name}.{ext}")
        print(f"Figure saved to {FIG_DIR / f'{name}.pdf'}.")


    # ── Figure 1: bifurcation and entropy ─────────────────────────────────────

    fig1, ax1 = grid(0.66)
    twins = []

    for col, key in enumerate(SWEEPS):
        d      = data[key]
        x      = d["x"]
        stable = d["fp_rho"] < 1.0

        # (a, b) attractor, its time average and the branches it is built from. In the
        # stable region the scatter collapses onto the branch; past the instability the
        # branch stays at the centre of the oscillation the scatter spreads over.
        ax = ax1[0, col]
        tail = d["tail"]
        ax.scatter(x.unsqueeze(1).expand_as(tail).reshape(-1), tail.reshape(-1),
                   s=0.3, alpha=0.06, color="0.5", linewidths=0, rasterized=True,
                   label="attractor")
        for mask, kw, lab in [
                (stable,  dict(color=C_STABLE), "stable"),
                (~stable, dict(facecolors="none", edgecolors=C_UNSTABLE), "unstable")]:
            if mask.any():
                ax.scatter(d["fp_x"][mask], d["fp_m"][mask], s=4, linewidths=0.5,
                           label=f"fixed point, {lab}", zorder=3, **kw)
        # last, and above the branches: where the fixed point is stable the two coincide,
        # and the average is the one that has to stay readable
        ax.plot(x, d["m_mean"], color=C_AVG, lw=0.9, label="time average", zorder=4)
        if col == 0:
            ax.set_ylabel(r"activity $m$")

        # (c, d) the entropy production against the entropies it is the difference of, all
        # per neuron and per bin as in fig_ising_EJ_maps. Two axes because Sigma is the
        # small difference of two much larger numbers: on one scale it would be a flat line.
        ax = ax1[1, col]
        ax.plot(x, d["sigma_mean"], color=C_SIGMA, lw=1.2, label=r"$\Sigma$")
        if col == 0:
            ax.set_ylabel(r"$\Sigma$  (nats)")
        ax.set_xlabel(SWEEPS[key]["label"])

        axH = ax.twinx()
        twins.append(axH)
        axH.plot(x, d["H_fwd_mean"], color=C_FWD, lw=1.0, ls="--", label=r"$H$")
        axH.plot(x, d["H_rev_mean"], color=C_REV, lw=1.0, ls=":", label=r"$\tilde H$")
        # the twins are shared below, so only the outer one carries numbers
        axH.tick_params(direction="out", right=True, labelright=(col == 1))
        if col == 1:
            axH.set_ylabel(r"$H,\ \tilde H$  (nats)")

        if col == 0:
            ax1[0, col].legend(loc="upper left", handletextpad=0.4, borderpad=0.3,
                               markerscale=2)
            lines = ax.get_lines() + axH.get_lines()
            ax.legend(lines, [ln.get_label() for ln in lines], loc="upper left",
                      handletextpad=0.4, borderpad=0.3)

    # sharey on the subplot grid does not reach the twins, which are created afterwards
    twins[1].sharey(twins[0])

    label_panels(ax1, ["(a) drive sweep", "(b) coupling sweep",
                       "(c) entropy production", "(d) entropy production"])
    save(fig1, "ising_sweeps_bifurcation")


    # ── Figure 2: stability and spectrum ──────────────────────────────────────

    # a 2x3 grid whose last column is a sliver: the colour bar belongs to the spectrograms,
    # and attached to the whole figure it would run up the side of the eigenvalue panels,
    # which have nothing to do with it. The sliver is left empty on the top row, so both
    # rows still give up the same width and the columns stay in register.
    fig2 = plt.figure(constrained_layout=True,
                      figsize=(hlp.SCIPOST_TEXTWIDTH, 0.66 * hlp.SCIPOST_TEXTWIDTH))
    gs   = fig2.add_gridspec(2, 3, width_ratios=[1, 1, 0.04])
    ax2  = numpy.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            ax2[r, c] = fig2.add_subplot(gs[r, c],
                                         sharex=ax2[0, c] if r else None,
                                         sharey=ax2[r, 0] if c else None)
            ax2[r, c].tick_params(direction="out", top=False, right=False,
                                  labelleft=(c == 0), labelbottom=(r == 1))
    cax = fig2.add_subplot(gs[1, 2])

    # Each panel is normalised to its own peak and shown over a fixed range below it, so
    # one colour bar serves both: the two sweeps differ by 20 dB in absolute power, and on
    # a shared absolute scale the weaker panel reads as empty. What the colour carries is
    # therefore the shape of each spectrum, not a comparison between the two.
    db_lim = dict(vmin=-DB_RANGE, vmax=0.0)

    for col, key in enumerate(SWEEPS):
        d      = data[key]
        dt     = d["dt"].item()
        stable = d["fp_rho"] < 1.0
        lead   = d["fp_lead"]

        # (a, b) the subleading moduli give the margin: a gap means the leading pair really
        # is the mode that goes unstable, no gap means the crossing is not a clean Hopf.
        ax = ax2[0, col]
        for k in range(1, lead.shape[1]):
            ax.scatter(d["fp_x"], lead[:, k].abs(), s=2, color="0.75", linewidths=0,
                       label="subleading" if k == 1 else None)
        ax.scatter(d["fp_x"][stable], lead[stable, 0].abs(), s=6, color=C_STABLE,
                   linewidths=0, label=r"leading $|\lambda|$, stable", zorder=3)
        ax.scatter(d["fp_x"][~stable], lead[~stable, 0].abs(), s=6, color=C_UNSTABLE,
                   linewidths=0, label=r"leading $|\lambda|$, unstable", zorder=3)
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        if col == 0:
            ax.set_ylabel(r"$|\lambda|$")

        # (c, d) the spectrogram is the check on the linearization: arg(lambda)/2*pi*dt is
        # the frequency at which each mode oscillates, and it should sit on a ridge wherever
        # that mode drives the dynamics. The first and third modes are the two leading
        # conjugate pairs, i.e. the two frequencies the linearization offers.
        ax = ax2[1, col]
        ax.grid(False)                  # the mesh is the data; a grid over it is noise
        db, freqs = pool_spectrum(d["power_db"], d["freqs"], MAX_FREQ / KHZ_TO_HZ)
        pcm = ax.pcolormesh(d["x"].numpy(), (freqs * KHZ_TO_HZ).numpy(),
                            (db - db.max()).numpy(),
                            shading="auto", cmap="magma", rasterized=True, **db_lim)
        for mode, c, mk in [(0, "w", "o"), (2, "c", "s")]:
            f_mode = lead[:, mode].angle().abs() / (2 * torch.pi) / dt * KHZ_TO_HZ
            # a real positive eigenvalue carries no frequency; drawn, it would sit on f = 0
            f_mode[lead[:, mode].imag.abs() < 1e-12] = float("nan")
            # markers, not a line: where several fixed points coexist the branches are
            # interleaved along the axis, and a line would join one branch to the next
            ax.scatter(d["fp_x"], f_mode, s=1.5, marker=mk, color=c, linewidths=0,
                       label=rf"$\arg\lambda_{mode+1}/2\pi\Delta_t$")
        ax.set_ylim(0, MAX_FREQ)
        ax.set_xlabel(SWEEPS[key]["label"])
        if col == 0:
            ax.set_ylabel("frequency (Hz)")
        if col == 1:
            cbar = fig2.colorbar(pcm, cax=cax)
            cbar.ax.set_title("power (dB)", pad=4)   # relative to each panel's peak
            cbar.outline.set_linewidth(0.6)
            cbar.ax.tick_params(width=0.6, size=2.5)
        if col == 0:
            ax2[0, col].legend(loc="lower right", handletextpad=0.4, borderpad=0.3)
            ax.legend(loc="lower right", handletextpad=0.4, borderpad=0.3, markerscale=5,
                      labelcolor="w", facecolor="0.15", framealpha=0.6, frameon=True)

    label_panels(ax2, ["(a) drive sweep", "(b) coupling sweep",
                       "(c) activity spectrum", "(d) activity spectrum"])
    save(fig2, "ising_sweeps_stability")


    # ── Figure 3: activity spectrum alone ─────────────────────────────────────

    # The spectrogram row of figure 2 on its own, for use where the eigenvalue panels are
    # not needed. omega_k is arg(lambda_k)/2*pi*dt, the frequency of the k-th mode. The two
    # branches are drawn in cyan and spring green: magma runs from
    # near-black purple through orange to pale yellow, so neither hue occurs anywhere in
    # the colour scale, and each marker also carries a dark halo so it stays visible where
    # it crosses a bright ridge.
    fig3 = plt.figure(constrained_layout=True,
                      figsize=(hlp.SCIPOST_TEXTWIDTH, 0.38 * hlp.SCIPOST_TEXTWIDTH))
    gs3  = fig3.add_gridspec(1, 3, width_ratios=[1, 1, 0.04])
    ax3  = numpy.empty(2, dtype=object)
    for c in range(2):
        ax3[c] = fig3.add_subplot(gs3[0, c], sharey=ax3[0] if c else None)
        ax3[c].tick_params(direction="out", top=False, right=False, labelleft=(c == 0))
    cax3 = fig3.add_subplot(gs3[0, 2])

    C_MODE = ["#00E5FF", "#00FF95"]          # cyan, spring green: absent from magma
    halo   = [path_effects.withStroke(linewidth=1.1, foreground="0.05")]

    for col, key in enumerate(SWEEPS):
        d  = data[key]
        dt = d["dt"].item()

        ax = ax3[col]
        ax.grid(False)
        db, freqs = pool_spectrum(d["power_db"], d["freqs"], MAX_FREQ / KHZ_TO_HZ)
        pcm = ax.pcolormesh(d["x"].numpy(), (freqs * KHZ_TO_HZ).numpy(),
                            (db - db.max()).numpy(),
                            shading="auto", cmap="magma", rasterized=True, **db_lim)
        for (mode, mk), c in zip([(0, "o"), (2, "s")], C_MODE):
            f_mode = d["fp_lead"][:, mode].angle().abs() / (2 * torch.pi) / dt * KHZ_TO_HZ
            f_mode[d["fp_lead"][:, mode].imag.abs() < 1e-12] = float("nan")
            ax.scatter(d["fp_x"], f_mode, s=4, marker=mk, color=c, linewidths=0,
                       path_effects=halo, zorder=3,
                       label=rf"$\omega_{mode+1}$")
        ax.set_ylim(0, MAX_FREQ)
        ax.set_xlabel(SWEEPS[key]["label"])
        if col == 0:
            ax.set_ylabel("frequency (Hz)")
            ax.legend(loc="lower right", ncols=2, handletextpad=0.3, borderpad=0.3,
                      columnspacing=1.0, markerscale=3, labelcolor="w", facecolor="0.15",
                      framealpha=0.75, frameon=True)
        else:
            cbar = fig3.colorbar(pcm, cax=cax3)
            cbar.ax.set_title("power (dB)", pad=4)
            cbar.outline.set_linewidth(0.6)
            cbar.ax.tick_params(width=0.6, size=2.5)

    for ax, t in zip(ax3, ["(a) drive sweep", "(b) coupling sweep"]):
        ax.set_title(t, loc="left", pad=4)
    save(fig3, "ising_sweeps_spectrum")

    plt.show()
