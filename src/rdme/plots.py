from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

""" The three figures the exploration scripts in `scripts/` redraw every time: a set of
labelled curves on one axes, a column of stacked time-series panels, and a row of scatter
plots with one panel per model. Each takes the trajectory dicts the models already return,
and each hands back the matplotlib objects it made, so anything these do not cover is a
normal matplotlib call on the returned axes rather than a new keyword here.

Deliberately not for the paper figures: those live in `paper/`, are drawn by hand against
`helpers.pub_style`, and should not inherit a default that a script changed. """


# Axis labels for the keys the trajectory dicts carry, so the quantity's units are written
# down once rather than in every script. Unknown keys fall back to the key itself.
LABELS = {
    "m":         'Mean Activity',
    "m_tot":     'Mean Activity',
    "r":         'Firing Rate (1/ms)',
    "r_tot":     'Firing Rate (1/ms)',
    "sigma":     'Entropy Production Rate (nats/ms)',
    "sigma_tot": 'Entropy Production Rate (nats/ms)',
    "H_fwd":     'Forward Entropy (nats)',
    "H_fwd_tot": 'Forward Entropy (nats)',
    "H_rev":     'Backward Entropy (nats)',
    "H_rev_tot": 'Backward Entropy (nats)',
}

TITLES = {
    "m":         'Activity',
    "m_tot":     'Activity',
    "r":         'Rate',
    "r_tot":     'Rate',
    "sigma":     'Entropy Production Rate',
    "sigma_tot": 'Entropy Production Rate',
    "H_fwd":     'Forward Entropy',
    "H_fwd_tot": 'Forward Entropy',
    "H_rev":     'Backward Entropy',
    "H_rev_tot": 'Backward Entropy',
}


def _np(x) -> np.ndarray:
    """ Tensor or array to a plottable array, with a trailing singleton axis dropped.

    The squeeze is not cosmetic: SpinModel.trajectory returns "m_tot" as (T, n_obs) while the
    mean field returns it as (T,), so the same key plotted from the two models would otherwise
    need a different call. """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x[..., 0] if x.ndim == 2 and x.shape[1] == 1 else x


def _time_axis(n: int, dt: float | None) -> tuple[np.ndarray, str]:
    """ Steps, or ms when dt is given. Comparing runs at different dt is the whole point of
    several of these scripts, and only the ms axis makes that comparison mean anything. """
    if dt is None:
        return np.arange(n), 'Time Steps'
    return np.arange(n) * dt, 'Time (ms)'


def curves(ys: Mapping[str, object],
           x=None,
           *,
           dt: float | None = None,
           title: str | None = None,
           xlabel: str | None = None,
           ylabel: str | None = None,
           ax: Axes | None = None,
           figsize: tuple[float, float] = (10, 4),
           **kw) -> Axes:
    """ Labelled curves on one axes: `ys` maps legend label to a 1-D sequence.

    `x` is shared by all of them; leaving it out indexes by position, and passing `dt` instead
    puts that index in ms. Extra keywords go to every `plot` call. """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    kw.setdefault("linewidth", 2)

    xs = None if x is None else _np(x)
    for label, y in ys.items():
        y = _np(y)
        ax.plot(xs if xs is not None else _time_axis(len(y), dt)[0], y, label=label, **kw)

    if xlabel is None and x is None:
        xlabel = _time_axis(0, dt)[1]

    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if len(ys) > 1:
        ax.legend()
    ax.grid()
    return ax


def panels(series: Mapping[str, Mapping[str, object]],
           keys: Sequence[str],
           *,
           dt: float | None = None,
           suptitle: str | None = None,
           titles: Mapping[str, str] | None = None,
           ylabels: Mapping[str, str] | None = None,
           col_names: Sequence[str] | None = None,
           figsize: tuple[float, float] | None = None,
           ) -> tuple[Figure, np.ndarray]:
    """ One stacked panel per key, each overlaying every entry of `series`, x shared.

    `series` maps a legend label to a trajectory dict -- the dicts the models return, passed
    straight through, e.g. {'Mean Field': mf_traj, 'Spin Model': sm_traj}. `keys` picks the
    rows; `LABELS`/`TITLES` name them unless `titles`/`ylabels` override.

    A key whose array is (T, P) is drawn as P columns, one per population, so the per-
    population and the total views are the same call on different keys. Every key in one
    figure has to agree on P, which is a real constraint rather than a limitation of the
    helper: a row of a shared-x grid cannot have its own column count.

    Returns (fig, axes) with axes always 2-D, (len(keys), P). """
    data = {label: {k: _np(traj[k]) for k in keys} for label, traj in series.items()}

    widths = {a.shape[1] if a.ndim == 2 else 1 for d in data.values() for a in d.values()}
    if len(widths) != 1:
        raise ValueError(f"keys {list(keys)} disagree on the number of columns: {sorted(widths)}")
    n_cols = widths.pop()

    if figsize is None:
        figsize = (6 * n_cols if n_cols > 1 else 12, 3 * len(keys))
    fig, axes = plt.subplots(len(keys), n_cols, sharex=True, sharey='row',
                             figsize=figsize, squeeze=False)

    for i, key in enumerate(keys):
        for c in range(n_cols):
            ax = axes[i, c]
            for label, d in data.items():
                y = d[key][:, c] if d[key].ndim == 2 else d[key]
                xs, xlabel = _time_axis(len(y), dt)
                ax.plot(xs, y, label=label, linewidth=2)

            title = (titles or {}).get(key, TITLES.get(key, key))
            if col_names is not None:
                title = f"{title} - {col_names[c]}"
            ax.set_title(title)
            ax.set_ylabel((ylabels or {}).get(key, LABELS.get(key, key)))
            ax.grid()
            if i == 0 and len(series) > 1:
                ax.legend()
            if i == len(keys) - 1:
                ax.set_xlabel(xlabel)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def scatter_grid(data: Mapping[str, tuple],
                 *,
                 suptitle: str | None = None,
                 xlabel: str | None = None,
                 ylabel: str | None = None,
                 s: float = 1,
                 figsize: tuple[float, float] | None = None,
                 **kw) -> tuple[Figure, np.ndarray]:
    """ One scatter panel per entry of `data`, side by side on shared axes limits.

    `data` maps a panel title to an (x, y) pair. Shared limits are the point: these panels
    exist to be compared with each other, and two orbits on independently scaled axes look
    alike whatever they are doing. Returns (fig, axes), axes 1-D. """
    if figsize is None:
        figsize = (5 * len(data), 5)
    fig, axes = plt.subplots(1, len(data), figsize=figsize, sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, (title, (x, y)) in zip(axes, data.items()):
        ax.set_title(title)
        ax.scatter(_np(x), _np(y), s=s, **kw)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def maps(fields: Mapping[str, object],
         *,
         extent: tuple[float, float, float, float],
         xlabel: str | None = None,
         ylabel: str | None = None,
         suptitle: str | None = None,
         ncols: int = 2,
         cbar_labels: Mapping[str, str] | None = None,
         figsize: tuple[float, float] | None = None,
         **kw) -> tuple[Figure, np.ndarray]:
    """ A grid of phase maps: `fields` maps a panel title to a 2-D array over the same axes.

    `extent` is matplotlib's (x_min, x_max, y_min, y_max) and is shared, which is the whole
    reason this is one call -- the panels are different quantities over one parameter plane,
    and the plane has to be drawn the same way in each. Each panel gets its own colorbar,
    labelled by the panel title unless `cbar_labels` says otherwise.

    Only the outer axes are labelled, so the inner ones are not repeated. Returns (fig, axes)
    with axes 2-D. Extra keywords go to every `imshow`. """
    kw.setdefault("origin", 'lower')
    kw.setdefault("aspect", 'auto')

    n_rows = -(-len(fields) // ncols)
    if figsize is None:
        figsize = (6 * ncols, 3 * n_rows)
    fig, axes = plt.subplots(n_rows, ncols, sharex=True, sharey=True,
                             figsize=figsize, squeeze=False)

    for i, (title, field) in enumerate(fields.items()):
        ax = axes[i // ncols, i % ncols]
        ax.set_title(title)
        im = ax.imshow(_np(field), extent=extent, **kw)
        fig.colorbar(im, ax=ax, label=(cbar_labels or {}).get(title, title))
        if i % ncols == 0:
            ax.set_ylabel(ylabel or "")
        if i // ncols == n_rows - 1:
            ax.set_xlabel(xlabel or "")
        # no grid: over a filled map the lines sit on top of the data and read as structure

    for j in range(len(fields), n_rows * ncols):      # blank any unused cell
        axes[j // ncols, j % ncols].set_axis_off()

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def return_maps(ms: Mapping[str, object], **kw) -> tuple[Figure, np.ndarray]:
    """ m_t against m_{t+1}, one panel per entry of `ms`. Thin shorthand for the scatter_grid
    call the scripts all write out: the lag is the only thing it decides. """
    kw.setdefault("xlabel", r'$m_t$')
    kw.setdefault("ylabel", r'$m_{t+1}$')
    pairs = {label: (_np(m)[:-1], _np(m)[1:]) for label, m in ms.items()}
    return scatter_grid(pairs, **kw)


# 1-D parameter sweeps
# ~~~~~~~~~~~~~~~~~~~~
#
# The standard figure set of the sweep scripts: what the simulation settles onto, what the
# linearization says it should settle onto, and the entropy production along the way. Model-
# specific, unlike everything above, and here rather than in a script because ising-E_sweep
# and ising-J_sweep differ only in which parameter is on the x axis.

def sweep_figures(axis: torch.Tensor,            # (B,) the swept parameter
                  m_tot: torch.Tensor,           # (B, T) activity, full in time
                  *,
                  dt: float,
                  label: str,                    # x-axis label, e.g. 'Recurrent coupling  J'
                  banner: str = "",              # parameter string for the suptitles
                  fp: Mapping[str, object] | None = None,   # fixed_points.branch_scan output
                  means: Mapping[str, object] | None = None,  # {'sigma_tot': (B,), ...}
                  n_points: int = 1000,          # tail points per cell in the scatter
                  max_freq: float | None = 0.2,  # spectrogram y-limit (1/ms); None = Nyquist
                  ) -> dict[str, Figure]:
    """ Bifurcation, stability and entropy-production figures over one swept axis.

    `m_tot` is wanted whole rather than reduced: the scatter needs its tail and the
    spectrogram needs all of it. Everything else is a time average, so `means` takes the
    reduced quantities instead. `fp` may be None, in which case the panels that need the
    fixed points say so rather than vanishing -- a missing branch diagram should be visible.

    Returns the figures by name so the caller can adjust or save them. """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    axis, m_tot = _np(axis), _np(m_tot)
    m_avg = m_tot.mean(axis=1)

    stable = (_np(fp["rho"]) < 1.0) if fp is not None else None
    # Every point along the axis where stability changes hands, marked on each panel: the
    # scatter should fan out at one. Found as sign changes rather than as max(x where
    # stable), because which side is the unstable one is a property of the axis -- J and E
    # destabilize as they grow, theta and beta restabilize -- and a one-sided rule silently
    # marks the far end of the axis for half of them.
    crits = []
    if stable is not None and stable.any() and not stable.all():
        order = np.argsort(_np(fp["x"]))
        xs, ss = _np(fp["x"])[order], stable[order]
        crits = [0.5 * (xs[i] + xs[i + 1]) for i in np.nonzero(ss[1:] != ss[:-1])[0]]

    def new_figure(title: str, n_rows: int, height: float):
        """ One figure per topic, all sharing the swept axis and the parameter banner. """
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, height), sharex=True)
        axes = [axes] if n_rows == 1 else list(axes)
        fig.suptitle(f'{title}    —    {banner}' if banner else title, fontsize=13)
        for ax in axes:
            ax.grid()
            for c in crits:
                ax.axvline(c, color='0.4', linewidth=1, linestyle=':')
        axes[-1].set_xlabel(label)
        return fig, axes

    figs = {}

    # ── Bifurcation ───────────────────────────────────────────────────────────
    # The simulated attractor and the fixed points it is built from, on a common axis: in
    # the stable region the scatter collapses onto the branch, past the threshold the branch
    # stays at the centre of the oscillation the scatter spreads over.

    figs["bifurcation"], (ax_scatter, ax_branch) = new_figure('Bifurcation', 2, 8)

    ax_scatter.set_title(f'Simulated activity (last {n_points} points per cell)')
    tail = m_tot[:, -n_points:]
    ax_scatter.scatter(np.repeat(axis, tail.shape[1]), tail.reshape(-1)/dt,
                       s=1, alpha=0.2, color='k', linewidths=0)
    ax_scatter.plot(axis, m_avg/dt, linewidth=1, color='C1', label='time average')
    ax_scatter.set_ylabel('m/dt  (1/ms)')
    ax_scatter.legend()

    ax_branch.set_title('Mean-field fixed points  (filled = stable, open = unstable)')
    if fp is not None:
        ax_branch.plot(axis, m_avg/dt, color='C1', linewidth=1, label='simulated time average')
        for mask, style, lab in [
                (stable,  dict(marker='o', color='C0'), 'stable'),
                (~stable, dict(marker='o', facecolors='none', edgecolors='C3'), 'unstable')]:
            if mask.any():
                ax_branch.scatter(_np(fp["x"])[mask], _np(fp["m_star"])[mask]/dt, s=22,
                                  label=lab, **style)
        ax_branch.legend()
    else:
        ax_branch.set_title('No fixed points: run the fixed-point stage')
    ax_branch.set_ylabel('m*/dt  (1/ms)')

    # ── Stability ─────────────────────────────────────────────────────────────
    # The spectrum of the one-step Jacobian at each fixed point, against the oscillation the
    # simulation actually shows: |λ| says whether the fixed point is an attractor, arg(λ)
    # says at what frequency it fails to be one, and the spectrogram is the check on both.

    figs["stability"], (ax_mod, ax_spec) = new_figure('Stability', 2, 8)

    ax_mod.set_title('Leading eigenvalues of the one-step Jacobian at each fixed point')
    if fp is not None:
        lead = _np(fp["lead"])
        # the subleading moduli give the margin: a gap means the leading pair really is the
        # mode that goes unstable, no gap means the crossing is not a clean Hopf
        for k in range(1, lead.shape[1]):
            ax_mod.scatter(_np(fp["x"]), abs(lead[:, k]), s=4, color='0.75',
                           label='subleading' if k == 1 else None)
        ax_mod.scatter(_np(fp["x"]), abs(lead[:, 0]), s=22,
                       c=['C0' if s else 'C3' for s in stable], label='leading |λ|', zorder=3)
        ax_mod.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label='unit circle')
        ax_mod.legend()
    else:
        ax_mod.set_title('No eigenvalues: run the fixed-point stage')
    ax_mod.set_ylabel('|λ|')

    ax_spec.set_title('Activity spectrogram')
    ax_spec.grid(False)                     # the mesh is the data; a grid over it is noise
    sig   = m_tot - m_tot.mean(axis=1, keepdims=True)   # (B, T), DC removed per cell
    power = abs(np.fft.rfft(sig, axis=1)) ** 2          # (B, F)
    freqs = np.fft.rfftfreq(sig.shape[1], d=dt)         # (F,) cycles/ms
    db    = 10 * np.log10(power.T + 1e-12)              # (F, B)
    pcm = ax_spec.pcolormesh(axis, freqs, db, shading='auto', cmap='magma')
    # the colorbar gets its own slot carved out of ax_spec, and ax_mod gets an identical slot
    # left blank -- attaching it to ax_spec alone would narrow only that panel and slide the
    # two x axes out of register despite the sharex
    cax = make_axes_locatable(ax_spec).append_axes("right", size="2%", pad=0.15)
    figs["stability"].colorbar(pcm, cax=cax, label='Power (dB)')
    make_axes_locatable(ax_mod).append_axes("right", size="2%", pad=0.15).set_axis_off()
    if max_freq is not None:
        ax_spec.set_ylim(0, max_freq)
    if fp is not None:
        # arg(λ) of the leading mode is the oscillation the linearization predicts; it should
        # sit on the spectrogram's ridge wherever that mode is the one driving the dynamics
        osc = abs(np.angle(_np(fp["lead"])[:, 0])) / (2 * np.pi) / dt
        ax_spec.plot(_np(fp["x"]), osc, color='c', linewidth=1.5, linestyle='--',
                     label='arg(λ) of the leading mode')
        ax_spec.legend(loc='upper left')
    ax_spec.set_ylabel('Frequency (1/ms)')

    # ── Entropy production ────────────────────────────────────────────────────

    if means is not None:
        figs["entropy"], (ax_sigma, ax_H) = new_figure('Entropy production', 2, 8)

        ax_sigma.set_title('Entropy production rate  Σ')
        ax_sigma.plot(axis, _np(means["sigma_tot"])/dt, linewidth=2)
        ax_sigma.set_ylabel('Σ  (nats/ms)')

        ax_H.set_title('Conditional entropies')
        ax_H.plot(axis, _np(means["H_fwd_tot"])/dt, linewidth=2, label='Forward')
        ax_H.plot(axis, _np(means["H_rev_tot"])/dt, linewidth=2, label='Backward')
        ax_H.set_ylabel('H  (nats/ms)')
        ax_H.legend()

    for fig in figs.values():
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return figs
