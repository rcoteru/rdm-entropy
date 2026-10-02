"""
The three hazards side by side, in the Ising (single-population) model.

App. "Time Discretization and the Continuous-Time Limit" gives one synchronous model and two
ways to turn it into a continuous-time one:

    synchronous   Phi = sigma(beta*h)                   no dt -> 0 limit
    async         Phi = lam0*dt * sigma(beta*h)         -> rate lam0*sigma(beta*h)
    escape        Phi = 1 - (1+e^{beta*h})^(-lam0*dt)   -> rate lam0*softplus(beta*h)

All three coincide at lam0*dt = 1. The two families differ in the discretization map -- linear
thinning against exact survival -- which only changes the speed of convergence, and in the rate
function, which changes the limit. So they converge to *different* continuous-time models, and
the size of that difference is set by the time quantum relative to the kernels, not by the step.

Three blocks, each its own operating point:

  1. Convergence in dt, at J = 0. With lam0 pinned, both continuum hazards must settle as dt
     refines while the synchronous model walks across a family and does not.

  2. Separation against dt0/tau, at J = 0 and strong drive. The gap between the two continuum
     models, swept over the time quantum. This is the figure for the rate-function claim.

  3. Entropy production, at J > 0 and oscillating, with all three kernels run across dt. The
     two continuum ones converge in both the rate and sigma/dt -- to different values of
     sigma, since their rate functions differ -- while the synchronous one walks off in both,
     being a different model at every step. H_fwd and H_rev diverge for all three as
     -r ln dt, being entropies of a binned process against a counting measure, so only
     sigma/dt is comparable across resolutions at all.

Why J = 0 for the first two blocks. With this model's time constants the self-excitatory
population oscillates for any J >~ 5, so there is no interacting *settled* point to measure a
mean rate at. That is not a limitation here: at J = 0 the neurons are independent under a
common drive, so the rate is fixed by the hazard and the drive alone, and the measurement
isolates the discretization from the network's self-consistency. Block 3 then supplies the
interacting case, where it is needed -- sigma vanishes identically at any fixed point, since a
constant input makes the reverse field equal the forward one as a function of age, so entropy
production can only be measured on a moving state.

Averaging an oscillating sigma over a window that is not a whole number of periods leaves a
sampling error of order amplitude/n_periods, which at short windows swamps the discretization
error being studied. Block 3 measures the burst period first and sets its window in periods.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR  = Path(__file__).parents[1] / "cache"
CACHE_FILE = CACHE_DIR / f"{bname}.pt"

run_sim   = True
run_plot  = True
overwrite = False

torch.set_default_dtype(torch.float64)

HAZARDS = ("async", "escape", "synchronous")
COLORS  = {"synchronous": "#6b6b6b", "async": "#2e6fb7", "escape": "#b4453c"}

# Kernels, shared by every block and following the repo's Ising convention.
KERNELS = dict(beta=30.0, theta=1.0, tau_int=20.0, tau_ref=3.0)

# Independent neurons, drive just above threshold: settled, moderate rate.
WEAK   = dict(KERNELS, J=0.0, E=1.1, K_ref=0.0)
# Independent neurons, drive well above threshold: the regime where the asynchronous rate's
# saturation at lam0 is binding, so the two continuum models separate.
DRIVEN = dict(KERNELS, J=0.0, E=2.0, K_ref=0.0)
# Interacting and oscillating, so that sigma is not identically zero.
OSC    = dict(KERNELS, J=10.0, E=0.95, K_ref=0.0)


def build(params, hazard, dt, dt0):
    return single.RDMIsingModel(**params, dt=dt, dt0=dt0, hazard=hazard)


def stats(params, hazard, dt, dt0, eq_ms=3000.0, rec_ms=600.0):
    """ (mean rate, peak-to-peak) in 1/ms, after equilibration. """
    net = build(params, hazard, dt, dt0)
    net.forward(int(eq_ms / dt))
    r = net.trajectory(int(rec_ms / dt), pb=False)["r"][:, 0]
    return float(r.mean()), float(r.max() - r.min())


def burst_period(params, dt=0.1, dt0=0.2, eq_ms=4000.0, rec_ms=1500.0) -> float:
    """ Dominant period in ms, from the first autocorrelation peak after the first zero
    crossing. argmax of the autocorrelation itself is lag 1 for any smooth signal and says
    nothing; the peak has to be looked for past the crossing. """
    net = build(params, "escape", dt, dt0)
    net.forward(int(eq_ms / dt))
    x = net.trajectory(int(rec_ms / dt), pb=False)["r"][:, 0]
    x = x - x.mean()
    ac = torch.stack([(x[:-k] * x[k:]).sum() for k in range(1, len(x) // 2)])
    neg = (ac < 0).nonzero()
    if len(neg) == 0:
        return float('nan')
    start = int(neg[0])
    return (int(ac[start:].argmax()) + start + 1) * dt


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    if CACHE_FILE.exists() and not overwrite:
        print(f"Results already exist at {CACHE_FILE}. Skipping.")
    else:
        res = {}
        t0 = time.time()

        # 1. convergence in dt at a pinned dt0 -----------------------------------
        # lam0*dt <= 1 caps the asynchronous hazard at dt <= dt0, so the sweep starts there.
        dt0 = 0.2
        dts = [0.2, 0.1, 0.05, 0.025]
        print(f"[1/3] convergence in dt at dt0 = {dt0}, J = {WEAK['J']} ...")
        res["conv_dts"] = torch.tensor(dts)
        res["conv"] = {h: torch.tensor([stats(WEAK, h, dt, dt0)[0] for dt in dts])
                       for h in HAZARDS}
        res["conv_amp"] = torch.tensor(stats(WEAK, "escape", 0.05, dt0)[1])

        # 2. separation against the time quantum ---------------------------------
        # dt is tied to dt0 so lam0*dt is fixed at 1/8 throughout: this compares models at
        # matched numerical resolution, not resolutions against each other.
        dt0s = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
        print(f"[2/3] separation over dt0 (tau_ref = {KERNELS['tau_ref']}, "
              f"tau_int = {KERNELS['tau_int']}) ...")
        res["sep_dt0s"] = torch.tensor(dt0s)
        res["sep"] = {h: torch.tensor([stats(DRIVEN, h, q / 8.0, q, eq_ms=2500.0,
                                             rec_ms=400.0)[0] for q in dt0s])
                      for h in ("async", "escape")}

        # 3. entropy production vs dt at an oscillating point ---------------------
        print("[3/3] entropy production vs dt ...")
        period = burst_period(OSC)
        n_per = 30
        print(f"      burst period ~ {period:.1f} ms; averaging over {n_per} periods")
        # down to 0.025: at 0.05 the asynchronous sigma was still moving 22% per halving, so
        # its limit could not be read off. The sliding EPR is what makes this step affordable
        # (Qm = 3685 there, ~25 s a point instead of minutes).
        epr_dts = [0.2, 0.1, 0.05, 0.025]
        res["epr_dts"] = torch.tensor(epr_dts)
        res["period_ms"] = torch.tensor(period)

        epr = {}
        for h in HAZARDS:
            rows = []
            for dt in epr_dts:
                # The synchronous model is run across dt too, even though refining it moves to
                # a different member of the family rather than to a better solution of one
                # model. That IS the demonstration: its rate and its entropy production both
                # walk off, while the two continuum kernels settle. Masking it out would hide
                # the contrast the figure exists to show.
                net = build(OSC, h, dt, dt0)
                net.forward(int(4000 / dt))
                t = net.entropy_trajectory(int(n_per * period / dt), method="sliding", pb=False)
                rows.append(torch.tensor([
                    float(t["r_tot"].mean()),
                    float(t["sigma_tot"].mean()) / dt,
                    float(t["H_fwd_tot"].mean()) / dt,
                    float(t["H_rev_tot"].mean()) / dt]))
            epr[h] = torch.stack(rows)
        res["epr"] = epr      # columns: r, sigma/dt, H_fwd/dt, H_rev/dt

        print(f"Done in {time.time() - t0:.1f}s")
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(res, CACHE_FILE)
        print(f"Results saved to {CACHE_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"{CACHE_FILE} not found. Run the simulation first.")
    res = torch.load(CACHE_FILE, weights_only=False)

    print(f"\n[1] mean rate vs dt (dt0 = 0.2, J = {WEAK['J']}, E = {WEAK['E']}). "
          f"Settled: peak-to-peak {float(res['conv_amp']):.1e} /ms.")
    print("    A convergent model's successive differences shrink; the synchronous model's")
    print("    do not, because refining dt moves it to a different model.")
    dts = res["conv_dts"]
    print(f"    {'dt':>7} " + "".join(f"{h:>14}" for h in HAZARDS))
    for j, dt in enumerate(dts):
        print(f"    {float(dt):7.3f} " + "".join(f"{float(res['conv'][h][j]):14.6f}"
                                                 for h in HAZARDS))
    for h in HAZARDS:
        d = res["conv"][h].diff().abs()
        print(f"    {h:>12}: successive |differences| = {['%.2e' % v for v in d.tolist()]}")

    print(f"\n[2] the two continuum models vs the time quantum, at strong drive "
          f"(J = {DRIVEN['J']}, E = {DRIVEN['E']}).")
    a, e = res["sep"]["async"], res["sep"]["escape"]
    gap = (a - e).abs() / a
    print(f"    {'dt0':>7} {'dt0/tau_ref':>12} {'async':>10} {'escape':>10} {'rel gap':>9}")
    for j, q in enumerate(res["sep_dt0s"]):
        print(f"    {float(q):7.2f} {float(q)/KERNELS['tau_ref']:12.2f} {float(a[j]):10.5f} "
              f"{float(e[j]):10.5f} {float(gap[j]):9.2%}")

    print(f"\n[3] entropy production vs dt at an oscillating point "
          f"(J = {OSC['J']}, period {float(res['period_ms']):.1f} ms).")
    print("    sigma/dt should converge; H/dt should diverge as -r ln dt.")
    print(f"    {'hazard':>12} {'dt':>6} {'r /ms':>9} {'sigma/dt':>11} "
          f"{'H_fwd/dt':>10} {'H_rev/dt':>10}")
    for h in HAZARDS:
        for j, dt in enumerate(res["epr_dts"]):
            r, sg, hf, hr = res["epr"][h][j].tolist()
            if r != r: continue
            print(f"    {h:>12} {float(dt):6.3f} {r:9.5f} {sg:11.6f} {hf:10.4f} {hr:10.4f}")
        col_r, col_s = res["epr"][h][:, 0], res["epr"][h][:, 1]
        ok = ~torch.isnan(col_r)
        if int(ok.sum()) > 2:
            dr, ds = col_r[ok].diff().abs(), col_s[ok].diff().abs()
            verdict = "CONVERGING" if (dr[-1] < 0.7 * dr[0] and ds[-1] < 0.7 * ds[0]) \
                      else "not converging"
            print(f"    {h:>12}: successive |dr| {['%.1e' % v for v in dr.tolist()]}, "
                  f"|dsigma| {['%.1e' % v for v in ds.tolist()]}  -> {verdict}")
        col = res["epr"][h][:, 2]
        ok = ~torch.isnan(col)
        if int(ok.sum()) > 1:
            pred = float(res["epr"][h][0, 0]) * float(torch.log(torch.tensor(2.0)))
            print(f"    {h:>12}: H_fwd/dt increase per halving "
                  f"{['%.4f' % v for v in col[ok].diff().tolist()]}  "
                  f"(-r ln dt predicts {pred:.4f})")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for h in HAZARDS:
        ax1.plot(dts, res["conv"][h], 'o-', color=COLORS[h], label=h, linewidth=2, markersize=7)
    ax1.set_xscale('log'); ax1.invert_xaxis()
    ax1.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
    ax1.set_ylabel(r'mean $r$ [1/ms]')
    ax1.set_title(r'Convergence at fixed $\lambda_0$')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(res["sep_dt0s"] / KERNELS['tau_ref'], a, 'o-', color=COLORS["async"],
             label='async', linewidth=2, markersize=7)
    ax2.plot(res["sep_dt0s"] / KERNELS['tau_ref'], e, 'o-', color=COLORS["escape"],
             label='escape', linewidth=2, markersize=7)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\Delta t_0 / \tau_{\mathrm{ref}}$')
    ax2.set_ylabel(r'mean $r$ [1/ms]')
    ax2.set_title('The two continuum models separate\nas the quantum slows')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()

    # Block 3, at the oscillating point. Rate, entropy production and forward entropy get one
    # panel each rather than sharing an axis: they live on different scales and tell different
    # stories, and a twin axis would invite reading a crossing that is not there.
    #
    # All three hazards appear in all three panels, which is the point. The synchronous model
    # walks off in BOTH the rate and sigma as dt refines -- it is a different model at every
    # step -- while the two continuum kernels settle, and settle on different values of sigma.
    fig2, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    panels = [(0, r'mean $r$ [1/ms]', 'Rate'),
              (1, r'$\sigma/\Delta_t$ [nats/ms]', 'Entropy production rate'),
              (2, r'$H_{\mathrm{fwd}}/\Delta_t$ [nats/ms]', r'Forward entropy')]
    for ax, (col, ylab, title) in zip(axs, panels):
        for h in HAZARDS:
            y = res["epr"][h][:, col]
            ok = ~torch.isnan(y)
            ax.plot(res["epr_dts"][ok], y[ok], 'o-', color=COLORS[h], label=h,
                    linewidth=2, markersize=7)
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.grid(alpha=0.3)
    axs[0].legend()
    fig2.suptitle(f"Oscillating point (J = {OSC['J']:g}): the two continuum kernels converge, "
                  f"the synchronous one does not.  $H$ diverges for all three as "
                  r"$-r\ln\Delta_t$.")
    fig2.tight_layout()

    plt.show()
