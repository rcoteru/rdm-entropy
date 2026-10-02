"""
The three hazards side by side: dynamics, rates, and entropy production.

App. "Time Discretization and the Continuous-Time Limit" gives one synchronous model and two
ways to turn it into a continuous-time one:

    synchronous   Phi = sigma(beta*h)                   no dt -> 0 limit
    async         Phi = lam0*dt * sigma(beta*h)         -> rate lam0*sigma(beta*h)
    escape        Phi = 1 - (1+e^{beta*h})^(-lam0*dt)   -> rate lam0*softplus(beta*h)

All three coincide at lam0*dt = 1. The two families differ in the discretization map -- linear
thinning against exact survival -- which only changes the speed of convergence, and in the rate
function, which changes the limit. So they converge to *different* continuous-time models, and
the size of that difference is set by the time quantum relative to the kernels, not by the step.

This script measures three things, each its own block:

  1. Convergence in dt. With lam0 pinned, both continuum hazards must settle as dt refines,
     while the synchronous model walks across a family and does not.

  2. Separation against dt0/tau. The gap between the two continuum models at strong drive,
     swept over the time quantum. This is the figure for the rate-function claim.

  3. Entropy production. sigma/dt has a continuum limit; H_fwd and H_rev do not -- they diverge
     as -r ln dt, being entropies of a binned process against a counting measure. Both are
     plotted against dt so the divergence is visible rather than asserted.

Two things to be careful about, both of which cost a wrong answer to find out:

  sigma vanishes identically at a fixed point. With a constant input the reverse field equals
  the forward one as a function of age, so there is nothing to measure. Block 3 therefore runs
  at an oscillating operating point, not a settled one.

  Averaging an oscillating sigma over a window that is not a whole number of periods leaves a
  sampling error of order amplitude/n_periods, which at short windows swamps the discretization
  error being studied. The windows below are set in periods, not in steps, for that reason.
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

HAZARDS = ("synchronous", "async", "escape")
COLORS  = {"synchronous": "#6b6b6b", "async": "#2e6fb7", "escape": "#b4453c"}


# ── Operating points ──────────────────────────────────────────────────────────

# Weakly driven and settled: the regime where the two rate functions agree, used for the
# convergence study where a clean mean rate matters more than a large separation.
SETTLED = dict(E_ratio=0.8, beta_E=30.0, beta_I=30.0, theta_E=1.0, theta_I=1.0,
               tau_int_E=6.0, tau_int_I=8.0, tau_ref_E=3.0, tau_ref_I=3.0,
               K_ref_E=0.5, K_ref_I=0.0,
               w_EE=0.5, w_EI=1.0, w_IE=-1.0, w_II=-1.0, E_exc=0.9, E_inh=0.9)

# Strongly driven and settled: the regime where the asynchronous rate's saturation at lam0 is
# binding, so the two continuum models separate.
DRIVEN = dict(SETTLED, w_EE=0.0, E_exc=2.0, E_inh=2.0)

# Oscillating, so that sigma is not identically zero.
OSCILLATING = dict(E_ratio=0.8, beta_E=30.0, beta_I=30.0, theta_E=1.0, theta_I=1.0,
                   tau_int_E=10.0, tau_int_I=12.0, tau_ref_E=3.0, tau_ref_I=3.0,
                   K_ref_E=0.5, K_ref_I=0.0,
                   w_EE=2.0, w_EI=2.0, w_IE=-1.0, w_II=-2.0, E_exc=1.0, E_inh=1.0)


def build(params, hazard, dt, dt0):
    return single.RDMWilsonCowan(**params, dt=dt, dt0=dt0, hazard=hazard)


def mean_rate(params, hazard, dt, dt0, eq_ms=2000.0, rec_ms=400.0):
    net = build(params, hazard, dt, dt0)
    net.forward(int(eq_ms / dt))
    return float(net.trajectory(int(rec_ms / dt), pb=False)["r"][:, 0].mean())


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    if CACHE_FILE.exists() and not overwrite:
        print(f"Results already exist at {CACHE_FILE}. Skipping.")
    else:
        res = {}
        t0 = time.time()

        # 1. convergence in dt at a pinned dt0 -----------------------------------
        dt0 = 0.2
        # lam0*dt <= 1 caps the asynchronous hazard at dt <= dt0, so the sweep starts there
        dts = [0.2, 0.1, 0.05, 0.025]
        print(f"[1/3] convergence in dt at dt0 = {dt0} ...")
        res["conv_dts"] = torch.tensor(dts)
        res["conv"] = {h: torch.tensor([mean_rate(SETTLED, h, dt, dt0) for dt in dts])
                       for h in HAZARDS}

        # 2. separation against the time quantum ---------------------------------
        # dt is tied to dt0 so that lam0*dt is fixed at 1/8 throughout: the comparison is
        # between models at matched numerical resolution, not between resolutions.
        dt0s = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        print(f"[2/3] separation over dt0 (tau_ref = {DRIVEN['tau_ref_E']}) ...")
        res["sep_dt0s"] = torch.tensor(dt0s)
        res["sep"] = {h: torch.tensor([mean_rate(DRIVEN, h, q / 8.0, q, eq_ms=1500.0,
                                                 rec_ms=300.0) for q in dt0s])
                      for h in ("async", "escape")}

        # 3. entropy production vs dt at an oscillating point ---------------------
        # The window is set in periods rather than steps: averaging sigma over a window that is
        # not a whole number of periods leaves a sampling error that swamps the effect.
        print("[3/3] entropy production vs dt ...")
        epr_dts = [0.2, 0.1, 0.05]
        probe = build(OSCILLATING, "synchronous", 0.2, 0.2)
        probe.forward(int(3000 / 0.2))
        x = probe.trajectory(int(600 / 0.2), pb=False)["r"][:, 0]
        x = x - x.mean()
        ac = torch.stack([(x[:-k] * x[k:]).sum() for k in range(1, len(x) // 2)])
        neg = (ac < 0).nonzero()
        period_ms = (int(ac[int(neg[0]):].argmax()) + int(neg[0]) + 1) * 0.2
        print(f"      burst period ~ {period_ms:.1f} ms; averaging over 60 periods")

        res["epr_dts"] = torch.tensor(epr_dts)
        res["period_ms"] = torch.tensor(period_ms)
        epr = {}
        for h in HAZARDS:
            rows = []
            for dt in epr_dts:
                if h == "synchronous" and dt != 0.2:
                    rows.append(torch.full((4,), float('nan')))   # no limit; only dt0 is defined
                    continue
                net = build(OSCILLATING, h, dt, 0.2)
                net.forward(int(3000 / dt))
                t = net.entropy_trajectory(int(60 * period_ms / dt), method="sliding", pb=False)
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

    # ── 1. convergence ────────────────────────────────────────────────────────
    print("\n[1] mean rate vs dt (dt0 = 0.2). A convergent model's successive")
    print("    differences shrink; the synchronous model's do not.")
    dts = res["conv_dts"]
    print(f"    {'dt':>7} " + "".join(f"{h:>14}" for h in HAZARDS))
    for j, dt in enumerate(dts):
        print(f"    {float(dt):7.3f} " + "".join(f"{float(res['conv'][h][j]):14.6f}"
                                                 for h in HAZARDS))
    for h in HAZARDS:
        d = res["conv"][h].diff().abs()
        print(f"    {h:>12}: successive |differences| = "
              f"{['%.2e' % v for v in d.tolist()]}")

    # ── 2. separation ─────────────────────────────────────────────────────────
    print("\n[2] the two continuum models vs the time quantum (tau_ref = 3), at strong drive.")
    a, e = res["sep"]["async"], res["sep"]["escape"]
    gap = (a - e).abs() / a
    print(f"    {'dt0':>7} {'dt0/tau':>8} {'async':>10} {'escape':>10} {'rel gap':>9}")
    for j, q in enumerate(res["sep_dt0s"]):
        print(f"    {float(q):7.2f} {float(q)/3.0:8.2f} {float(a[j]):10.5f} "
              f"{float(e[j]):10.5f} {float(gap[j]):9.2%}")

    # ── 3. entropy production ─────────────────────────────────────────────────
    print(f"\n[3] entropy production vs dt at an oscillating point "
          f"(period {float(res['period_ms']):.1f} ms).")
    print("    sigma/dt should converge; H/dt should diverge as -r ln dt.")
    print(f"    {'hazard':>12} {'dt':>6} {'r /ms':>9} {'sigma/dt':>11} "
          f"{'H_fwd/dt':>10} {'H_rev/dt':>10}")
    for h in HAZARDS:
        for j, dt in enumerate(res["epr_dts"]):
            r, sg, hf, hr = res["epr"][h][j].tolist()
            if r != r: continue        # nan: the synchronous model away from its quantum
            print(f"    {h:>12} {float(dt):6.3f} {r:9.5f} {sg:11.6f} {hf:10.4f} {hr:10.4f}")
        col = res["epr"][h][:, 2]
        ok = ~torch.isnan(col)
        if int(ok.sum()) > 1:
            # -r ln dt predicts an increase of r*ln2 per halving
            pred = float(res["epr"][h][0, 0]) * torch.log(torch.tensor(2.0))
            print(f"    {h:>12}: H_fwd/dt increase per halving "
                  f"{['%.4f' % v for v in col[ok].diff().tolist()]}  "
                  f"(-r ln dt predicts {float(pred):.4f})")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # convergence: the quantity is a rate, the axis is a resolution -> log x
    for h in HAZARDS:
        ax1.plot(dts, res["conv"][h], 'o-', color=COLORS[h], label=h, linewidth=2, markersize=7)
    ax1.set_xscale('log'); ax1.invert_xaxis()
    ax1.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
    ax1.set_ylabel(r'mean $r_E$ [1/ms]')
    ax1.set_title(r'Convergence at fixed $\lambda_0$')
    ax1.legend(); ax1.grid(alpha=0.3)

    # separation: a ratio against a ratio -> both axes meaningful, gap on a second panel
    ax2.plot(res["sep_dt0s"] / 3.0, a, 'o-', color=COLORS["async"], label='async',
             linewidth=2, markersize=7)
    ax2.plot(res["sep_dt0s"] / 3.0, e, 'o-', color=COLORS["escape"], label='escape',
             linewidth=2, markersize=7)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\Delta t_0 / \tau_{\mathrm{ref}}$')
    ax2.set_ylabel(r'mean $r_E$ [1/ms]')
    ax2.set_title('The two continuum models separate\nas the quantum slows')
    ax2.legend(); ax2.grid(alpha=0.3)

    # entropy: sigma converges, H diverges -- two different stories, so two y-scales would be
    # a dual axis. Plot H here and note sigma in the printout instead.
    for h in HAZARDS:
        col = res["epr"][h][:, 2]
        ok = ~torch.isnan(col)
        ax3.plot(res["epr_dts"][ok], col[ok], 'o-', color=COLORS[h], label=h,
                 linewidth=2, markersize=7)
    ax3.set_xscale('log'); ax3.invert_xaxis()
    ax3.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
    ax3.set_ylabel(r'$H_{\mathrm{fwd}}/\Delta_t$ [nats/ms]')
    ax3.set_title(r'The entropies diverge as $-r\ln\Delta_t$')
    ax3.legend(); ax3.grid(alpha=0.3)

    fig.tight_layout()

    # sigma on its own: it is the quantity with a limit, and it is small, so it would be
    # invisible beside the diverging entropies
    fig2, ax = plt.subplots(figsize=(6.5, 4.6))
    for h in HAZARDS:
        col = res["epr"][h][:, 1]
        ok = ~torch.isnan(col)
        ax.plot(res["epr_dts"][ok], col[ok], 'o-', color=COLORS[h], label=h,
                linewidth=2, markersize=7)
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel(r'$\Delta_t$ [ms]  (refining $\rightarrow$)')
    ax.set_ylabel(r'$\sigma/\Delta_t$ [nats/ms]')
    ax.set_title('Entropy production rate converges')
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()

    plt.show()
