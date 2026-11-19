""" Convergence order of the mean field's sigma in dt, on the E1 oscillating points: sigma,
r_mean and the period at dt = 0.2, 0.1, 0.05, the observed order and Richardson values; the
per-halving increments of H_fwd and H_rev against r_mean ln 2; and the phase profile of the
per-step sigma, with the fraction of the sigma change (0.1 -> 0.05) inside the burst. """

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.analysis as an
import rdme.helpers as hlp
from rdme.mean_field import RDMIsingModel

# ── Parameters ────────────────────────────────────────────────────────────────

# the E1 points of ising_mf_vs_sm.py (single excitatory population on the Schwalger 2017
# Fig. 3 neuron; J_H = 276.64 at E = 15 mV, 139.34 at E = 30 mV), plus E = 30 at 1.25 J_H
POINTS = {
    "E1-15-onset":  dict(J=304.31, E=15.0, beta=0.5, theta=15.0,     # 1.1 J_H
                         tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                         nu0=0.01, t_abs=4.0),
    "E1-30-onset":  dict(J=153.27, E=30.0, beta=0.5, theta=15.0,     # 1.1 J_H
                         tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                         nu0=0.01, t_abs=4.0),
    "E1-15-strong": dict(J=553.29, E=15.0, beta=0.5, theta=15.0,     # 2 J_H
                         tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                         nu0=0.01, t_abs=4.0),
    "E1-30-strong": dict(J=278.68, E=30.0, beta=0.5, theta=15.0,     # 2 J_H
                         tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                         nu0=0.01, t_abs=4.0),
    "E30-1.25JH":   dict(J=174.17, E=30.0, beta=0.5, theta=15.0,     # 1.25 J_H
                         tau_int=20.0, tau_ref=1.0, K_ref=0.0,
                         nu0=0.01, t_abs=4.0),
}

hazard  = "poisson_exact"
IC      = "silent"
AGE_MAX = 25.0              # ms, only read by the "random" start

# numerical parameters
DTS    = (0.2, 0.1, 0.05)   # ms, halving; each must divide t_abs
TOL    = 1e-2
DEVICE = 'cpu'
T_EQUI, T_MEAS = 5000.0, 3000.0   # ms, as in ising_mf_vs_sm.py
N_BINS = 400                # phase bins of the profiles
P_RANGE = (0.5, 3.0)        # an observed order outside this is "out of range"

run_sim   = True
run_plot  = True
overwrite = False   # the hash covers every parameter, so this is only for a change to the CODE


# ── Paths, dicts & switches ──────────────────────────────────────────────────

ROOT = Path(__file__).parents[1]
FIG  = Path(__file__).parent / "figures" / "E1-sigma_order"
KEYS = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")


def params(name: str, dt: float) -> dict:
    return dict(POINTS[name], dt=dt, hazard=hazard)


def cache(name: str, dt: float) -> Path:
    return hlp.cache_path(Path(__file__).stem,
                          dict(params(name, dt), t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                               ic=IC, age_max=AGE_MAX),
                          tag=f"{name}-dt{dt:g}", root=ROOT / "cache")


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:
    for name in POINTS:
        for dt in DTS:
            path = cache(name, dt)
            if path.exists() and not overwrite:
                continue
            print(f"Running {name} at dt={dt:g}.")
            mf = RDMIsingModel(**params(name, dt), ic=IC, age_max=AGE_MAX, tol=TOL,
                               device=DEVICE)
            mf.trajectory(T=hlp.n_steps(T_EQUI, dt, name="T_EQUI"))
            tr = mf.entropy_trajectory(T=hlp.n_steps(T_MEAS, dt, name="T_MEAS"))
            hlp.save_atomic({"params": params(name, dt), "Q": mf.Q, "t_equi": T_EQUI,
                             "t_meas": T_MEAS, "mf_traj2": {k: tr[k].cpu() for k in KEYS}},
                            path)
            print(f"Saved to {path}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    FIG.mkdir(parents=True, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    for name in POINTS:
        R = {}
        for dt in DTS:
            d  = torch.load(cache(name, dt), weights_only=False)
            tr = d["mf_traj2"]
            r  = tr["m_tot"].double() / dt * 1e3       # Hz
            s  = tr["sigma_tot"].double() / dt * 1e3   # nats per neuron per s, and so the H
            hf = tr["H_fwd_tot"].double() / dt * 1e3
            hr = tr["H_rev_tot"].double() / dt * 1e3
            a, b, n = an.cycle_window(r)
            R[dt] = dict(r=r, s=s, rbar=r[a:b].mean().item(), sig=s[a:b].mean().item(),
                         Hf=hf[a:b].mean().item(), Hr=hr[a:b].mean().item(),
                         per=(b - a) * dt / n, n=n, ratio=an.cycle_amplitude_ratio(r),
                         chk=(hr[a:b] - hf[a:b] - s[a:b]).abs().max().item())
        sig = [R[dt]["sig"] for dt in DTS]
        rb  = [R[dt]["rbar"] for dt in DTS]
        pr  = [R[dt]["per"] for dt in DTS]

        out(f"=== {name}   cycles {[R[dt]['n'] for dt in DTS]}  "
            f"amp ratio {[round(R[dt]['ratio'], 4) for dt in DTS]}  "
            f"max|H_rev - H_fwd - sigma| {max(R[dt]['chk'] for dt in DTS):.2e} nats/s")
        out(f"  sigma (nats/s)  dt=0.2 {sig[0]:.6g}  0.1 {sig[1]:.6g}  0.05 {sig[2]:.6g}")
        out(f"  r_mean (Hz)     dt=0.2 {rb[0]:.6g}  0.1 {rb[1]:.6g}  0.05 {rb[2]:.6g}")
        out(f"  period (ms)     dt=0.2 {pr[0]:.6g}  0.1 {pr[1]:.6g}  0.05 {pr[2]:.6g}")
        for lab, v in (("sigma", sig), ("r_mean", rb), ("period", pr)):
            p, why = an.observed_order(v, P_RANGE)
            out(f"  p[{lab}] = {p:.3f}  {why}  (diffs {v[0] - v[1]:+.4g}, {v[1] - v[2]:+.4g})")
        p, why = an.observed_order(sig, P_RANGE)
        if why:
            out(f"  Richardson: not asymptotic ({why})")
        else:
            ext = {q: an.richardson(sig, p=q, p_range=P_RANGE)[0] for q in (p, 1, 2)}
            out(f"  Richardson (0.1, 0.05): observed p {ext[p]:.6g}   "
                f"p=1 {ext[1]:.6g}   p=2 {ext[2]:.6g}")

        # the divergent part of the path entropies: r_mean ln 2 per halving, r_mean at the finer dt
        for c, f in ((0.2, 0.1), (0.1, 0.05)):
            rl = R[f]["rbar"] * math.log(2)
            dHf, dHr = R[f]["Hf"] - R[c]["Hf"], R[f]["Hr"] - R[c]["Hr"]
            out(f"  H {c:g}->{f:g}: dH_fwd {dHf:+.5g}  dH_rev {dHr:+.5g}  r_mean ln2 {rl:.5g} "
                f"(r_mean at {f:g})  remainders fwd {dHf - rl:+.5g}  rev {dHr - rl:+.5g}   "
                f"(H_fwd {R[c]['Hf']:.5g} -> {R[f]['Hf']:.5g})")

        # where in the cycle the sigma change sits; per-step sigma is an entropy flow, so the
        # profiles are diagnostic only
        prof = {dt: (an.phase_profile(R[dt]["s"], R[dt]["r"], N_BINS),
                     an.phase_profile(R[dt]["r"], R[dt]["r"], N_BINS)) for dt in DTS}
        ds = prof[0.05][0] - prof[0.1][0]
        burst   = prof[0.05][1] > R[0.05]["rbar"]
        burst01 = prof[0.1][1] > R[0.1]["rbar"]
        out(f"  phase-binned sigma change 0.1->0.05 {float(ds.mean()):+.5g} nats/s "
            f"(period-avg diff {sig[2] - sig[1]:+.5g}); in burst (r_0.05 > mean): "
            f"{float(ds[burst].sum() / ds.sum()):.4f}  [burst from r_0.1: "
            f"{float(ds[burst01].sum() / ds.sum()):.4f}]  burst phase width "
            f"{float(burst.double().mean()):.3f}")

        ph = (torch.arange(N_BINS) + 0.5) / N_BINS
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
        for dt in DTS:
            ax1.plot(ph, prof[dt][0], label=f"dt = {dt:g} ms")
            ax2.plot(ph, prof[dt][1], label=f"dt = {dt:g} ms")
        ax2.axhline(R[0.05]["rbar"], color="k", lw=0.7, ls="--", label="mean r (dt = 0.05)")
        ax1.set_ylabel("σ per unit time (nats / neuron / s)")
        ax2.set_ylabel("r (Hz)")
        ax2.set_xlabel("phase (0 = upward crossing of mean r)")
        ax1.set_title(f"{name}: per-step σ against phase, mean field ({N_BINS} bins; "
                      "an entropy flow, diagnostic only)", fontsize=9)
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=120)
        plt.close(fig)

    (FIG / "tables.txt").write_text(
        f"e1_sigma_order.py; mean field only; {hazard}; {IC} start; T_equi {T_EQUI:g} ms, "
        f"T_meas {T_MEAS:g} ms; averages over whole cycles (analysis.cycle_window on r)\n\n"
        + "\n".join(lines) + "\n")
