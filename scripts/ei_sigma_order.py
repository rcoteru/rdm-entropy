""" Convergence order of the mean field's sigma in dt, on the E-I oscillating presets: per
population (E, I) and for the aggregate, sigma, r_mean, the period and r_max at dt = 0.2, 0.1,
0.05 and the observed orders; the per-halving increments of H_fwd and H_rev against
r_mean ln 2; and the phase profile of the per-step sigma. The E-I counterpart of
e1_sigma_order.py. """

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.analysis as an
import rdme.helpers as hlp
from rdme.mean_field import RDMWilsonCowan

# ── Parameters ────────────────────────────────────────────────────────────────

# the E-I presets of wc_mf_vs_sm.py (copied, with their comments there)
NEURON = dict(beta_E=0.4, beta_I=0.4, theta_E=15.0, theta_I=15.0,
              tau_int_E=20.0, tau_int_I=20.0, tau_ref_E=1.0, tau_ref_I=1.0,
              K_ref_E=0.0, K_ref_I=0.0, nu0=0.01, t_abs=4.0)
POINTS = {
    # E-I presets along w(E→I) through Fig. 5. Neuron, w(E→E) = 960, w(I→E) = -1200,
    # w(I→I) = -1200 and E_exc = E_inh = 24 mV all as published; only w(E→I) moves. On this
    # cut the mean field has one Hopf, at J_H = 885.98 (19.91 Hz; r_E 21.75 / r_I 18.69 Hz),
    # unstable BELOW J_H, and the cycle grows continuously from it, with the silent start and
    # a start at the fixed point agreeing everywhere (no bistability; ei_bifurcations.py and a
    # mean-field attractor map, 2026-11-19). The onset regime is narrow -- r_E max passes the
    # 1/t_abs = 250 Hz ceiling near 0.94 J_H -- so the presets are placed by behaviour, not
    # by fixed fractions of J_H. r_max is dt-resolved down to ~0.97 J_H (dt = 0.2/0.1/0.05).
    "EI-onset": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=875.0,     # E→I; 0.988 J_H: small cycle, r_E max ~62 Hz
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
    "EI-cycle": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=865.0,     # E→I; 0.976 J_H: cycle, r_E max ~100 Hz, still resolved
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
    "EI-burst": dict(NEURON,
                     w_EE=960.0,     # E→E
                     w_EI=800.0,     # E→I; 0.903 J_H: bursts past the ceiling, r_max NOT
                                     # resolved in dt (428/678/1022 Hz at dt 0.2/0.1/0.05)
                     w_IE=-1200.0,   # I→E
                     w_II=-1200.0,   # I→I
                     E_exc=24.0, E_inh=24.0),
}
E_RATIO = 0.8

hazard  = "poisson_exact"
IC      = "silent"
AGE_MAX = 25.0              # ms, only read by the "random" start

# numerical parameters
DTS    = (0.2, 0.1, 0.05)   # ms, halving; each must divide t_abs
TOL    = 1e-2
DEVICE = 'cpu'
T_EQUI, T_MEAS = 1000.0, 2000.0   # ms, as in wc_mf_vs_sm.py
N_BINS = 400                # phase bins of the profiles
P_RANGE = (0.5, 3.0)        # an observed order outside this is "out of range"
REM_TOL = 0.05              # H_rev remainder above this fraction of r_mean ln 2 is unresolved.
                            # 5 % for the presets (2026-11-19); likely lowered for the paper

run_sim   = True
run_plot  = True
overwrite = False   # the hash covers every parameter, so this is only for a change to the CODE


# ── Paths, dicts & switches ──────────────────────────────────────────────────

ROOT = Path(__file__).parents[1]
FIG  = Path(__file__).parent / "figures" / "EI-sigma_order"
KEYS = ("m", "sigma", "H_fwd", "H_rev", "m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")
POPS = ("E", "I", "tot")


def params(name: str, dt: float) -> dict:
    return dict(POINTS[name], E_ratio=E_RATIO, dt=dt, hazard=hazard)


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
            mf = RDMWilsonCowan(**params(name, dt), ic=IC, age_max=AGE_MAX, tol=TOL,
                                device=DEVICE)
            mf.trajectory(T=hlp.n_steps(T_EQUI, dt, name="T_EQUI"))
            tr = mf.entropy_trajectory(T=hlp.n_steps(T_MEAS, dt, name="T_MEAS"))
            hlp.save_atomic({"params": params(name, dt), "Q": mf.Q, "t_equi": T_EQUI,
                             "t_meas": T_MEAS, "mf_traj2": {k: tr[k].cpu() for k in KEYS}},
                            path)
            print(f"Saved to {path}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

def series(tr: dict, pop: str, key: str, dt: float) -> torch.Tensor:
    """ A per-step trace of one population (or the aggregate), per unit time (1/s). """
    x = tr[key + "_tot"] if pop == "tot" else tr[key][:, POPS.index(pop)]
    return x.double() / dt * 1e3


if run_plot:

    FIG.mkdir(parents=True, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    for name in POINTS:
        R = {}
        for dt in DTS:
            tr = torch.load(cache(name, dt), weights_only=False)["mf_traj2"]
            # whole cycles of r_E, one window for every population
            a, b, n = an.cycle_window(series(tr, "E", "m", dt))
            R[dt] = {"n": n, "ratio": an.cycle_amplitude_ratio(series(tr, "E", "m", dt)),
                     "per": (b - a) * dt / n if n else float("nan")}
            for p in POPS:
                r, s = series(tr, p, "m", dt), series(tr, p, "sigma", dt)
                hf, hr = series(tr, p, "H_fwd", dt), series(tr, p, "H_rev", dt)
                R[dt][p] = dict(r=r, s=s, rbar=r[a:b].mean().item(), rmax=r[a:b].max().item(),
                                sig=s[a:b].mean().item(), Hf=hf[a:b].mean().item(),
                                Hr=hr[a:b].mean().item(),
                                chk=(hr[a:b] - hf[a:b] - s[a:b]).abs().max().item())

        pr = [R[dt]["per"] for dt in DTS]
        p_per, why_per = an.observed_order(pr, P_RANGE)
        out(f"=== {name}   cycles of r_E {[R[dt]['n'] for dt in DTS]}  "
            f"amp ratio {[round(R[dt]['ratio'], 4) for dt in DTS]}  "
            f"max|H_rev - H_fwd - sigma| "
            f"{max(R[dt][p]['chk'] for dt in DTS for p in POPS):.2e} nats/s")
        out(f"  period (ms)  dt=0.2 {pr[0]:.6g}  0.1 {pr[1]:.6g}  0.05 {pr[2]:.6g}   "
            f"p = {p_per:.3f} {why_per}  (diffs {pr[0] - pr[1]:+.4g}, {pr[1] - pr[2]:+.4g})")
        unresolved = [f"period order {why_per}"] if why_per else []
        for p in POPS:
            out(f"  [{p}]")
            for lab, key in (("sigma (nats/s)", "sig"), ("r_mean (Hz)", "rbar"),
                             ("r_max (Hz)", "rmax")):
                v = [R[dt][p][key] for dt in DTS]
                o, why = an.observed_order(v, P_RANGE)
                out(f"    {lab:<15} dt=0.2 {v[0]:.6g}  0.1 {v[1]:.6g}  0.05 {v[2]:.6g}   "
                    f"p = {o:.3f} {why}  (diffs {v[0] - v[1]:+.4g}, {v[1] - v[2]:+.4g})")
            # the divergent part of the path entropies: r_mean ln 2 per halving
            rem = []
            for c, f in ((0.2, 0.1), (0.1, 0.05)):
                rl = R[f][p]["rbar"] * math.log(2)
                dHf = R[f][p]["Hf"] - R[c][p]["Hf"]
                dHr = R[f][p]["Hr"] - R[c][p]["Hr"]
                rem.append(dHr - rl)
                out(f"    H {c:g}->{f:g}: dH_fwd {dHf:+.5g}  dH_rev {dHr:+.5g}  r_mean ln2 "
                    f"{rl:.5g}  remainders fwd {dHf - rl:+.5g}  rev {dHr - rl:+.5g}")
            rl = R[0.05][p]["rbar"] * math.log(2)
            if rem[1] > REM_TOL * rl:
                unresolved.append(f"{p}: dH_rev - r_mean ln 2 = {rem[1]:+.4g} > {REM_TOL:g} "
                                  f"r_mean ln 2 (remainders {rem[0]:+.4g}, {rem[1]:+.4g})")
        out(f"  verdict: {'UNRESOLVED (' + '; '.join(unresolved) + ')' if unresolved else 'resolved'}")

        # where in the cycle sigma sits, per population; diagnostic only
        if all(R[dt]["n"] for dt in DTS):
            ph = (torch.arange(N_BINS) + 0.5) / N_BINS
            fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 7))
            for j, p in enumerate(("E", "I")):
                for dt in DTS:
                    rE = R[dt]["E"]["r"]
                    axs[0, j].plot(ph, an.phase_profile(R[dt][p]["s"], rE, N_BINS),
                                   label=f"dt = {dt:g} ms")
                    axs[1, j].plot(ph, an.phase_profile(R[dt][p]["r"], rE, N_BINS),
                                   label=f"dt = {dt:g} ms")
                axs[0, j].set_title(f"{name}, {p}: per-step σ against the phase of r_E "
                                    f"({N_BINS} bins; diagnostic only)", fontsize=8)
                axs[0, j].set_ylabel("σ per unit time (nats / neuron / s)")
                axs[1, j].set_ylabel(f"r_{p} (Hz)")
                axs[1, j].set_xlabel("phase (0 = upward crossing of mean r_E)")
            for ax in axs.flat:
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG / f"{name}.png", dpi=120)
            plt.close(fig)

    (FIG / "tables.txt").write_text(
        f"ei_sigma_order.py; mean field only; {hazard}; {IC} start; T_equi {T_EQUI:g} ms, "
        f"T_meas {T_MEAS:g} ms; averages over whole cycles of r_E (analysis.cycle_window); "
        f"unresolved if the period's order is outside {P_RANGE} or the sign changes, or if a "
        f"population's H_rev grows faster than r_mean ln 2 by more than {REM_TOL:g} r_mean ln 2 "
        f"at the finer halving (the resolved E1 points have small negative remainders; "
        f"E1-30-strong +10 against r_mean ln 2 = 33)\n\n" + "\n".join(lines) + "\n")
