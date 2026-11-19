""" E-I bifurcations along w(E→I): the fixed points, their leading growth rates and every local
bifurcation of them on a grid of w(E→I), with no simulation of the model. Everything but w(E→I) is
Schwalger 2017 Fig. 5's.

stability.scan_cut finds the fixed points (batched Newton on the GPU, plus the exact 1-D starts
when w(I→I) = 0) and their growth rates (batched power iteration); stability.find_bifurcations
bisects every crossing of the unit circle with ARPACK (Hopf, period-doubling or branch point;
side, frequency) and every saddle-node on the fixed-point count. With w(I→I) != 0 a fixed point no Newton start reaches is not reported. """

import torch

import rdme.stability as stab
from rdme.mean_field import RDMWilsonCowanBatch

# ── Parameters ────────────────────────────────────────────────────────────────

# Schwalger, Deger & Gerstner 2017 Fig. 5 neuron (Delta_u = 2.5 mV), identical for E and I
NEURON = dict(beta_E=0.4, beta_I=0.4, theta_E=15.0, theta_I=15.0,
              tau_int_E=20.0, tau_int_I=20.0, tau_ref_E=1.0, tau_ref_I=1.0,
              K_ref_E=0.0, K_ref_I=0.0)
# w_XY is X onto Y (RDMWilsonCowan stores [[w_EE, w_IE], [w_EI, w_II]], rows = targets)
FIXED = dict(w_EE=960.0,    # E→E
             w_IE=-1200.0,  # I→E
             w_II=-1200.0,  # I→I
             E_exc=24.0, E_inh=24.0)
# Tried first, 2026-11-19: w(I→I) = 0 with E_inh = 24 (down to 18) has no crossing -- I,
# driven by E_inh alone, silences E -- and with E_inh = 15 one Hopf at 448.52 whose cycle is
# the large burst from the onset on, bistable with the fixed point below it: no gradual
# transition to place presets on.
E_RATIO = 0.8
W_MAX, N_W = 3840.0, 81     # scan of w(E→I), mV ms: 0 .. 4 x 960
R_CAP   = 0.5               # the scan ends where a stable rate reaches r t_abs = R_CAP

hazard, nu0, t_abs = "poisson_exact", 0.01, 4.0
dt      = 0.1               # ms
TOL     = 0.01              # mV ms, the bisection of each bifurcation
PI_T    = 1000.0            # ms of power iteration per growth rate, after as long a burn-in
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"   # the batched scan's


def build(w_EI: torch.Tensor, device: str):
    return RDMWilsonCowanBatch(E_RATIO, w_EI=w_EI, **FIXED, **NEURON, dt=dt, hazard=hazard,
                               nu0=nu0, t_abs=t_abs, device=device)


# ── Scan ─────────────────────────────────────────────────────────────────────

scan = stab.scan_cut(build, torch.linspace(0.0, W_MAX, N_W).tolist(), device=DEVICE, t_ms=PI_T)
for k, s in enumerate(scan):
    if any(mu < 0 and float(r.max()) * t_abs * 1e-3 >= R_CAP for mu, r in zip(s["mu"], s["r"])):
        print(f"stop: a stable rate reaches r t_abs = {R_CAP} at w(E→I) = {s['value']:g}")
        scan = scan[:k + 1]
        break

for s in scan:
    print(f"w(E→I) {s['value']:7.1f}: {len(s['m'])} fp  " + "  ".join(
        f"[r_E {float(r[0]):8.3f}  r_I {float(r[1]):8.3f} Hz  mu {float(mu):+.5f} /ms]"
        for r, mu in zip(s["r"], s["mu"])))

counts = [len(s["m"]) for s in scan]
print(f"\nfixed-point counts over the scan: {sorted(set(counts))}")
print(f"\nbifurcations (bisected to {TOL:g} mV ms):")
for lm in stab.find_bifurcations(scan, build, tol=TOL):
    if lm["kind"] == "saddle-node":
        print(f"  w(E→I) = {lm['value']:.3f} in [{lm['lo']:g}, {lm['hi']:g}]: saddle-node, "
              f"{lm['counts'][0]} -> {lm['counts'][1]} fixed points")
    else:
        print(f"  w(E→I) = {lm['value']:.3f} in [{lm['lo']:g}, {lm['hi']:g}]: {lm['kind']}, "
              f"unstable {lm['side']}; mu {lm['mu']:+.2e} /ms, {lm['freq_hz']:.3f} Hz "
              f"(period {1e3 / max(lm['freq_hz'], 1e-12):.3f} ms); r_E {float(lm['r'][0]):.3f} "
              f"r_I {float(lm['r'][1]):.3f} Hz")
