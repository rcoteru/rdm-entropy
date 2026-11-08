"""
Is the absolute refractory window t_abs load-bearing at S1-sat, now that it is an explicit
model parameter rather than folded into a kernel? Never checked at this preset before
([[P1 refractory-window]] cites no dump for it). S1, by contrast, is ALREADY settled: the
windowed rate there is bit-identical at t_abs = 0 vs t_abs = 1 ms (S1-api_checks.py check 3),
so it is not re-run here.

S1-sat is the rate-ceiling point -- strong drive, deep own kernel (K_ref = 25, tau_ref = 25),
J = 40 -- and CLAUDE.md names it as the one where "the realised rate should sit at 1/t_abs".
That is the opposite end from S1's "the window never gets close enough to matter", so it is
where t_abs ought to show up in the rate if it shows up anywhere in this one-population system.

Method: hold dt fixed at a step that divides every t_abs in the ladder, sweep t_abs from 0 (the
explicit "no dead time" case CLAUDE.md says has to be asked for) up through and past the
preset's own t_abs = 1 ms, and read the stationary rate and entropy production at each. The
closed-form ceiling this is read against is route B's own: r <= 1/t_abs for t_abs > 0, with no
cap at all for t_abs = 0 (the hazard itself is unbounded; only the dead time bounds the rate
under route B).

Exploration-grade per the 2026-11-08 brief: one dt, no Richardson ladder -- this establishes
WHETHER the window matters at S1-sat, which a single resolved dt answers, not a converged
number for a final figure.
"""

import time

import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.rdm_single import RDMIsingModel


PRESET = "S1-sat"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a preset

DT = 0.25   # divides every t_abs below, including 0 trivially; finer than S1-mf_vs_sm.py's 0.5
            # because the ceiling test is exactly where the hazard is largest and dt-sensitive
T_EQUI_MS, T_MEAS_MS = 6000.0, 4000.0

T_ABS_LADDER = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)   # 1.0 is the preset's own value; 0 is the
                                                  # explicit "no window" case, not a default
TOL = 1e-2

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def run(t_abs: float) -> dict:
    base = {k: v for k, v in pre.params(PRESET, hazard=HAZARD, dt=DT).items()
           if k not in ("dt", "t_abs")}
    Qm = hlp.fit_age_grid(
        lambda qm: RDMIsingModel(**base, dt=DT, t_abs=t_abs, Qm=qm, device=DEVICE),
        tol=TOL, equi=hlp.n_steps(T_EQUI_MS + T_MEAS_MS, DT), verbose=False)
    net = RDMIsingModel(**base, dt=DT, t_abs=t_abs, Qm=Qm, device=DEVICE)
    net.forward(hlp.n_steps(T_EQUI_MS, DT))
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS_MS, DT), pb=False)
    r_tot = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r_tot)
    window = r_tot[a:z] if n_cyc else r_tot
    x = window - window.mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    return {
        "Q": Qm[0], "r": float(window.mean()), "r_max": float(r_tot.max()),
        "sigma": float(traj["sigma_tot"][a:z if n_cyc else slice(None)].mean()) / DT,
        "n_cyc": n_cyc, "acf1": acf1, "trunc": float(net.truncation_error()),
    }


print("=" * 86)
print(f"{PRESET}, hazard = {HAZARD}, dt = {DT:g}: the absolute refractory window t_abs")
print("=" * 86)
print(f"{'t_abs':>8}{'1/t_abs':>10}{'Q':>7}{'r [1/ms]':>13}{'r_max [1/ms]':>14}"
      f"{'r*t_abs':>10}{'sigma/dt':>13}{'n_cyc':>7}{'acf1':>8}")

rows = {}
t0 = time.time()
for t_abs in T_ABS_LADDER:
    out = run(t_abs)
    rows[t_abs] = out
    ceiling_col = f"{out['r'] * t_abs:10.4f}" if t_abs > 0 else f"{'n/a':>10}"
    print(f"{t_abs:8.2f}{(1.0 / t_abs if t_abs > 0 else float('inf')):10.3f}{out['Q']:7d}"
          f"{out['r']:13.8f}{out['r_max']:14.8f}{ceiling_col}"
          f"{out['sigma']:13.4e}{out['n_cyc']:7d}{out['acf1']:+8.3f}"
          f"   ({time.time() - t0:.0f}s)", flush=True)

r0 = rows[0.0]["r"]
r1 = rows[1.0]["r"]
rel = abs(r1 - r0) / abs(r0)
print(f"\nrate at t_abs=1 vs t_abs=0: {r1:.8f} vs {r0:.8f}, relative difference {rel:.4e}")
if rel > 1e-3:
    print("LOAD-BEARING at S1-sat: the window measurably changes the rate here, unlike at S1")
    print("(bit-identical there, S1-api_checks.py check 3). This is CLAUDE.md's stated")
    print("expectation -- t_abs is 'load-bearing at strong drive, which is... what S1-sat is")
    print("for' -- now with a number rather than a claim.")
else:
    print("The window is STILL inert here to 1e-3 relative, which contradicts the expectation")
    print("recorded in CLAUDE.md and S1-api_checks.py check 3's closing note. Stop-and-flag:")
    print("this is a surprise, not a result to file quietly -- check that S1-sat's own rate")
    print("(0.008 /ms at dt=0.5 per S1-sat-age_grid.py) is actually close enough to 1/t_abs")
    print("for the window to have anything to do, since a rate far below the ceiling would")
    print("correctly show no effect regardless of t_abs.")

print(f"\nCeiling check, on the MEAN rate (r*t_abs, not r_max*t_abs -- the per-neuron dead time")
print(f"bounds each neuron's own ISI, not the instant population rate: many independently-")
print(f"phased neurons can still fire in the same bin, so r_max can and does sit far above")
print(f"1/t_abs without the per-neuron cap being violated). At the largest t_abs in the")
print(f"ladder, r*t_abs = {rows[T_ABS_LADDER[-1]]['r'] * T_ABS_LADDER[-1]:.4f} (1.0 would mean")
print(f"the stationary rate sits exactly at the dead-time ceiling; far under 1 means the")
print(f"kernel's own hazard, not the dead time, is what is setting the mean rate here).")

worst_acf1 = min(out["acf1"] for out in rows.values() if out["n_cyc"] == 0) \
    if any(out["n_cyc"] == 0 for out in rows.values()) else float('nan')
print(f"\nStep-artifact screen: acf1 printed per row above (n_cyc=0 rows are a settled fixed")
print(f"point, where acf1 has no oscillation to measure and is read as a sampling-noise")
print(f"floor, not a cycle diagnostic); on any cycling row (n_cyc>0) acf1 >= 0.97 is the bar.")
