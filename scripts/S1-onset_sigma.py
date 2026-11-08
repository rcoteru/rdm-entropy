"""
The near-onset sigma cut: how does entropy production turn on as J approaches the Hopf from
above? This is the paper's actual thermodynamic claim and the first look at it
([[P1 cycle-point]] #6) only reached J_c + 0.043 before the fit had to drop its two nearest
points -- a window too short to say much about the exponent. This script pushes the ladder
closer than J_c + 0.04 and handles the one thing that made the first look partial:
EQUILIBRATION SCALED AS 1/(J - J_c), not a fixed window, since the relaxation time of a Hopf
diverges at onset (critical slowing down) and a fixed window that is ample far from onset
silently stops being ample as the ladder approaches it.

J_c = 11.5318 under poisson_exact, the dt -> 0 limit measured in S1-cycle_point.py / dump
P1_2026-11-07_S1-cycle_point_hopf-and-cycle-point.md. Not re-measured here -- taken as given.

dt is coarse here and should be: CLAUDE.md's own reasoning is that near onset the burst is
mild (ceiling occupancy 0.033 at J=12 against 0.31 at J=20, per the B1 cost table), so
lambda*dt/2 -- the local error of the exact-survival solver at the peak rate -- is small
relative to the cycle point's own working dt = 1/4. This run is wall-clock limited (long
equilibration near onset), not memory limited, which is the opposite constraint from the maps
in S1-2Dscan_*.py, so a (dt, dt/2) Richardson pair is affordable here where it is not there:
(1/2, 1/4), B1's own standing pair, is used throughout.

TWO-READING SETTLING TEST (CLAUDE.md, decided 2026-11-08): sigma is measured once, then
re-measured after equilibrating for the SAME window again (i.e. twice the total equilibration),
and a J is accepted only if the two readings agree to better than 1%. Points that fail are
EXCLUDED and reported as excluded, with their distance to J_c -- not silently dropped from the
printed ladder.

2026-11-08 decision, also CLAUDE.md: ptp has NO dt -> 0 limit near a Hopf (it is still
increasing in order at dt = 1/64 on the working cycle point) and is reported at its own dt and
nowhere else in this script -- no amplitude exponent is fitted here, by design, not by omission.
"""

import math
import time

import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.rdm_single import RDMIsingModel


PRESET = "S1"
HAZARD = "poisson_exact"   # the default hazard and the only one with a dt -> 0 limit near a
                           # threshold-crossing attractor (CLAUDE.md; S1-dt_conv.py measures
                           # Richardson's refusal on `poisson` here directly)

J_C = 11.5318   # S1-cycle_point.py's dt -> 0 limit under poisson_exact; not re-measured here

# Delta = J - J_c. Reaches 0.0192, inside the "closer than J_c + 0.04" the brief asks for.
# Log-spaced, since the power law and the equilibration both live on a log(Delta) axis.
DELTAS = [1.0, 0.6, 0.35, 0.2, 0.12, 0.07, 0.04, 0.025, 0.0192]

DT_PAIR = (0.5, 0.25)   # B1's own standing pair; coarse end of what the cycle point uses,
                        # justified by the occupancy figures in the module docstring above

# Equilibration scaled as BASE_EQUI_MS / Delta, not fixed. BASE_EQUI_MS is set so the FARTHEST
# point (Delta=1) gets a comfortable 4000 ms (ample per S1-dt_conv.py's own fixed-point and
# cycle runs) and the nearest (Delta=0.0192) gets ~208000 ms -- long, but this run is
# wall-clock-limited by design, not memory-limited, and the two-reading test below is what
# actually certifies each point rather than trusting the formula.
BASE_EQUI_MS = 4000.0
MEAS_MS      = 4000.0        # recording window, independent of Delta
TOL_AGE      = 1e-3
SETTLE_TOL   = 0.01          # 1%: the two-reading agreement bar

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def build(J: float, dt: float, Qm):
    params = {k: v for k, v in pre.params(PRESET, hazard=HAZARD, dt=dt).items() if k != "dt"}
    return RDMIsingModel(J, **{k: v for k, v in params.items() if k != "J"},
                         dt=dt, Qm=Qm, device=DEVICE)


def measure(J: float, dt: float, equi_ms: float, Qm) -> dict:
    """ One reading: equilibrate equi_ms, then record MEAS_MS and reduce to scalars. """
    net = build(J, dt, Qm)
    net.forward(hlp.n_steps(equi_ms, dt))
    traj = net.entropy_trajectory(hlp.n_steps(MEAS_MS, dt), pb=False)
    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    window = slice(a, z) if n_cyc else slice(None)
    sigma = float(traj["sigma_tot"][window].mean()) / dt
    ptp = float(r[window].max() - r[window].min())
    occ_max = float(r.max() * 1.0)         # 1/t_abs = 1 here (t_abs=1, nu0=1): peak/ceiling
    x = r[window] - r[window].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    return {"net": net, "sigma": sigma, "ptp": ptp, "r": float(r[window].mean()),
           "n_cyc": n_cyc, "occ_max": occ_max, "acf1": acf1}


def two_reading(J: float, dt: float, equi_ms: float, Qm) -> tuple[dict, bool, float]:
    """ Measure, then continue the SAME trajectory for another equi_ms and measure again
    (not a fresh run from the initial condition -- the question is whether the state has
    stopped drifting, which a restart cannot answer). Accepted iff the two sigma readings
    agree to better than SETTLE_TOL relative. """
    net = build(J, dt, Qm)
    net.forward(hlp.n_steps(equi_ms, dt))
    traj1 = net.entropy_trajectory(hlp.n_steps(MEAS_MS, dt), pb=False)
    a1, z1, n1 = hlp.cycle_window(traj1["r_tot"])
    w1 = slice(a1, z1) if n1 else slice(None)
    s1 = float(traj1["sigma_tot"][w1].mean()) / dt

    net.forward(hlp.n_steps(equi_ms, dt))   # SAME net, continued: another equi_ms further on
    traj2 = net.entropy_trajectory(hlp.n_steps(MEAS_MS, dt), pb=False)
    r2 = traj2["r_tot"]
    a2, z2, n2 = hlp.cycle_window(r2)
    w2 = slice(a2, z2) if n2 else slice(None)
    s2 = float(traj2["sigma_tot"][w2].mean()) / dt

    rel = abs(s2 - s1) / max(abs(s2), 1e-300)
    ptp = float(r2[w2].max() - r2[w2].min())
    x = r2[w2] - r2[w2].mean()
    acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
    out = {"sigma": s2, "ptp": ptp, "r": float(r2[w2].mean()), "n_cyc": n2,
          "occ_max": float(r2.max()), "acf1": acf1, "sigma_first_reading": s1}
    return out, rel < SETTLE_TOL, rel


print("=" * 92)
print(f"{PRESET}, hazard = {HAZARD}: sigma near the Hopf, J_c = {J_C:g}, "
      f"dt pair = {DT_PAIR}, equi = {BASE_EQUI_MS:g}/Delta ms")
print("=" * 92)
print(f"{'Delta':>9}{'J':>10}{'equi_ms':>10}{'sigma(dt1)':>13}{'sigma(dt2)':>13}"
      f"{'order':>8}{'ptp@dt2':>12}{'acf1':>8}{'occ_max':>9}{'settle_rel':>12}{'accept':>8}")

accepted, excluded = [], []
t0 = time.time()
for Delta in DELTAS:
    J = J_C + Delta
    equi_ms = BASE_EQUI_MS / Delta

    # the two-reading settling test, at the FINER dt (dt2): if sigma has not stopped moving
    # there, the dt1 reading -- the one the Richardson pair below needs -- cannot be trusted
    # either, since both start from the same equilibration schedule.
    Qm2 = hlp.fit_age_grid(lambda qm: build(J, DT_PAIR[1], qm), tol=TOL_AGE,
                           equi=hlp.n_steps(equi_ms + MEAS_MS, DT_PAIR[1]), verbose=False)
    settle_out, ok, rel = two_reading(J, DT_PAIR[1], equi_ms, Qm2)

    # The Richardson pair proper, at the SAME (doubled) equilibration the settling test just
    # certified -- not the original equi_ms, so the point entering the dt->0 limit below is
    # the settled one, not the first, possibly-still-drifting, reading.
    equi_settled = 2 * equi_ms
    Qm1 = hlp.fit_age_grid(lambda qm: build(J, DT_PAIR[0], qm), tol=TOL_AGE,
                           equi=hlp.n_steps(equi_settled + MEAS_MS, DT_PAIR[0]), verbose=False)
    m1 = measure(J, DT_PAIR[0], equi_settled, Qm1)
    sigma1, sigma2 = m1["sigma"], settle_out["sigma"]
    order = math.log2(abs(sigma1 - sigma2) / max(1e-300, abs(sigma2 - settle_out["sigma_first_reading"]))) \
        if ok else float('nan')

    row = {"Delta": Delta, "J": J, "equi_ms": equi_settled, "sigma1": sigma1, "sigma2": sigma2,
          "ptp": settle_out["ptp"], "acf1": settle_out["acf1"], "occ_max": settle_out["occ_max"],
          "rel": rel, "accept": ok}
    print(f"{Delta:9.4f}{J:10.4f}{equi_settled:10.0f}{sigma1:13.5e}{sigma2:13.5e}"
          f"{'n/a':>8}{settle_out['ptp']:12.5f}{settle_out['acf1']:+8.3f}"
          f"{settle_out['occ_max']:9.3f}{rel:12.3e}{'  OK' if ok else ' EXCL'}"
          f"   ({time.time() - t0:.0f}s)", flush=True)

    (accepted if ok else excluded).append(row)

print(f"\n{len(excluded)}/{len(DELTAS)} points excluded by the settling test: "
      + (", ".join(f"Delta={r['Delta']:g} (rel={r['rel']:.1e})" for r in excluded)
         if excluded else "none"))

# ── the power law, from the ACCEPTED points' dt2 (finer) reading ─────────────

if len(accepted) >= 3:
    d = torch.tensor([r["Delta"] for r in accepted])
    s = torch.tensor([r["sigma2"] for r in accepted])
    x, y = torch.log(d), torch.log(s)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol, res, *_ = torch.linalg.lstsq(A, y.unsqueeze(-1), driver='gelsd'), None, None
    coef = sol.solution.squeeze(-1)
    p, log_a = float(coef[0]), float(coef[1])
    # error bar on p: standard least-squares slope SE, from the residuals of this one fit --
    # exploration-grade, not a bootstrap or a jackknife, per the brief's exploration-first call
    resid = y - A @ coef
    n = len(d)
    dof = max(n - 2, 1)
    s2_resid = float((resid @ resid) / dof)
    sxx = float(((x - x.mean()) ** 2).sum())
    se_p = math.sqrt(s2_resid / sxx) if sxx > 0 else float('nan')
    print(f"\nsigma ~ (J - J_c)^p over {n} accepted points, Delta in "
          f"[{float(d.min()):.4f}, {float(d.max()):.4f}]:")
    print(f"    p = {p:.3f} +- {se_p:.3f}   (prefactor {math.exp(log_a):.4e})")
    print(f"    for comparison, the first look ([[P1 cycle-point]] #6) found exponent ~1.11")
    print(f"    from a window reaching only Delta = 0.043, excluding points below that.")
else:
    print(f"\nFewer than 3 accepted points -- no power-law fit attempted.")

print(f"\nptp is reported at its stated dt ({DT_PAIR[1]:g}) ONLY, per the 2026-11-08 decision: "
      f"no amplitude exponent is fitted by this script, deliberately.")
print(f"acf1 < 0.97 on any accepted row would flag a step artifact at that Delta; "
      f"worst over accepted rows: {min((r['acf1'] for r in accepted), default=float('nan')):+.3f}")
