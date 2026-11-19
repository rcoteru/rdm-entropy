""" E-I landmarks along w(E->I): fixed points and the leading mode of their Jacobian, no
simulation of the model. The growth rate on the grid comes from power iteration on the
linearized map, the crossings from ARPACK (analysis.leading_modes), and nothing from the dense
spectrum. Counts the fixed points on a grid of w(E->I), brackets every crossing of the unit
circle on each branch and bisects it, and reports which side is unstable and the critical
pair's frequency. Everything but w(E→I) is Schwalger 2017 Fig. 5's.

Fixed points: Newton (on the CPU) from a lattice over the physical box. With w(I→I) = 0 the
problem is also one-dimensional -- r_I = F_I(r_E) -- and a start inside every sign change of
G_E(m_E, F_I(m_E)) on a fine m_E grid is added; the lattice alone missed the saddle and the
unstable focus there. With w(I→I) != 0 there is no such guarantee: a fixed point no lattice
start reaches is not reported. """

import math

import torch

import rdme.analysis as an
from rdme.mean_field import RDMWilsonCowan

# ── Parameters ────────────────────────────────────────────────────────────────

# Schwalger, Deger & Gerstner 2017 Fig. 5 neuron (Delta_u = 2.5 mV), identical for E and I
NEURON = dict(beta_E=0.4, beta_I=0.4, theta_E=15.0, theta_I=15.0,
              tau_int_E=20.0, tau_int_I=20.0, tau_ref_E=1.0, tau_ref_I=1.0,
              K_ref_E=0.0, K_ref_I=0.0)
# w_XY is X onto Y (RDMWilsonCowan stores [[w_EE, w_IE], [w_EI, w_II]], rows = targets)
FIXED = dict(E_ratio=0.8,
             w_EE=960.0,    # E→E
             w_IE=-1200.0,  # I→E
             w_II=-1200.0,  # I→I
             E_exc=24.0, E_inh=24.0)
# Tried first, 2026-11-19: w(I→I) = 0 with E_inh = 24 (down to 18) has no crossing -- I,
# driven by E_inh alone, silences E -- and with E_inh = 15 one Hopf at 448.52 whose cycle is
# the large burst from the onset on, bistable with the fixed point below it: no gradual
# transition to place presets on.
W_MAX, N_W = 3840.0, 81     # scan of w_EI (E→I), mV ms: 0 .. 4 x 960
R_CAP   = 0.5               # stop once a stable rate reaches r t_abs = R_CAP

hazard, nu0, t_abs = "poisson_exact", 0.01, 4.0
dt      = 0.1               # ms
N_GRID  = 4001              # m_E grid over the physical range [0, dt/t_abs] (w(I→I) = 0)
N_LAT   = 12                # Newton lattice per axis over [0, dt/t_abs]^2
BISECT_TOL = 0.01           # mV ms
PI_T    = 1000.0            # ms of power iteration for the leading growth rate, after as long
                            # a burn-in. Against ARPACK on the Fig. 5 cut: within 2e-5 /ms;
                            # within 5e-4 of the dense value where two leading pairs are
                            # degenerate (w(E→I) = 1968), converging slowly with PI_T


# ── Fixed points and spectra ─────────────────────────────────────────────────

def model(w_EI: float):
    return RDMWilsonCowan(**FIXED, w_EI=w_EI, **NEURON, dt=dt, hazard=hazard, nu0=nu0,
                          t_abs=t_abs)


def guesses(mf) -> torch.Tensor:
    """ Newton starts: the lattice, and with w(I→I) = 0 one start inside every sign change of
    G_E(m_E, F_I(m_E)). """
    hi  = dt / t_abs
    ax  = torch.linspace(hi / (2 * N_LAT), hi * (1 - 1 / (2 * N_LAT)), N_LAT,
                         dtype=torch.float64)
    lat = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
    if FIXED["w_II"] != 0.0:
        return lat
    mE = torch.linspace(0.0, hi, N_GRID, dtype=torch.float64)
    z  = torch.stack([mE, torch.zeros_like(mE)], -1)
    mI = (z - mf.fp_residual(z))[:, 1]           # F_I, which reads m_E only
    g  = mf.fp_residual(torch.stack([mE, mI], -1))[:, 0]
    i  = ((g[:-1] * g[1:]) <= 0).nonzero().flatten()
    return torch.cat([lat, torch.stack([0.5 * (mE[i] + mE[i + 1]),
                                        0.5 * (mI[i] + mI[i + 1])], -1)])


def point(w_EI: float) -> list[dict]:
    """ Every fixed point at w_EI: rates (Hz) and the leading growth rate mu (1/ms). """
    mf = model(w_EI)
    g = guesses(mf)
    out = [dict(m=m, r=(m / dt * 1e3).tolist(), mu=an.leading_growth(mf.state_map,
                                                       mf.flatten_state(*mf.fp_state(m)),
                                                       dt, PI_T))
           for m in (mf.fixed_points(guesses=g) if len(g) else [])]
    return sorted(out, key=lambda f: f["r"][0])


def mode_at(w_EI: float, m_near: torch.Tensor) -> dict:
    """ The fixed point at w_EI nearest m_near, with its leading mode from ARPACK (mu in 1/ms,
    omega in rad/ms): for the bisection of a crossing, where the critical mode is isolated. """
    mf = model(w_EI)
    m = min(mf.fixed_points(guesses=guesses(mf)), key=lambda r: float((r - m_near).norm()))
    o = an.leading_modes(mf.state_map, mf.flatten_state(*mf.fp_state(m)), dt, k=2)
    return dict(m=m, r=(m / dt * 1e3).tolist(), mu=float(o["mu"][0]),
                omega=float(o["omega"][0]))


def nearest(fps: list[dict], m: torch.Tensor) -> dict:
    return min(fps, key=lambda f: float((f["m"] - m).norm()))


def freq(f: dict) -> float:
    """ Hz of a fixed point's leading mode. """
    return abs(f["omega"]) / (2 * math.pi) * 1e3


def kind(f: dict) -> str:
    if abs(f["omega"]) * dt < 1e-9:
        return "fold (real +1)"
    if abs(abs(f["omega"]) * dt - math.pi) < 1e-9:
        return "flip (real -1)"
    return "Hopf (complex pair)"


# ── Scan ─────────────────────────────────────────────────────────────────────

W = torch.linspace(0.0, W_MAX, N_W).tolist()
scan = []
for w in W:
    fps = point(w)
    scan.append((w, fps))
    print(f"w(E→I) {w:7.1f}: {len(fps)} fp  " + "  ".join(
        f"[r_E {f['r'][0]:8.3f}  r_I {f['r'][1]:8.3f} Hz  mu {f['mu']:+.5f} /ms]"
        for f in fps), flush=True)
    stable_cap = [f for f in fps if f["mu"] < 0 and max(f["r"]) * t_abs * 1e-3 >= R_CAP]
    if stable_cap:
        print(f"stop: a stable rate reaches r t_abs = {R_CAP} at w(E→I) = {w:g}")
        break

counts = [len(f) for _, f in scan]
print(f"\nfixed-point counts over the scan: {sorted(set(counts))}; "
      f"changes at " + (", ".join(f"{scan[i][0]:g}->{scan[i + 1][0]:g} "
                                  f"({counts[i]}->{counts[i + 1]})"
                                  for i in range(len(scan) - 1)
                                  if counts[i] != counts[i + 1]) or "none"))
w0, f0 = scan[0]
print("at w(E→I) = 0: " + "; ".join(
    f"r_E {f['r'][0]:.4f} r_I {f['r'][1]:.4f} Hz, mu {f['mu']:+.6f} /ms "
    f"({'stable' if f['mu'] < 0 else 'unstable'})" for f in f0))

# crossings of the unit circle along each branch, followed by nearest root
print("\ncrossings (bisected to "
      f"{BISECT_TOL:g} mV ms on the branch through the bracket's left root):")
for i in range(len(scan) - 1):
    (wa, fa), (wb, fb) = scan[i], scan[i + 1]
    if len(fa) != len(fb):      # a fold: bisect on the count
        lo, hi = wa, wb
        while hi - lo > BISECT_TOL:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if len(point(mid)) == len(fa) else (lo, mid)
        f_lo = point(lo)
        print(f"  w(E→I) = {0.5 * (lo + hi):.3f} in [{wa:g}, {wb:g}]: fold, {len(fa)} -> "
              f"{len(fb)} fixed points; just before it: " + "; ".join(
                  f"r_E {f['r'][0]:.3f} r_I {f['r'][1]:.3f} Hz, mu {f['mu']:+.6f} /ms"
                  for f in f_lo))
        continue
    for f in fa:
        g = nearest(fb, f["m"])
        if (f["mu"] < 0) == (g["mu"] < 0):
            continue
        # the bracket's ends and every midpoint from ARPACK: near a crossing the critical
        # mode is isolated
        lo, hi = wa, wb
        flo, fhi = mode_at(wa, f["m"]), mode_at(wb, g["m"])
        while hi - lo > BISECT_TOL:
            mid = 0.5 * (lo + hi)
            h = mode_at(mid, flo["m"])
            if (h["mu"] < 0) == (flo["mu"] < 0):
                lo, flo = mid, h
            else:
                hi, fhi = mid, h
        J_H = 0.5 * (lo + hi)
        crit = flo if abs(flo["mu"]) < abs(fhi["mu"]) else fhi
        side = "above" if fhi["mu"] > 0 else "below"
        print(f"  w(E→I) = {J_H:.3f} in [{wa:g}, {wb:g}]: {kind(crit)}, unstable "
              f"{side}; critical mu {crit['mu']:+.2e} /ms, frequency {freq(crit):.3f} Hz "
              f"(period {1e3 / max(freq(crit), 1e-12):.3f} ms); r_E {crit['r'][0]:.3f} "
              f"r_I {crit['r'][1]:.3f} Hz")
