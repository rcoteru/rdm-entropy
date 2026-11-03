"""
Where the Hopf sits, per hazard, as the solver step refines.

The companion to ising-dt_conv.py, and the reason its cycle regime is measured at J = 17
rather than at 15. That script asks how fast each kernel's observables converge; this one asks
a prior question -- whether the two operating points it measures are even the same kind of
state for every kernel, which they need not be below dt0.

rho(J) is the spectral radius of the one-step Jacobian at the fixed point, so the Hopf is
rho = 1 and is located by bisection. At dt = dt0 all of async, escape and poisson are the same
map, bit for bit, so the three must agree to every digit there: that is the control, and it is
what makes a disagreement below dt0 readable as a statement about the kernels rather than
about the root finder. exp_surv is not calibrated to the quantum and is expected to differ
everywhere, including at dt = dt0.

What came out of it, at the standard Ising set (E = 0, beta = 6, theta = 0.5,
tau_int = tau_ref = K_ref = 15, dt0 = 1):

    dt       async    escape   poisson
     1      12.0588   12.0588  12.0588     <- the control: one map, one spectrum
     1/2    12.187    11.971   13.326
     1/4    12.263    11.941   13.872
     1/8      --        --     ~14.13

Refining does not move the bifurcation in one direction: async rises, escape falls, poisson
rises hard and is still rising at dt = 1/8, its increments halving cleanly toward ~14.36. The
consequence is the one that matters for any J-indexed comparison -- at J = 14 and dt = 1/4 the
poisson fixed point is still stable (rho = 0.99968) where async and escape have both lost
stability. Below dt0 the attractor at a given J is kernel-dependent, not merely the numbers
measured on it.

That is also why J = 15 was a poor choice for the convergence study: it sat 0.64 above
poisson's own bifurcation while the other two sat 2.7-3.1 above theirs, and the amplitude of a
Hopf cycle goes like sqrt(J - Jc).

The crossing is shallow -- rho - 1 runs ~1e-3 over a unit of J -- so a scan that looks for a
cycle by its AMPLITUDE will not resolve one until well past the bifurcation. A reported onset
near J ~ 14.5 from such a scan is not in conflict with rho = 1 at 12.06.
"""

import time

import torch

from rdme.rdm_single import RDMIsingModel


PARAMS: dict = dict(E=0.0, beta=6.0, theta=0.5, tau_int=15.0, tau_ref=15.0, K_ref=15.0,
                    dt0=1.0)

HAZARDS = ("async", "escape", "poisson", "exp_surv")
DTS     = (1.0, 0.5, 0.25)
U_MAX   = 180.0      # ms, held fixed across dt so the grid is not a second variable
BRACKET = (6.0, 20.0)
N_BISECT = 22        # 14 units of J to ~3e-6; the cost is one spectrum per step
PROBE_J = (13.0, 14.0, 14.5, 15.0, 16.0)

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def spectral_radius(J: float, hazard: str, dt: float) -> float:
    """ max |eig| of the linearized one-step map at the fixed point.

    Over every root rather than a chosen one: the set is a single point throughout the J range
    scanned here, but taking the max is what makes that an observation rather than an
    assumption -- a second branch appearing would show up as a discontinuity in rho instead of
    being silently picked up or dropped. """
    net = RDMIsingModel(J, **PARAMS, dt=dt, hazard=hazard,
                        Qm=[max(1, round(U_MAX / dt))], device=DEVICE)
    roots = net.fixed_points(n_grid=401)
    if roots.numel() == 0:
        raise RuntimeError(f"no fixed point at J={J}, hazard={hazard}, dt={dt}")
    return max(float(net.fp_eigenvalues(r).abs().max()) for r in roots)

o
def hopf(hazard: str, dt: float, lo: float = BRACKET[0], hi: float = BRACKET[1]) -> float:
    """ The J at which rho crosses 1, by bisection on the bracket. """
    r_lo, r_hi = spectral_radius(lo, hazard, dt), spectral_radius(hi, hazard, dt)
    if not (r_lo < 1.0 < r_hi):
        raise ValueError(f"{hazard} at dt={dt}: bracket [{lo}, {hi}] does not straddle the "
                         f"crossing (rho = {r_lo:.6f}, {r_hi:.6f}). Widen it.")
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, hazard, dt) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == '__main__':
    for dt in DTS:
        for hazard in HAZARDS:
            t0 = time.time()
            print(f"dt = {dt:6.4f}  {hazard:9s}  Hopf at J = {hopf(hazard, dt):9.6f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
        # the shape of rho(J) around it: a bisected root says nothing about how steep the
        # crossing is, and the steepness is what decides whether an amplitude scan can see it
        for hazard in HAZARDS:
            vals = " ".join(f"{spectral_radius(J, hazard, dt):.5f}" for J in PROBE_J)
            print(f"    rho at J = {'/'.join('%g' % J for J in PROBE_J)}, "
                  f"{hazard:9s}: {vals}", flush=True)
