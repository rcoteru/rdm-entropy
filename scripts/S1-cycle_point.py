"""
Where the Hopf sits at S1 under the exact-survival solver, and which J to call the cycle point.

Three things, all off one J scan, because the columns already exist and answering them costs
no extra compute:

  1. J_c, the Hopf, as a dt -> 0 LIMIT. rho(J) is the spectral radius of the one-step Jacobian
     at the fixed point, so the bifurcation is rho = 1 and is located by bisection. Done at a
     (dt, dt/2) pair plus one coarser step, so the pair's own order can be read off a triple
     rather than assumed -- which is what CLAUDE.md's reporting rule asks for.

  2. The cycle point. The amplitude of a Hopf cycle goes like sqrt(J - J_c), so a point chosen
     just past onset is read in the sensitive neighbourhood of a bifurcation that is itself
     still moving with dt. The candidate is J = 17; this says whether it is far enough past
     onset to be readable and not so far that the exponential rate has made the step hopeless.
     Raul sets POINTS["S1/cycle"] from the dump -- this script proposes and does not add.

  3. Three things that ride along:
     - EPR ACROSS THE HOPF. sigma = 0 identically on the stable side, so onset is continuous
       from zero and the shape past onset is the question. First test: sigma ~ (J - J_c), i.e.
       ~ amplitude^2. A FIRST LOOK, not a converged measurement.
     - THE UPPER TRANSITION. Route A at a driven point showed a Hopf at J ~ 3.0 and an
       unexplained RE-STABILIZATION at J ~ 15.9. The suspicion is that it was the route-A rate
       ceiling, which route B does not have. J = 12-20 crosses it, so: does the cycle survive
       to J = 20, and what is the ceiling occupancy along the way?
     - CEILING OCCUPANCY. Under route B the only cap on the realised rate is the dead time,
       1/t_abs = 1 /ms. Two columns for it: the fraction of mass sitting in the n_abs blocked
       age bins, and max(rate)*t_abs.

Preset S1, hazard poisson_exact (the solver: single-signed per-bin error, monotone in dt,
which is what a Richardson limit is taken with). dt divides t_abs = 1 ms throughout.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.epr as epr
import rdme.hazards as hzd
import rdme.helpers as hlp
import rdme.kernels as krn
import rdme.plots as rp
import rdme.presets as pre
from rdme.helpers import orders, richardson
from rdme.rdm_batch import RDMIsingModelBatch
from rdme.rdm_single import RDMIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

PRESET = "S1"
HAZARD = "poisson_exact"   # the solver; see the docstring

PARAMS: dict = pre.params(PRESET, hazard=HAZARD, dt=0.5)
MODEL: dict = {k: v for k, v in PARAMS.items() if k not in ("dt", "hazard")}

# The Richardson triple. (DT_PAIR) = the (dt, dt/2) pair the standing rule asks for; DT_COARSE
# sits one doubling above it so the observed order of that pair can be READ rather than assumed
# at 1. All three divide t_abs = 1 ms.
DT_COARSE = 1.0
DT_PAIR   = (0.5, 0.25)
DTS       = (DT_COARSE,) + DT_PAIR

# The scan. 12-20 crosses the J at which route A re-stabilized (15.9) and was laid out around
# the 14.4-14.6 estimate the brief carried. It is NOT a near-onset window: the measured J_c is
# 11.53 (see the dump), so every point of it sits 0.5 to 8.5 above onset and a power law fitted
# across it is a global exponent, not the critical one.
#
# J_ONSET is the window that answers "is sigma ~ (J - J_c)?". It hugs the measured J_c from
# above, in a geometric ladder so that log(J - J_c) is evenly sampled -- which is what a
# log-log slope wants and what a linear grid over [12, 20] does not give.
J_SCAN = torch.linspace(12.0, 20.0, 81)      # spacing 0.1
J_C_MEASURED = 11.5339                       # from the hopf stage; see the dump
J_ONSET = J_C_MEASURED + torch.logspace(-1.5, 0.7, 34)   # J_c + 0.0316 ... J_c + 5.01

# Bisection bracket for rho = 1. Wide enough to straddle both the legacy 12.06 and anything
# route B is expected to give; the bisection asserts that it does.
BRACKET  = (8.0, 24.0)
N_BISECT = 24          # 16 units of J to ~1e-6; the cost is one spectrum per step

# Monotonicity at the proposed cycle point: B1 cannot use poisson_exact unless it converges
# from one side. Four steps so there are two successive differences to compare signs on.
DT_MONO = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)

TOL     = 1e-5     # age-grid accuracy. Tight: the grid error here contaminates a dt difference
DT_FIT  = 0.25     # step the age window is fitted at; U_max in ms is held across the sweep
TOL_FIT = TOL / 100   # fitted two decades tighter than it is asserted -- see S1-dt_conv.py

T_EQUI, T_MEAS = 8000.0, 4000.0   # ms. ~60 cycles recorded at the ~67 ms period of S1 past onset

# CRITICAL SLOWING DOWN, and how the onset window is guarded against it.
#
# Just above a Hopf the approach to the cycle is exponential at a rate mu proportional to
# (J - J_c), so the time to REACH the attractor diverges as J -> J_c and no fixed equilibration
# settles the whole window. The first run of this scan did exactly that: amplitude flattened at
# ~0.010 /ms and sigma at ~2e-5 nats/ms below J - J_c ~ 0.1, which is a residual transient read
# as a cycle, and a power law through it came back with an exponent of 0.07 where the normal
# form says 0.5.
#
# The guard is a REPEATED MEASUREMENT, not a within-record statistic. Measuring the amplitude
# over the first and second halves of one 4000 ms record would be useless here, and in exactly
# the wrong direction: a transient creeping toward the attractor with a 30000 ms time constant
# changes by 7% over 2000 ms and would be passed as settled by any tolerance loose enough to
# admit real data. So the onset window is measured TWICE -- once after T_EQUI_ONSET, once after
# another T_EQUI_ONSET on top -- and `settle` is the ratio of the two amplitudes. The lever arm
# is then the equilibration time itself, 60000 ms, rather than half a record.
#
# The ladder starts at J_c + 0.0316 rather than J_c + 0.003. From rho(J) measured in the hopf
# stage, mu ~ 2.3e-3 * (J - J_c) /ms, so x = 0.0316 is a 13600 ms time constant -- 4.4 of them
# inside T_EQUI_ONSET, and 8.8 by the second measurement. At x = 0.003 it would be 135000 ms,
# i.e. not settleable at any affordable window, which is why those points are not attempted.
# The top of the ladder is J_c + 5, which overlaps the far-field scan so the two can be read
# against each other.
T_EQUI_ONSET = 60000.0            # ms per stage; the second measurement is at 2x this
SETTLE_TOL   = 0.01               # |amp(2x) / amp(1x) - 1| a point must be inside to be fitted

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
KEY = dict(PARAMS, preset=PRESET, t_equi_onset=T_EQUI_ONSET, j=(float(J_SCAN[0]), float(J_SCAN[-1]), len(J_SCAN)),
           j_onset=(float(J_ONSET[0]), float(J_ONSET[-1]), len(J_ONSET)),
           dts=list(DTS), dt_mono=list(DT_MONO), bracket=list(BRACKET), n_bisect=N_BISECT,
           t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL, tol_fit=TOL_FIT, dt_fit=DT_FIT)
CACHE_HOPF = hlp.cache_path(STEM, KEY, tag="hopf", root=CACHE_DIR)
CACHE_SCAN = hlp.cache_path(STEM, KEY, tag="scan", root=CACHE_DIR)
CACHE_MONO = hlp.cache_path(STEM, KEY, tag="mono", root=CACHE_DIR)

run_hopf  = True
run_scan  = True
run_mono  = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE

for _dt in set(DTS) | set(DT_MONO) | {DT_FIT}:
    hzd.check_step(HAZARD, PARAMS["nu0"], _dt)


def age_window() -> float:
    """ U_max = Qm*dt in MILLISECONDS, fitted once at DT_FIT and held across the sweep.

    Refitting per dt lays up to `tol` of grid error on top of the dt differences the whole
    exercise is about, so the window is a physical length and Qm = U_max/dt follows it.

    Fitted at the BOTTOM of the scan, J = 12, which is the binding case: that is where the
    rate is lowest, the firing interval longest and the mass reaching furthest back in age. A
    window that holds there holds everywhere above it. """
    J0 = float(J_SCAN[0])
    Qm = hlp.fit_age_grid(
        lambda q: RDMIsingModel(J0, **MODEL, dt=DT_FIT, hazard=HAZARD, Qm=q, device="cpu"),
        tol=TOL_FIT, equi=hlp.n_steps(T_EQUI + T_MEAS, DT_FIT))
    return Qm[0] * DT_FIT


# ── 1. The Hopf, by bisection on rho(J) ───────────────────────────────────────

def spectral_radius(J: float, dt: float, Q: int) -> float:
    """ max |eig| of the linearized one-step map over EVERY fixed point at this J.

    Over every root rather than a chosen one: taking the max makes the single-branch structure
    an observation instead of an assumption -- a second branch appearing shows up as a
    discontinuity in rho rather than being silently picked up or dropped. """
    net = RDMIsingModel(J, **MODEL, dt=dt, hazard=HAZARD, Qm=[Q], device="cpu")
    roots = net.fixed_points(n_grid=401)
    if roots.numel() == 0:
        raise RuntimeError(f"no fixed point at J={J}, dt={dt}")
    return max(float(net.fp_eigenvalues(r).abs().max()) for r in roots)


def hopf(dt: float, Q: int) -> float:
    """ The J at which rho crosses 1, by bisection on BRACKET. """
    lo, hi = BRACKET
    r_lo, r_hi = spectral_radius(lo, dt, Q), spectral_radius(hi, dt, Q)
    if not (r_lo < 1.0 < r_hi):
        raise ValueError(f"dt={dt}: bracket {BRACKET} does not straddle rho = 1 "
                         f"(rho = {r_lo:.6f}, {r_hi:.6f}). Widen it.")
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        if spectral_radius(mid, dt, Q) < 1.0: lo = mid
        else:                                 hi = mid
    return 0.5 * (lo + hi)


if run_hopf and (overwrite or not CACHE_HOPF.exists()):
    u_max = age_window()
    print(f"\n[hopf] U_max = {u_max:.1f} ms, held across dt; bisecting rho(J) = 1 on {BRACKET}")
    # The MKL DGEBAL landmine: fp_eigenvalues SEGFAULTS (does not raise) when the age-grid
    # Jacobian overflows at large beta*h. S1 is beta = 6 and the bracket tops out at J = 24,
    # which is mild -- but this is the one stage that can take the process down, so the dt it
    # is on is printed before each bisection rather than after.
    out = {"u_max": u_max, "J_c": {}, "rho": {}}
    for dt in DTS:
        Q = max(1, round(u_max / dt))
        t0 = time.time()
        print(f"  dt = {dt:6.4f}  Q = {Q:5d} ...", end="", flush=True)
        out["J_c"][dt] = hopf(dt, Q)
        out["rho"][dt] = {float(J): spectral_radius(float(J), dt, Q)
                          for J in (13.0, 14.0, 14.5, 15.0, 16.0, 17.0, 18.0, 20.0)}
        print(f" J_c = {out['J_c'][dt]:.6f}   ({time.time() - t0:.0f}s)", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(out, CACHE_HOPF)
    print(f"  saved {CACHE_HOPF}")


# ── 2. The J scan, at the (dt, dt/2) pair ─────────────────────────────────────

# Fraction of the population sitting in the n_abs age bins the dead time blocks, and the peak
# rate against the 1/t_abs ceiling those bins impose. Both are read on the attractor, over a
# probe window after the recording, because the peak of a burst is an instant and the mean of
# a relaxation oscillator is nowhere near it.
def occupancy(mf, n_probe: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """ (max blocked-bin mass, max population rate, max HAZARD rate, max Phi) over n_probe steps.

    The two rates are different quantities and the distinction is the whole diagnostic. The
    POPULATION rate r = m/dt is what the dead time caps at 1/t_abs. The HAZARD rate
    lam = nu0 e^{beta h} is unbounded, and it is the one hazards.step_error wants: the local
    relative error of the exact-survival map is lam*dt/2 at the fastest-firing cohort, and
    nu0*dt says nothing about it. Sizing the step on r instead under-reports it by orders.

    mf.p, not mf.age_density(): the latter is the density p/dt, and what is wanted here is the
    MASS fraction sitting in the mf.n_abs bins the dead time blocks. At the ceiling the whole
    population is inside that window and r = 1/t_abs. """
    occ  = torch.zeros(mf.B, device=mf.device)
    rmax = torch.zeros(mf.B, device=mf.device)
    lmax = torch.zeros(mf.B, device=mf.device)
    phi  = 0.0
    for _ in range(n_probe):
        if mf.n_abs:
            occ = torch.maximum(occ, mf.p[0][:, :mf.n_abs].sum(dim=-1))
        rmax = torch.maximum(rmax, mf.rates()[:, 0])
        # the hazard rate over the ELIGIBLE bins: inside the blocked window R_abs drives the
        # field to a large negative number by construction, so including those bins would
        # report the floor rather than the burst.
        lmax = torch.maximum(lmax, mf.hazard_rate()[0][:, mf.n_abs:].max(dim=-1).values)
        phi = max(phi, float(mf.max_step_prob().max()))
        mf.update()
    return occ.cpu(), rmax.cpu(), lmax.cpu(), phi


SCAN_COLS = ("r", "ptp", "period", "sigma", "H_fwd", "H_rev", "H_fwd_rel", "H_rev_rel",
             "occ", "rate_max", "lam_max", "acf1", "phi_max", "settle")


def scan(dt: float, u_max: float, J_SCAN: torch.Tensor,
         t_equi: float = T_EQUI, repeat: bool = False) -> torch.Tensor:
    """ One (len(J_SCAN), len(SCAN_COLS)) block: the whole scan at one solver step.

    Every per-time quantity is divided by dt, so the columns are in 1/ms and nats/ms and are
    comparable between the two members of the pair -- which is the only way a Richardson
    extrapolation down the J axis means anything. """
    Q     = max(1, round(u_max / dt))
    equi  = hlp.n_steps(t_equi, dt, name="T_EQUI")
    steps = hlp.n_steps(T_MEAS, dt, name="T_MEAS")
    B_tile, chunk = hlp.epr_plan(len(J_SCAN), Q, device=DEVICE)
    tiles = torch.tensor_split(J_SCAN, max(1, -(-len(J_SCAN) // B_tile)))

    print(f"  dt = {dt:6.4f}  Q = {Q:5d}  {equi} + {steps} steps  "
          f"{len(tiles)} tile(s)", flush=True)
    rows = []
    for k, Js in enumerate(tiles):
        mf = RDMIsingModelBatch(**MODEL, J=Js, dt=dt, hazard=HAZARD, Qm=[Q], device=DEVICE)
        mf.forward(equi, pb=False)
        traj = mf.entropy_trajectory(steps, pb=True, chunk=chunk, inplace=True)

        trunc = float(mf.truncation_error().max())
        assert trunc < TOL, (f"truncation_error = {trunc:.2e} at dt = {dt}, Q = {Q}: the "
                             f"window fitted at dt = {DT_FIT} does not hold at this step.")

        r = traj["r_tot"]                                   # (B, T), 1/ms
        blk = []
        for b in range(len(Js)):
            a, z, n_cyc = hlp.cycle_window(r[b])
            per = (z - a) * dt / n_cyc if n_cyc else float('nan')
            # lag-1 autocorrelation of the rate over the averaged window: -1 is the whole
            # population firing in alternate bins, which is a step artifact and not a cycle.
            x = r[b, a:z] - r[b, a:z].mean()
            acf1 = float((x[:-1] * x[1:]).sum() / x.pow(2).sum().clamp_min(1e-300))
            m = traj["m_tot"][b, a:z]
            # H relative to the Poisson reference: the finite half of the pair. sigma is
            # untouched by the subtraction, both terms losing the same H_ref.
            href = epr.poisson_reference_entropy(m, PARAMS["nu0"], dt)
            blk.append([float(r[b, a:z].mean()),
                        float(r[b, a:z].max() - r[b, a:z].min()),
                        per,
                        float(traj["sigma_tot"][b, a:z].mean()) / dt,
                        float(traj["H_fwd_tot"][b, a:z].mean()) / dt,
                        float(traj["H_rev_tot"][b, a:z].mean()) / dt,
                        float((traj["H_fwd_tot"][b, a:z] - href).mean()) / dt,
                        float((traj["H_rev_tot"][b, a:z] - href).mean()) / dt,
                        0.0, 0.0, 0.0, acf1, 0.0, 1.0])

        # The ceiling probe, over one period of the SLOWEST cell in the tile, so every cell in
        # it is covered: a peak is one instant, and the mean of a relaxation oscillator is
        # nowhere near it. 200 ms where no cell in the tile has a period (all settled).
        periods = [b[SCAN_COLS.index("period")] for b in blk]
        slowest = max((p for p in periods if p == p), default=200.0)
        occ, rmax, lmax, phi = occupancy(mf, max(1, hlp.n_steps(slowest, dt)))
        for b in range(len(Js)):
            blk[b][SCAN_COLS.index("occ")] = float(occ[b])
            blk[b][SCAN_COLS.index("rate_max")] = float(rmax[b])
            blk[b][SCAN_COLS.index("lam_max")] = float(lmax[b])
            blk[b][SCAN_COLS.index("phi_max")] = phi

        # The equilibration test: run the SAME model another t_equi and re-measure the
        # amplitude. `settle` is the ratio. On a settled orbit it is 1; while the orbit is
        # still growing into (or decaying onto) the attractor it is not, and the lever arm is
        # the whole equilibration rather than half a record. Only done when asked for -- the
        # far-field scan is far from onset and does not need to pay for it.
        if repeat:
            mf.forward(equi, pb=False)
            t2 = mf.entropy_trajectory(steps, pb=True, chunk=chunk, inplace=True)
            r2 = t2["r_tot"]
            for b in range(len(Js)):
                # Both amplitudes over the WHOLE record, not the cycle-aligned window: an
                # orbit that has not settled has no well-defined period to align to, so
                # comparing an aligned window against a whole one would mix the thing being
                # tested with the window it is tested in.
                a1 = float(r[b].max() - r[b].min())
                a2 = float(r2[b].max() - r2[b].min())
                blk[b][SCAN_COLS.index("settle")] = a2 / a1 if a1 > 0 else float('nan')
            del t2, r2
        rows += blk
        del mf, traj
        if DEVICE == 'cuda': torch.cuda.empty_cache()
    return torch.tensor(rows)


if run_scan and (overwrite or not CACHE_SCAN.exists()):
    u_max = (torch.load(CACHE_HOPF, weights_only=False)["u_max"]
             if CACHE_HOPF.exists() else age_window())
    print(f"\n[scan] J in [{J_SCAN[0]:g}, {J_SCAN[-1]:g}], {len(J_SCAN)} points, "
          f"U_max = {u_max:.1f} ms, on {DEVICE}")
    t0 = time.time()
    res = {"u_max": u_max, "J": J_SCAN, "J_onset": J_ONSET, "cols": SCAN_COLS,
           **{dt: scan(dt, u_max, J_SCAN) for dt in DT_PAIR},
           **{("onset", dt): scan(dt, u_max, J_ONSET, t_equi=T_EQUI_ONSET, repeat=True)
              for dt in DT_PAIR}}
    print(f"  done in {time.time() - t0:.0f}s")
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(res, CACHE_SCAN)
    print(f"  saved {CACHE_SCAN}")


# ── 3. Monotone convergence at the candidate cycle point ──────────────────────

CAND_J = 17.0     # the candidate; the scan says whether it is the right one


def measure_point(J: float, dt: float, u_max: float) -> list[float]:
    """ (r, sigma/dt, period, ptp) for one system at one step, averaged over whole cycles. """
    Q = max(1, round(u_max / dt))
    net = RDMIsingModel(J, **MODEL, dt=dt, hazard=HAZARD, Qm=[Q], device="cpu")
    net.forward(hlp.n_steps(T_EQUI, dt))
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, dt), pb=False)
    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    return [float(r[a:z].mean()),
            float(traj["sigma_tot"][a:z].mean()) / dt,
            (z - a) * dt / n_cyc if n_cyc else float('nan'),
            float(r[a:z].max() - r[a:z].min())]


if run_mono and (overwrite or not CACHE_MONO.exists()):
    u_max = (torch.load(CACHE_HOPF, weights_only=False)["u_max"]
             if CACHE_HOPF.exists() else age_window())
    print(f"\n[mono] J = {CAND_J:g}, dt = {DT_MONO}, U_max = {u_max:.1f} ms")
    rows = []
    for dt in DT_MONO:
        t0 = time.time()
        rows.append(measure_point(CAND_J, dt, u_max))
        print(f"  dt = {dt:6.4f}  r = {rows[-1][0]:.8f}  sigma/dt = {rows[-1][1]:11.4e}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save({"J": CAND_J, "dts": list(DT_MONO), "cols": ("r", "sigma", "period", "ptp"),
                "rows": torch.tensor(rows), "u_max": u_max}, CACHE_MONO)
    print(f"  saved {CACHE_MONO}")


# ── Analysis ──────────────────────────────────────────────────────────────────

if run_plot:
    for f in (CACHE_HOPF, CACHE_SCAN, CACHE_MONO):
        if not f.exists(): raise FileNotFoundError(f"{f} not found. Run the stages first.")

    H = torch.load(CACHE_HOPF, weights_only=False)
    S = torch.load(CACHE_SCAN, weights_only=False)
    M = torch.load(CACHE_MONO, weights_only=False)
    col = {k: i for i, k in enumerate(S["cols"])}
    J   = S["J"]
    dt1, dt2 = DT_PAIR

    # 1. the Hopf, extrapolated
    jc = torch.tensor([H["J_c"][dt] for dt in DTS])
    jc_lim, p = richardson(jc)
    print(f"\n[1] Hopf, rho(J) = 1 at S1 under {HAZARD}")
    for dt, v in zip(DTS, jc): print(f"    dt = {dt:6.4f}   J_c = {float(v):.6f}")
    print(f"    observed order over the triple: {['%.2f' % o for o in orders(jc).tolist()]}")
    if jc_lim != jc_lim:
        print("    NO LIMIT: richardson refused -- the triple is not in its asymptotic regime.")
    else:
        print(f"    Richardson from ({dt1:g}, {dt2:g}), order {p:.2f}:  J_c -> {jc_lim:.4f}")
        print(f"    error bar: |J_c(dt/2) - limit| = {abs(float(jc[-1]) - jc_lim):.4f}")
    print(f"    against the 14.4-14.6 expected under the exact solver, and 12.0588 "
          f"with the legacy sigmoid kernel.")

    # 2. the scan, Richardson'd per J at first order (a bare pair has no order to read)
    lim = {}
    for key in ("r", "ptp", "sigma", "period", "H_fwd_rel", "H_rev_rel", "occ", "rate_max"):
        a, b = S[dt1][:, col[key]], S[dt2][:, col[key]]
        lim[key] = b + (b - a)          # first order: x(0) = 2*x(dt/2) - x(dt)

    # the cycle, by amplitude, on the extrapolated column
    amp = lim["ptp"]
    alive = amp > 1e-6 * amp.max()
    print(f"\n[2] The cycle over J in [{float(J[0]):g}, {float(J[-1]):g}], "
          f"Richardson from ({dt1:g}, {dt2:g}) at first order")
    print(f"    amplitude first resolvable at J = {float(J[alive][0]) if alive.any() else float('nan'):g}")
    print(f"    amplitude at J = 20: {float(amp[-1]):.6g} /ms   "
          f"({'ALIVE' if bool(alive[-1]) else 'DEAD -- the cycle does not survive'})")
    # the route-A re-stabilization question: is there a J above onset where the cycle dies
    # again? Read as a gap in `alive` after its first True.
    if alive.any():
        first = int(alive.nonzero()[0])
        gap = (~alive[first:]).nonzero()
        if len(gap):
            print(f"    RE-STABILIZES at J = {float(J[first + int(gap[0])]):g} "
                  f"-- the cycle dies again inside the scan")
        else:
            print(f"    no re-stabilization in [{float(J[0]):g}, {float(J[-1]):g}]: the cycle "
                  f"survives to the top of the scan")
    else:
        print("    NO cycle anywhere in the scan -- check J_c against the bracket")

    print(f"\n    {'J':>6}{'r':>12}{'ptp':>12}{'period':>10}{'sigma/dt':>13}"
          f"{'occ':>10}{'r_max*t_abs':>13}{'acf1':>9}")
    for i in range(0, len(J), 5):
        print(f"    {float(J[i]):6.2f}{float(lim['r'][i]):12.6f}{float(lim['ptp'][i]):12.6f}"
              f"{float(lim['period'][i]):10.2f}{float(lim['sigma'][i]):13.4e}"
              f"{float(lim['occ'][i]):10.4f}"
              f"{float(lim['rate_max'][i]) * PARAMS['t_abs']:13.4f}"
              f"{float(S[dt2][i, col['acf1']]):9.3f}")

    # 3. sigma against (J - J_c): the first look at the thermodynamic half
    def loglog_slope(x, y):
        """ Least-squares slope of ln y against ln x: the exponent of a power law. """
        lx, ly = torch.log(x), torch.log(y.clamp_min(1e-300))
        return float(((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum())

    print(f"\n[3] EPR across the Hopf -- FIRST LOOK, not a converged measurement")
    print(f"    The test is sigma ~ (J - J_c)^1, i.e. sigma linear in the squared amplitude,")
    print(f"    which is the normal-form statement and holds only CLOSE TO ONSET.")

    JO  = S["J_onset"]
    lim_o = {}
    for key in ("sigma", "ptp"):
        a, b = S[("onset", dt1)][:, col[key]], S[("onset", dt2)][:, col[key]]
        lim_o[key] = b + (b - a)
    xo = JO - J_C_MEASURED

    # Settled or not. Critical slowing down means the points closest to J_c have NOT reached
    # the attractor even after T_EQUI_ONSET, and fitting through them measures the transient.
    set_o = S[("onset", dt2)][:, col["settle"]]
    ok_o  = (set_o - 1.0).abs() < SETTLE_TOL
    first = float(xo[ok_o][0]) if bool(ok_o.any()) else float('nan')
    print(f"\n    near-onset window, J_c + {float(xo[0]):.4g} to J_c + {float(xo[-1]):.4g} "
          f"({len(JO)} points, geometric), equilibrated {T_EQUI_ONSET:g} ms:")
    print(f"      settled (|amp(2x equi)/amp(1x equi) - 1| < {SETTLE_TOL:g}): "
          f"{int(ok_o.sum())} of {len(JO)}, from J - J_c = {first:.4g} upward")
    if int(ok_o.sum()) < len(JO):
        print(f"      *** the rest are NOT settled: just above a Hopf the approach rate goes")
        print(f"      *** to zero with J - J_c, so they report a residual transient, not a")
        print(f"      *** cycle. They are excluded from every fit below.")
    for lo, hi, tag in ((0.0, 0.3, "J - J_c < 0.3"), (0.0, 1.01, "the whole settled window")):
        m = ok_o & (xo > lo) & (xo < hi) & (lim_o["sigma"] > 0)
        if int(m.sum()) >= 3:
            print(f"      sigma ~ (J - J_c)^{loglog_slope(xo[m], lim_o['sigma'][m]):.3f}   "
                  f"ptp ~ (J - J_c)^{loglog_slope(xo[m], lim_o['ptp'][m]):.3f}   "
                  f"[{tag}, {int(m.sum())} points]")
    m = ok_o & (lim_o["sigma"] > 0)
    if int(m.sum()) >= 3:
        print(f"      sigma ~ ptp^{loglog_slope(lim_o['ptp'][m], lim_o['sigma'][m]):.3f}  "
              f"(needs no J_c)")
    print(f"      (the Hopf normal form predicts ptp ~ (J - J_c)^0.5 and sigma ~ ptp^2)")

    if jc_lim == jc_lim:
        x = J - jc_lim
        m = (x > 0) & (lim["sigma"] > 0)
        print(f"\n    far field, the J in [{float(J[0]):g}, {float(J[-1]):g}] scan "
              f"(J - J_c from {float(x[m][0]):.3g} to {float(x[m][-1]):.3g}):")
        print(f"      sigma ~ (J - J_c)^{loglog_slope(x[m], lim['sigma'][m]):.3f}   "
              f"ptp ~ (J - J_c)^{loglog_slope(x[m], amp[m]):.3f}   "
              f"[{int(m.sum())} points] -- a GLOBAL exponent, not the critical one")
        a2 = amp[m] ** 2
        print(f"      sigma ~ (peak-to-peak^2)^{loglog_slope(a2, lim['sigma'][m]):.3f}  "
              f"(needs no J_c)")

    # 4. monotone convergence at the candidate
    print(f"\n[4] Monotone convergence at J = {M['J']:g}, dt = {M['dts']}")
    mc = {k: i for i, k in enumerate(M["cols"])}
    asymptotic: dict = {}
    for key in ("r", "sigma", "period", "ptp"):
        v = M["rows"][:, mc[key]]
        d = v.diff()
        mono = bool(((d[:-1] * d[1:]) > 0).all())
        o = orders(v)
        print(f"    {key:>7}: {[f'{float(x):.8g}' for x in v]}")
        print(f"    {'':>7}  diffs {['%.3e' % x for x in d.tolist()]}  "
              f"orders {['%.2f' % x for x in o.tolist()]}  -> "
              f"{'MONOTONE' if mono else 'NOT MONOTONE -- B1 cannot use this point'}")
        l, pp = richardson(v)
        asymptotic[key] = (l == l)
        print(f"    {'':>7}  richardson: {l:.8g} (order {pp:.2f})"
              if l == l else f"    {'':>7}  richardson REFUSED")

    # 5. the step diagnostics
    print(f"\n[5] Step diagnostics along the scan, at dt = {dt2:g}")
    print(f"    max over J of max_step_prob: {float(S[dt2][:, col['phi_max']].max()):.3e}")
    print(f"    min lag-1 autocorrelation:   {float(S[dt2][:, col['acf1']].min()):.4f}  "
          f"(-1 would be the alternate-bin artifact)")
    print(f"    max ceiling occupancy:       {float(S[dt2][:, col['occ']].max()):.4f}  "
          f"(mass in the {krn.n_abs_bins(PARAMS['t_abs'], dt2)} blocked bins)")
    print(f"    max rate * t_abs:            {float(S[dt2][:, col['rate_max']].max()) * PARAMS['t_abs']:.4f}  "
          f"(1.0 would be the dead-time ceiling)")
    lam = float(S[dt2][:, col['lam_max']].max())
    print(f"    max HAZARD rate lam over the scan: {lam:.4g} /ms  "
          f"(the population rate above is what the dead time caps; lam is not capped)")
    print(f"    lambda*dt/2 at that peak rate:      {hzd.step_error(lam, dt2):.3e}  "
          f"-- hazards.step_error, the local relative error of the exact-survival map")

    # 6. the H control: raw H against the Poisson-referenced one
    print(f"\n[6] H control, at J = {CAND_J:g}")
    i = int((J - CAND_J).abs().argmin())
    for key in ("H_fwd", "H_rev"):
        raw = (float(S[dt1][i, col[key]]), float(S[dt2][i, col[key]]))
        rel = (float(S[dt1][i, col[key + "_rel"]]), float(S[dt2][i, col[key + "_rel"]]))
        r_lim = float(lim["r"][i])
        print(f"    {key}: raw {raw[0]:.5f} -> {raw[1]:.5f}, increment {raw[1] - raw[0]:+.5f} "
              f"(predicts r ln 2 = {r_lim * 0.6931471805599453:+.5f})")
        print(f"    {'':>{len(key)}}  referenced {rel[0]:.5f} -> {rel[1]:.5f}, "
              f"increment {rel[1] - rel[0]:+.5f}")
    s1 = float(S[dt1][i, col["H_rev"]] - S[dt1][i, col["H_fwd"]])
    s2 = float(S[dt1][i, col["H_rev_rel"]] - S[dt1][i, col["H_fwd_rel"]])
    print(f"    sigma from raw H: {s1:.8e};  from referenced H: {s2:.8e};  "
          f"difference {s1 - s2:.2e} (must be 0)")


# ── Plots ─────────────────────────────────────────────────────────────────────

if run_plot:
    hlp.pub_style()
    banner = pre.banner(PRESET, dict(PARAMS, dt=DT_PAIR[1]))

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.set_label('cycle-point')
    (ax_a, ax_s), (ax_c, ax_o) = axs

    for dt, ls in zip(DT_PAIR, (':', '--')):
        ax_a.plot(J, S[dt][:, col["ptp"]], ls, color='0.5', linewidth=1,
                  label=f'$\\Delta_t = {dt:g}$')
    ax_a.plot(J, amp, '-', color='C0', label='Richardson limit')
    if jc_lim == jc_lim:
        ax_a.axvline(jc_lim, color='C3', linewidth=1,
                     label=f'$J_c = {jc_lim:.3f}$ ($\\rho = 1$)')
    ax_a.axvline(CAND_J, color='C2', linestyle='-.', linewidth=1,
                 label=f'candidate $J = {CAND_J:g}$')
    ax_a.set_ylabel('peak-to-peak of $r$ [1/ms]')
    # The amplitude panel must not quietly show a limit the convergence check refused. If ptp
    # is not in its asymptotic regime at the candidate point, the extrapolated curve is a
    # first-order formula applied outside the regime it is valid in, and the title says so.
    ok_ptp = asymptotic.get("ptp", True)
    ax_a.set_title('Cycle amplitude' if ok_ptp else
                   'Cycle amplitude  —  limit NOT converged (see [4])')
    if not ok_ptp:
        ax_a.text(0.03, 0.95, 'ptp is not in its asymptotic regime:\n'
                              'the extrapolated curve is indicative only',
                  transform=ax_a.transAxes, va='top', fontsize=7.5, color='C3')
    ax_a.legend(fontsize=8, loc='lower right')

    ax_s.plot(J, lim["sigma"], '-', color='C0')
    ax_s.set_yscale('log')
    if jc_lim == jc_lim: ax_s.axvline(jc_lim, color='C3', linewidth=1)
    ax_s.axvline(CAND_J, color='C2', linestyle='-.', linewidth=1)
    ax_s.set_ylabel(r'$\sigma/\Delta_t$ [nats/ms]')
    ax_s.set_title('Entropy production rate (first look)')

    ax_c.plot(J, lim["period"], '-', color='C0')
    ax_c.set_ylabel('period [ms]'); ax_c.set_title('Cycle period')

    ax_o.plot(J, lim["occ"], '-', color='C0', label='mass in the blocked bins')
    ax_o.plot(J, lim["rate_max"] * PARAMS["t_abs"], '-', color='C1',
              label=r'$\max(r)\,t_{\rm abs}$')
    ax_o.axhline(1.0, color='k', linewidth=0.8, linestyle='--', label='dead-time ceiling')
    ax_o.set_ylabel('ceiling occupancy'); ax_o.set_title('How close to the 1/$t_{abs}$ cap')
    ax_o.legend(fontsize=8)

    for ax in axs.flatten():
        ax.set_xlabel('Recurrent coupling  $J$'); ax.grid(alpha=0.3)
    fig.suptitle(f'S1 cycle point and Hopf location\n{banner}')
    fig.tight_layout()

    # sigma against distance to onset, on log-log: the first look at the exponent
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig2.set_label('sigma-vs-onset')
    # left: sigma against distance to onset, both windows on one axis
    axL.loglog(xo[ok_o], lim_o["sigma"][ok_o].clamp_min(1e-300), 'o', markersize=3.5,
               color='C0', label='near-onset, settled')
    if int((~ok_o).sum()):
        axL.loglog(xo[~ok_o], lim_o["sigma"][~ok_o].clamp_min(1e-300), 'x', markersize=4,
                   color='0.6', label='near-onset, NOT settled')
    if jc_lim == jc_lim:
        x = J - jc_lim
        m = x > 0
        axL.loglog(x[m], lim["sigma"][m].clamp_min(1e-300), 's', markersize=3.5, color='C1',
                   markerfacecolor='none', label='far field (J = 12-20)')
    k = float(lim_o["sigma"][-1]) / float(xo[-1])
    xs = torch.tensor([float(xo[0]), 10.0])
    axL.loglog(xs, k * xs, 'k--', linewidth=0.8, label=r'slope 1 ($\propto J - J_c$)')
    axL.set_xlabel(r'$J - J_c$'); axL.set_ylabel(r'$\sigma/\Delta_t$ [nats/ms]')
    axL.set_title('EPR past onset -- FIRST LOOK, not converged')
    axL.legend(fontsize=8); axL.grid(alpha=0.3, which='both')
    # right: sigma against the squared amplitude, which needs no J_c
    axR.loglog(lim_o["ptp"][ok_o].clamp_min(1e-300) ** 2,
               lim_o["sigma"][ok_o].clamp_min(1e-300), 'o',
               markersize=3.5, color='C0', label='near-onset, settled')
    axR.loglog(amp.clamp_min(1e-300) ** 2, lim["sigma"].clamp_min(1e-300), 's', markersize=3.5,
               color='C1', markerfacecolor='none', label='far field')
    axR.set_xlabel(r'(peak-to-peak of $r$)$^2$'); axR.set_ylabel(r'$\sigma/\Delta_t$ [nats/ms]')
    axR.set_title(r'$\sigma$ against amplitude$^2$ (no $J_c$ needed)')
    axR.legend(fontsize=8); axR.grid(alpha=0.3, which='both')
    fig2.suptitle(banner, fontsize=8)
    fig2.tight_layout()

    rp.show()
