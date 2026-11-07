"""
Convergence of the microscopic model to the mean field in the system size N.

The mean field IS the N -> infinity limit, so the question is not how well the spin network
agrees but how fast, and what it sees that the mean field cannot. Two operating points, as in
S1-dt_conv.py: J = 10, a fixed point where the mean field is a flat line with sigma = 0, and
J = 15, the limit cycle. dt is SHARED by both models and held fixed, so the discretization error
is common and cancels -- which is what makes a 1e-5 agreement meaningful at a step whose own
error is 1e-2.

Three quantities, converging differently. The mean carries no resolvable bias. The variance
obeys Var_N = Var_MF + c/N with the same c either side of the Hopf. The spectral linewidth
separates the regimes outright: at the fixed point the peak is a QUASI-CYCLE, noise ringing a
damped mode, so its width is the damping rate and does not move with N, while on the cycle it is
phase diffusion and falls as 1/N. Coherence that improves with N is an oscillation; coherence
that does not is noise with a resonance -- a distinction the mean field cannot draw at all.

Three things that look like over-complications until they are removed:

  the autocorrelation is UNBIASED, divided by the overlap count (n - k). The biased form's
      triangular taper is an exponential decay at rate 1/(n*dt) -- 4e-5 /ms here -- which once
      swallowed the entire 1/N result. See `coherence`.
  the linewidth is measured TWICE. The envelope fit dies at the fixed point and the spectral
      width dies on the cycle, so both are carried with their own floor and the regime decides.
  the mean is quoted WITH its floors. A sample's mean cannot be cycle-aligned, so it carries a
      phase error of amplitude/n_periods, larger here than any finite-N shift -- and the seed
      spread cannot see it, both seeds making the same error.

Results, with the tables and figures this prints: see the companion write-up.
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.helpers as hlp
import rdme.plots as rp
import rdme.presets as pre
from rdme.rdm_single import RDMIsingModel
from rdme.spin_model import SpinIsingModel


# ── Parameters ────────────────────────────────────────────────────────────────

# Shared by both models, so a regime is one dict and one J. dt is a SOLVER setting here, not
# the object of study: fixed at 0.25 ms, where the rate is converged to ~1% (see S1-dt_conv.py)
# and, more to the point, where both models make the same error -- which is what lets a 1e-5
# agreement mean something at a step whose own error is 1e-2.
PRESET = "S1"
HAZARD = "poisson_exact"   # stated per script: the hazard is its own axis, not part of a
                           # preset. The pre-port dict said "async" (route A, = "bernoulli"
                           # here). The mean field and the spin model take the SAME key, which
                           # is the whole point of this comparison -- never mix the two.

# S1 leaves J out -- it is the axis the system is studied along -- so PARAMS is exactly what
# both constructors take, and REGIMES supplies the J.
PARAMS: dict = pre.params(PRESET, hazard=HAZARD, dt=0.25)

# The cycle regime is still at the route-A J = 15. Under route B the Hopf moved to an expected
# J ~ 14.4-14.6 (S1-cycle_point.py measures it), so J = 15 may sit barely past onset where the
# amplitude goes like sqrt(J - Jc) and the linewidth result this script turns on is read in the
# worst possible place. Re-point it at POINTS["S1/cycle"] once that is set.
REGIMES: dict[str, float] = {"fixed point": 10.0, "limit cycle": 15.0}

# Four decades. The spin step is O(N) but launch-latency bound well past 1e5 -- 1e6 neurons cost
# 1.5x the time of 1e5 and 0.09 GiB -- so the top of this range is nearly free and is what pins
# the asymptote. Half-decade spacing, so a power law has 8 points to show itself on.
NS = [100, 316, 1000, 3162, 10000, 31623, 100000, 1000000]

# The EPR arm. Capped by memory: the chunked estimator holds ~8 buffers of (chunk + overlap, N)
# in float32, which is 2 GiB at N = 1e4 and would be 20 at 1e6.
NS_EPR = [100, 316, 1000, 3162, 10000, 31623]

SEEDS = (0, 1)       # repeats per (regime, N), for an error bar on every point

EQUI_MS = 4000.0     # verified in the dt study: the attractor is reached well inside this
REC_MS  = 30000.0    # ~420 periods, which is what the spectral width needs, not the mean
EPR_MS  = 16000.0    # ~26 s per 8000 ms, independent of N

# The reverse recursion resets at spikes, so a neuron that has not fired inside the window tail
# carries the zero boundary condition. At r ~ 0.013 /ms the mean ISI is ~77 ms and the LONGEST
# of N draws is ln(N) of those -- ~700 ms at N = 1e4 -- so the tail is sized on that, not on a
# few tau. check_overlap=True in the call says when it is not enough.
EPR_CHUNK, EPR_OVERLAP = 4096, 4096          # steps, i.e. 1024 ms each at dt = 0.25

SEG_MS = 7500.0      # Welch segment: ~105 periods, so the peak is resolved to f0/105
TOL    = 1e-5        # mean-field age grid; tight, since it is the reference everything uses

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COLS = ("r", "var", "period", "decay", "f0", "peak", "fwhm")


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE     = hlp.cache_path(STEM, dict(PARAMS, preset=PRESET, regimes=REGIMES, ns=NS, ns_epr=NS_EPR,
                                      seeds=list(SEEDS), equi=EQUI_MS, rec=REC_MS,
                                      epr=EPR_MS, seg=SEG_MS, tol=TOL),
                           root=CACHE_DIR)

run_sim   = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE
torch.set_default_dtype(torch.float64)


def checkpoint(res: dict) -> None:
    """ Write the cache, atomically, after every unit of work.

    Not at the end of the run, which is the obvious place and the wrong one. This sweep is
    half an hour of independent units, and a single save at the end means an interrupted run
    leaves nothing, a run in progress is invisible to anyone else, and the first thing anyone
    does on seeing no cache is start a SECOND copy competing for the same GPU. Saving per unit
    makes the file usable while it fills and makes the run resumable: `res` is reloaded below
    and finished units are skipped.

    Through a temporary and a rename, because the reader may be the person watching it fill:
    `replace` is atomic within a filesystem, so a loader sees either the previous version or
    the new one, never the half-written middle. """
    CACHE_DIR.mkdir(exist_ok=True)
    # Fold in whatever is on disk before writing. Two runs of this script -- mine and yours --
    # each hold their own `res` in memory, so a plain save makes the last writer win and
    # silently discard every unit the other measured. setdefault keeps what THIS process just
    # computed and adopts any unit it does not have, which makes concurrent runs additive
    # instead of destructive: they divide the work by racing for it rather than redoing it.
    if CACHE.exists():
        try:
            disk = torch.load(CACHE, weights_only=False)
        except Exception:
            disk = {}                      # a torn read from another writer: ours still stands
        adopted = [k for k in disk if k not in res]
        for k in adopted:
            res[k] = disk[k]
        if adopted:
            print(f"  (+{len(adopted)} units from another run of this script)")
    tmp = CACHE.with_suffix(".tmp")
    torch.save(res, tmp)
    tmp.replace(CACHE)


# ── Signal measures ───────────────────────────────────────────────────────────
#
# Three estimators, all reading the same population-rate trace. They are here rather than in a
# module because each one encodes a choice about THIS signal -- how long a lag is worth looking
# at, how many periods a segment should hold -- and those choices are the measurement.

def psd(x: torch.Tensor, dt: float, seg_ms: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ One-sided power spectral density by Welch's method, in (1/ms)^2 per (1/ms).

    Segmented and averaged rather than one long periodogram: a periodogram's variance does not
    fall with record length (each bin keeps ~100% error however long the run), while averaging K
    segments divides it by K. The cost is resolution, which is why the segment is set in
    PERIODS -- 105 of them -- rather than in samples. Hann-windowed and mean-subtracted per
    segment, so a slow drift between segments does not leak into the peak. """
    n_seg = int(seg_ms / dt)
    step  = n_seg // 2                                    # 50% overlap, the Welch default
    win   = torch.hann_window(n_seg, dtype=x.dtype)
    norm  = win.pow(2).sum() * dt                         # keeps the density calibrated
    acc, k = torch.zeros(n_seg // 2 + 1, dtype=x.dtype), 0
    for i in range(0, len(x) - n_seg + 1, step):
        seg = (x[i:i + n_seg] - x[i:i + n_seg].mean()) * win
        acc += torch.fft.rfft(seg).abs().pow(2)
        k += 1
    return torch.fft.rfftfreq(n_seg, d=dt), 2.0 * acc / (k * norm)


def peak_stats(f: torch.Tensor, S: torch.Tensor) -> tuple[float, float, float]:
    """ (f0, height, FWHM) of the spectral peak, by linear interpolation of the half-power
    crossings on either side of the maximum.

    The FWHM is the quantity the whole spectral argument rests on: it is a damping rate at a
    fixed point and a phase-diffusion rate on a cycle, and only its scaling in N tells the two
    apart. Returns nan for the width when the peak does not come back down inside the band,
    which is the honest answer rather than a half-width read off the edge. """
    i = int(S.argmax())
    if i == 0 or i == len(S) - 1:
        return float(f[i]), float(S[i]), float('nan')
    half = S[i] / 2

    def cross(lo: int, hi: int, step: int) -> float:
        j = lo
        while 0 <= j + step < len(S) and S[j + step] > half:
            j += step
        if not 0 <= j + step < len(S):
            return float('nan')
        # linear in S between the bracketing bins
        a, b = float(S[j]), float(S[j + step])
        t = (float(half) - a) / (b - a) if b != a else 0.0
        return float(f[j]) + t * (float(f[j + step]) - float(f[j]))

    lo, hi = cross(i, 0, -1), cross(i, 0, +1)
    return float(f[i]), float(S[i]), hi - lo


def coherence(x: torch.Tensor, dt: float, max_lag_ms: float) -> tuple[float, float]:
    """ (period, decay rate) from the envelope of the autocorrelation, in ms and 1/ms.

    C(tau) of a noisy oscillator is ~ exp(-D*tau) * cos(2*pi*f0*tau), so the successive peaks of
    C fall on a straight line in log, and their slope is D. What D means differs by regime and
    that is exactly the measurement: at a fixed point it is the DAMPING of the stable focus, set
    by the deterministic spectrum and independent of N; on a limit cycle the deterministic decay
    is zero and what is left is PHASE DIFFUSION, which is caused by the finite-size noise and so
    falls as 1/N.

    Peaks are taken after the first zero crossing (lag 1 is the trivial maximum of any smooth
    signal) and only while the envelope is above 5% of C(0), below which the estimate is
    measuring the noise floor of the autocorrelation itself. """
    y = x - x.mean()
    n = len(y)
    L = min(int(max_lag_ms / dt), n - 1)
    F = torch.fft.rfft(y, 2 * n)
    C = torch.fft.irfft(F * F.conj(), 2 * n)[:L]
    # UNBIASED: divide by the number of overlapping pairs at each lag. The FFT gives the raw
    # sum, whose expectation carries a factor (1 - k/n) -- the triangular taper of the biased
    # estimator -- and that taper is itself an exponential-looking decay at rate 1/(n*dt).
    # Measured: a PURE sinusoid, true D = 0, reads D = 4.03e-5 /ms through the biased form at
    # this record length, against 1/T = 3.33e-5. That is not a small correction here, it is the
    # floor the whole measurement was sitting on: the cycle's apparent decay saturated at
    # 4.04e-5 for every N above 3162, which was the estimator measuring its own window.
    C = C / (n - torch.arange(L, device=C.device))
    C = C / C[0].clamp_min(1e-300)

    neg = (C < 0).nonzero()
    if len(neg) == 0:
        return float('nan'), float('nan')
    start = int(neg[0])
    mid   = C[start:]
    pk    = ((mid[1:-1] > mid[:-2]) & (mid[1:-1] > mid[2:])).nonzero().flatten() + start + 1
    pk    = pk[C[pk] > 0.05]
    if len(pk) < 2:
        return float('nan'), float('nan')

    lags   = pk.to(C.dtype) * dt
    period = float(lags.diff().mean()) if len(pk) > 1 else float('nan')
    # least squares of log C against lag: the envelope's slope, with no fit of f0 involved
    lg = torch.log(C[pk])
    A  = torch.stack([lags, torch.ones_like(lags)], dim=1)
    slope = float(torch.linalg.lstsq(A, lg.unsqueeze(-1)).solution[0])
    return period, -slope


def summarize(r: torch.Tensor, dt: float) -> list[float]:
    """ The row every run contributes: the four scalars plus the peak triple. """
    period, decay = coherence(r, dt, max_lag_ms=REC_MS / 3)
    f, S = psd(r, dt, SEG_MS)
    f0, height, fwhm = peak_stats(f, S)
    return [float(r.mean()), float(r.var()), period, decay, f0, height, fwhm]


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    # Resume: the hash pins the parameters, so an existing file is this run, finished or not.
    res: dict = ({"params": PARAMS, "regimes": REGIMES, "ns": NS, "ns_epr": NS_EPR,
                  "seeds": list(SEEDS)} if overwrite or not CACHE.exists()
                 else torch.load(CACHE, weights_only=False))
    done = sum(1 for k in res if isinstance(k, tuple))
    if done:
        print(f"Resuming {CACHE.name}: {done} units already measured.")
    dt = PARAMS["dt"]
    t_all = time.time()

    for name, J in REGIMES.items():

        # --- the reference ----------------------------------------------------
        # Fitted age grid, as every mean-field measurement in this repo does; the spin network
        # carries an age per neuron and has no grid, which is the one thing the two models do
        # not share.
        if (name, "mf") in res:
            print(f"\n[{name}, J = {J:g}] mean field: cached, "
                  f"r = {float(res[(name, 'mf')][0]):.6f}")
        else:
            print(f"\n[{name}, J = {J:g}] mean field ...")
            Qm = hlp.fit_age_grid(lambda q: RDMIsingModel(J, **PARAMS, Qm=q, device="cpu"),
                                  tol=TOL, equi=int((EQUI_MS + REC_MS) / dt))
            mf = RDMIsingModel(J, **PARAMS, Qm=Qm, device="cpu")
            mf.forward(int(EQUI_MS / dt))
            mf_r = mf.trajectory(int(REC_MS / dt), pb=False)["r"][:, 0]

            # the mean field is deterministic, so its average is taken over whole cycles; the
            # samples below cannot be (a noisy trace crosses its mean many times a period) and
            # get a plain mean over ~420 periods, where the residual phase error is far under
            # their own sampling error
            a, b, n_cyc = hlp.cycle_window(mf_r)
            mf_row = summarize(mf_r, dt)
            mf_row[0] = float(mf_r[a:b].mean())
            mf_row[1] = float(mf_r[a:b].var())

            mf_ep = mf.entropy_trajectory(int(EPR_MS / dt), pb=False)
            c, d, _ = hlp.cycle_window(mf_ep["r_tot"])
            mf_sigma = float(mf_ep["sigma_tot"][c:d].mean()) / dt

            res[(name, "mf")] = torch.tensor(mf_row)
            res[(name, "mf_sigma")] = mf_sigma
            res[(name, "mf_traj")] = mf_r.float().clone()
            res[(name, "mf_psd")] = torch.stack(psd(mf_r, dt, SEG_MS))
            checkpoint(res)
            print(f"  Q={Qm[0]}  r={mf_row[0]:.6f}  sd={mf_row[1]**0.5:.6f}  "
                  f"period={mf_row[2]:.2f} ms  sigma/dt={mf_sigma:.4e}  ({n_cyc} cycles)")

        # --- the samples ------------------------------------------------------
        for N in NS:
            if (name, N) in res:
                print(f"  N={N:8d}  cached")
                continue
            rows, psds = [], []
            for seed in SEEDS:
                t0 = time.time()
                torch.manual_seed(seed)
                sm = SpinIsingModel(N, J, **PARAMS, device=DEVICE, ic="random")
                sm.forward(int(EQUI_MS / dt))
                r = sm.trajectory(int(REC_MS / dt))["m_tot"][:, 0].double() / dt
                rows.append(summarize(r, dt))
                psds.append(torch.stack(psd(r, dt, SEG_MS)))
                # The WHOLE trace, in float32, not the slice the figures need. Today's lesson:
                # the decay estimator was wrong and the trace it was computed from had been
                # thrown away, so a 20-minute arm had to be re-run to fix a four-line function.
                # 0.5 MiB a run buys the right to revise any of this offline.
                if seed == SEEDS[0]:
                    res[(name, N, "traj")] = r.float().clone()
                del sm
                if DEVICE == 'cuda': torch.cuda.empty_cache()
            res[(name, N)] = torch.tensor(rows)                     # (seeds, COLS)
            res[(name, N, "psd")] = torch.stack(psds).mean(dim=0)   # (2, F)
            checkpoint(res)
            m = torch.tensor(rows).mean(dim=0)
            print(f"  N={N:8d}  r={m[0]:.6f}  sd={m[1] ** 0.5:.6f}  "
                  f"excess sd={max(m[1] - mf_row[1], 0) ** 0.5:.2e}  "
                  f"decay={m[3]:.2e}/ms  ({time.time() - t0:.0f}s/seed)")

        # --- entropy production ----------------------------------------------
        for N in NS_EPR:
            if (name, N, "sigma") in res:
                print(f"  N={N:8d}  sigma cached")
                continue
            t0 = time.time()
            torch.manual_seed(SEEDS[0])
            sm = SpinIsingModel(N, J, **PARAMS, device=DEVICE, ic="random")
            sm.forward(int(EQUI_MS / dt))
            ep = sm.entropy_trajectory_chunked(int(EPR_MS / dt), chunk=EPR_CHUNK,
                                               overlap=EPR_OVERLAP, check_overlap=True)
            sg = ep["sigma_tot"].double() / dt
            # Blocked, not pointwise: sigma_tot[t] is a sampled log-ratio that fluctuates in
            # sign and is correlated across a burst, so a naive SEM over steps understates the
            # error by the autocorrelation time. 40 blocks of ~400 ms each are long against the
            # period and numerous enough for their own spread to mean something.
            nb = 40
            blk = sg[:len(sg) // nb * nb].reshape(nb, -1).mean(dim=1)
            res[(name, N, "sigma")] = torch.tensor([float(sg.mean()),
                                                    float(blk.std() / nb ** 0.5)])
            res[(name, N, "H_fwd")] = float(ep["H_fwd_tot"].double().mean()) / dt
            checkpoint(res)
            print(f"  N={N:8d}  sigma/dt={float(sg.mean()):+.4e} "
                  f"+- {float(blk.std() / nb ** 0.5):.1e}  ({time.time() - t0:.0f}s)")
            del sm, ep
            if DEVICE == 'cuda': torch.cuda.empty_cache()

    checkpoint(res)
    print(f"\nDone in {time.time() - t_all:.0f}s. "
          f"{CACHE} ({CACHE.stat().st_size / 2**20:.1f} MiB).")


# ── Analysis ──────────────────────────────────────────────────────────────────

def power_fit(ns: list[int], y: torch.Tensor,
              mask: torch.Tensor | None = None) -> tuple[float, float]:
    """ (exponent, prefactor) of y ~ a * N^p, by least squares in log-log. The exponent is the
    whole point: 1/N and 1/sqrt(N) are different physics, and a factor that does not move with
    N is a third thing again.

    `mask` drops points the measurement cannot resolve -- at large N the autocorrelation decays
    by less than its own noise floor over the longest lag available, and those points saturate
    rather than continue the law. Fitting through them flattens the exponent toward zero and
    would turn a measured 1/N into a false plateau. """
    ok = torch.isfinite(y) & (y > 0)
    if mask is not None:
        ok = ok & mask
    if int(ok.sum()) < 2:
        return float('nan'), float('nan')
    x = torch.log(torch.tensor([float(n) for n in ns])[ok])
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, torch.log(y[ok]).unsqueeze(-1)).solution
    return float(sol[0]), float(torch.exp(sol[1]))


if run_plot:

    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} not found. Run the simulation first.")

    print(f"\nReading {CACHE.name}.")
    res = torch.load(CACHE, weights_only=False)

    # The cache fills unit by unit and is readable while it does, so reading a PARTIAL file is
    # the normal case, not a corruption. Say what is missing and stop, rather than failing on
    # whichever KeyError happens to come first.
    missing = [k for name in REGIMES for k in
               [(name, "mf")] + [(name, N) for N in NS] + [(name, N, "sigma") for N in NS_EPR]
               if k not in res]
    if missing:
        raise SystemExit(f"{CACHE.name} holds {len(res)} entries and is still filling: "
                         f"{len(missing)} units missing, first {missing[:4]}. "
                         f"Re-run when the simulation finishes, or set run_plot = False.")
    ns, ns_epr, dt = res["ns"], res["ns_epr"], PARAMS["dt"]
    col = {k: i for i, k in enumerate(COLS)}

    def ref_period(name) -> float:
        """ The timescale a regime is read on. On the cycle it is the mean field's own period.
        At a fixed point the mean field is constant and HAS no period -- but the stochastic
        system still rings, at the damped mode's frequency, so the reference there comes from
        the samples' spectral peak. That is not a fallback: the quasi-cycle period is the
        physical timescale of the fixed point, and the mean field cannot report it. """
        p = float(res[(name, "mf")][col["period"]])
        if p == p:
            return p
        f0 = torch.tensor([float(res[(name, N)][:, col["f0"]].mean()) for N in ns])
        return 1.0 / float(f0.median())

    def stat(name, key):     # (mean, sem) over seeds, per N
        rows = torch.stack([res[(name, N)][:, col[key]] for N in ns])     # (N, seeds)
        return rows.mean(dim=1), rows.std(dim=1) / rows.shape[1] ** 0.5

    for name in REGIMES:
        mf = res[(name, "mf")]
        r_mf, var_mf = float(mf[col["r"]]), float(mf[col["var"]])
        r_n, r_se = stat(name, "r")
        v_n, _    = stat(name, "var")
        d_n, d_se = stat(name, "decay")

        print(f"\n[{name}, J = {REGIMES[name]:g}]  mean field: r = {r_mf:.8f}, "
              f"sd = {var_mf ** 0.5:.6f}, period = {float(mf[col['period']]):.2f} ms")
        print(f"    {'N':>9} {'r':>12} {'(r-r_MF)/r_MF':>15} {'sd':>10} "
              f"{'excess sd':>11} {'decay /ms':>11} {'f0 /ms':>9}")
        for j, N in enumerate(ns):
            excess = max(float(v_n[j]) - var_mf, 0.0) ** 0.5
            f0 = float(res[(name, N)][:, col["f0"]].mean())
            print(f"    {N:9d} {float(r_n[j]):12.8f} {(float(r_n[j]) - r_mf) / r_mf:15.2e} "
                  f"{float(v_n[j]) ** 0.5:10.6f} {excess:11.3e} "
                  f"{float(d_n[j]):11.3e} {f0:9.5f}")

        # the three laws, each as an exponent
        bias = (r_n - r_mf).abs()
        p_b, _ = power_fit(ns, bias)
        p_v, a_v = power_fit(ns, (v_n - var_mf).clamp_min(0))
        # the two routes to the same D, and the floor below which each is measuring its own
        # window rather than the signal: pi/SEG_MS for a Lorentzian width of one Welch bin,
        # and ~1/max_lag for an envelope that has not decayed inside the longest lag looked at
        w_n, _ = stat(name, "fwhm")
        d_psd  = torch.pi * w_n
        floor_psd, floor_acf = torch.pi / SEG_MS, 1.0 / (REC_MS / 3)
        # resolved = the seeds agree on it. A hard cut at the 1/max_lag line would throw away
        # points that are measured perfectly well: an envelope decaying at D = 0.5/max_lag
        # still falls by 40% across the window, which two seeds can agree on to a few percent.
        # Resolvable when the OSCILLATION dominates the NOISE, which is the condition the
        # envelope fit actually needs: var_MF > excess var, i.e. the deterministic swing is
        # bigger than the finite-size jitter riding on it. Below that the autocorrelation's
        # peaks are noise and the fit measures the peak-picking.
        #
        # Two cruder criteria were tried first and both were wrong, in opposite directions. A
        # 2-seed SEM test kept 3 of 8 points, which is just the SEM being unreliable at n = 2.
        # Requiring the envelope to fall 5% across the longest lag (D*max_lag > 0.05) threw out
        # everything above N = 3162 -- yet those rows' local slopes sit at -1.0 to three
        # digits, because a near-deterministic trace of 120k samples resolves a 0.06% decay
        # perfectly well. The method has far more dynamic range than that cut assumed.
        ok_acf = torch.isfinite(d_n) & (d_n > 0) & (var_mf > (v_n - var_mf).clamp_min(0))
        ok_psd = torch.isfinite(d_psd) & (d_psd > 2 * floor_psd)
        p_d, _ = power_fit(ns, d_n, ok_acf)
        p_w, _ = power_fit(ns, d_psd, ok_psd)
        # A sample's mean is taken over a fixed window, not a whole number of cycles (a noisy
        # trace has no cycles to align to), so it carries a phase error of order
        # amplitude/n_periods -- and on this cycle that is 4e-5, LARGER than the deviations
        # below. The mean field's own mean is cycle-aligned and does not. Quote the floor with
        # the number: a trend smaller than it is the window moving, not the physics.
        n_per = REC_MS / max(ref_period(name), 1e-300)
        phase = float(mf[col["var"]]) ** 0.5 / n_per       # amplitude/n_periods
        # ... which is ZERO at a fixed point, where the mean field has no amplitude at all. The
        # scale there is the sampling error of a mean taken over a noisy trace, for which the
        # seed spread is the only estimate on hand. A deviation has to clear whichever of the
        # two is larger; quoting it against the phase floor alone called sampling scatter at a
        # fixed point -- non-monotonic and sign-flipping across N -- a "real finite-N shift".
        floor = max(phase, float(r_se.max()))
        print(f"    mean:     |r_N - r_MF| ~ N^{p_b:+.2f}; largest deviation "
              f"{float(bias.max()):.1e} /ms against max(window phase {phase:.1e}, "
              f"seed spread {float(r_se.max()):.1e}) /ms"
              + ("  -> no bias is resolvable" if float(bias.max()) < 3 * floor else
                 "  -> the deviation clears both and is a real finite-N shift"))
        print(f"    variance: excess ~ {a_v:.3g} * N^{p_v:+.2f}   (quadrature law wants -1)")
        verdict = lambda p: ('phase diffusion, falls with N' if p < -0.5 else
                             'damping, N-independent' if p > -0.25 else 'neither, cleanly')
        print(f"    coherence, period {ref_period(name):.2f} ms, two estimators of D:")
        print(f"      autocorrelation envelope: N^{p_d:+.2f} over the {int(ok_acf.sum())} of "
              f"{len(ns)} points where the orbit dominates the noise"
              + (f"   ({verdict(p_d)})" if int(ok_acf.sum()) > 2 else "   [not resolved here]"))
        print(f"      spectral width pi*FWHM:   N^{p_w:+.2f} over the {int(ok_psd.sum())} of "
              f"{len(ns)} points clearing the {2 * floor_psd:.1e}/ms Welch resolution"
              + (f"   ({verdict(p_w)})" if int(ok_psd.sum()) > 2 else "   [not resolved here]"))
        # local slopes, because a single fitted exponent averages over the range where the
        # law holds and the range where the estimator has run out -- and those are the two
        # things worth telling apart
        print(f"      {'N':>9} {'D_acf /ms':>12} {'slope':>7} {'D_psd /ms':>12} {'slope':>7}")
        lsl = lambda v, j: (math.log(float(v[j + 1]) / float(v[j]))
                            / math.log(ns[j + 1] / ns[j])
                            if j + 1 < len(ns) and float(v[j]) > 0 and float(v[j + 1]) > 0
                            else float('nan'))
        for j, N in enumerate(ns):
            flag = "" if ok_acf[j] or ok_psd[j] else "   [both unresolved]"
            print(f"      {N:9d} {float(d_n[j]):12.3e} {lsl(d_n, j):7.2f} "
                  f"{float(d_psd[j]):12.3e} {lsl(d_psd, j):7.2f}{flag}")

        if (name, ns_epr[0], "sigma") in res:
            sig = torch.stack([res[(name, N, "sigma")] for N in ns_epr])   # (N, [mean, sem])
            mf_s = res[(name, "mf_sigma")]
            print(f"    entropy production, against the mean field's {mf_s:.4e} nats/ms:")
            for j, N in enumerate(ns_epr):
                m, se = float(sig[j, 0]), float(sig[j, 1])
                print(f"    {N:9d} {m:+.4e} +- {se:.1e}  ({m / se:+6.1f} sigma from zero, "
                      f"{(m - mf_s) / se:+6.1f} from the mean field)")


# ── Plots ─────────────────────────────────────────────────────────────────────

if run_plot:

    # N = 100 is deliberately not drawn. At dt = 0.25 its activity is quantised in steps of
    # 1/(N*dt) = 0.04 /ms against a mean rate of 0.013 -- the trace takes the values 0, 0.04,
    # 0.08 and is a point process rather than a rate. It belongs in the tables, where the
    # variance law uses it happily, and it swamps any axis it is drawn on.
    SHOW = [n for n in (1000, 10000, 1000000) if n in ns]
    PERIOD = {name: ref_period(name) for name in REGIMES}
    cmap = plt.get_cmap("viridis")
    CN = {N: cmap(0.05 + 0.65 * j / max(1, len(SHOW) - 1)) for j, N in enumerate(SHOW)}

    def n_axis(ax):
        ax.set_xscale('log'); ax.set_xlabel('system size  $N$')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 1. What the convergence looks like before it is a number. Left: the rate against time,
    # the mean field under the samples. Right: the same data delay-embedded at a quarter
    # period, which turns the cycle into a closed curve and the fixed point into a cloud --
    # and makes "the noise shrinks onto the deterministic object" a picture rather than a claim.
    fig, axs = plt.subplots(len(REGIMES), 2, figsize=(13, 4.6 * len(REGIMES)), squeeze=False)
    for i, name in enumerate(REGIMES):
        ax_t, ax_p = axs[i]
        period = PERIOD[name]
        lag = max(1, int(round(period / 4 / dt)))          # a quarter period, so a circle
        t_ms = torch.arange(int(30 * period / dt)) * dt

        n_t, n_p = int(6 * period / dt), int(60 * period / dt)   # 6 periods, 60 for the cloud
        mfx = res[(name, "mf_traj")].double()
        # largest N drawn last and the mean field over everything, so the collapse reads as
        # curves landing ON the black one rather than hiding behind it
        for j, N in enumerate(SHOW):
            x = res[(name, N, "traj")].double()
            ax_t.plot(t_ms[:n_t], x[:n_t], lw=0.9, color=CN[N], zorder=j, label=f'N = {N:,}')
            # the embedding is a CLOUD, so points, not a polyline: joining successive samples
            # of a noisy orbit draws the chords between them and fills the panel with scribble
            ax_p.plot(x[:n_p - lag], x[lag:n_p], '.', ms=1.2, alpha=0.3, color=CN[N], zorder=j)
        # dashed, because at N = 1e6 the sample lies exactly ON the mean field: a solid black
        # line on top would hide the very curve the panel is there to show has arrived
        ax_t.plot(t_ms[:n_t], mfx[:n_t], lw=1.6, color='k', ls=(0, (6, 3)), zorder=9,
                  label='mean field')
        ax_p.plot(mfx[:n_p - lag], mfx[lag:n_p], lw=1.8, color='k', ls=(0, (6, 3)), zorder=9)

        # Axes set by the MEAN FIELD plus a margin for the noise of the coarsest N drawn, not by
        # the data range: one outlier burst at small N otherwise rescales the panel and flattens
        # the object the figure is about.
        sd0 = float(res[(name, SHOW[0])][:, col["var"]].mean()) ** 0.5
        # clamped at zero: a rate cannot be negative, and letting the noise margin run below it
        # spends a third of the panel on values the model cannot take
        lo, hi = max(0.0, float(mfx.min()) - 3 * sd0), float(mfx.max()) + 3 * sd0
        ax_t.set_ylim(lo, hi)
        ax_p.set_xlim(lo, hi); ax_p.set_ylim(lo, hi); ax_p.set_aspect('equal')

        ax_t.set_xlim(0, 6 * period); ax_t.set_xlabel('t [ms]'); ax_t.set_ylabel('$r$ [1/ms]')
        ax_t.set_title(f'Population rate -- {name}'); ax_t.legend(fontsize=8); ax_t.grid(alpha=0.3)
        ax_p.set_xlabel('$r(t)$ [1/ms]'); ax_p.set_ylabel(f'$r(t + T/4)$ [1/ms]')
        ax_p.set_title(f'Delay embedding -- {name}'); ax_p.grid(alpha=0.3)
    fig.suptitle('The spin network collapses onto the mean field as $N$ grows')
    fig.tight_layout()

    # 2. The quadrature law. Left: the variance itself, against the mean-field value it has to
    # reach. Right: what is left after subtracting it, which is the finite-size noise alone and
    # is the same object in both regimes -- the fixed point's whole variance and the cycle's
    # correction lie on one line of slope -1.
    fig, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, mk in zip(REGIMES, ('s', 'o')):
        var_mf = float(res[(name, "mf")][col["var"]])
        v = torch.stack([res[(name, N)][:, col["var"]] for N in ns]).mean(dim=1)
        ax_v.plot(ns, v, mk + '-', markersize=6, label=f'{name}')
        # only where there IS a deterministic variance to converge to: at a fixed point it is
        # zero, and its float residue would set the bottom of the axis eleven decades down
        if var_mf > 1e-6 * float(v.max()):
            ax_v.axhline(var_mf, ls=':', lw=1, color=ax_v.lines[-1].get_color())
        ex = (v - var_mf).clamp_min(0)
        p, a = power_fit(ns, ex)
        ax_e.plot(ns, ex, mk, markersize=6, label=f'{name}:  $N^{{{p:+.2f}}}$')
        ax_e.plot(ns, a * torch.tensor([float(n) for n in ns]) ** p, '-', lw=1,
                  color=ax_e.lines[-1].get_color())
    ax_v.set_yscale('log'); ax_v.set_ylabel(r'$\mathrm{Var}[r]$  [(1/ms)$^2$]')
    ax_v.set_title('Variance, against the mean-field value (dotted)')
    ax_e.set_yscale('log')
    ax_e.set_ylabel(r'$\mathrm{Var}_N[r] - \mathrm{Var}_{\mathrm{MF}}[r]$')
    ax_e.set_title('The excess is finite-size noise: one line, slope $-1$')
    for ax in (ax_v, ax_e): n_axis(ax)
    fig.suptitle(r'$\mathrm{Var}[r] = \mathrm{Var}_{\mathrm{MF}}[r] + c/N$ '
                 r'-- the noise adds in quadrature')
    fig.tight_layout()

    # 3. The part the mean field cannot answer. Both regimes have a spectral peak at nearly the
    # same frequency and they are different objects, which only the N-dependence reveals: a
    # quasi-cycle's width is a damping rate and does not move, a cycle's width is phase
    # diffusion and falls as 1/N.
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, name in zip(axs, REGIMES):
        top = 0.0
        for N in SHOW:
            f, S = res[(name, N, "psd")]
            ax.plot(f, S, lw=1.0, color=CN[N], label=f'N = {N:,}')
            top = max(top, float(S.max()))
        f, S = res[(name, "mf_psd")]
        # A settled mean field has no spectrum: its "peak" here is 1e-23, the rounding of a
        # constant, and drawing it stretches the log axis over 35 decades and flattens every
        # curve the panel is about. Drawn only when it is a spectrum rather than numerical dust.
        if float(S.max()) > 1e-6 * top:
            ax.plot(f, S, lw=1.2, color='k', alpha=0.8, label='mean field')
        # five decades below the strongest peak: enough to show the noise floor each N sits on,
        # without the cycle's inter-harmonic troughs (1e-28) setting the scale
        ax.set_ylim(top / 1e5, 3 * top)
        ax.axvline(1.0 / PERIOD[name], color='0.5', lw=0.8, ls='--')
        ax.set_xlim(0, 4.0 / PERIOD[name])
        ax.set_yscale('log'); ax.set_xlabel(r'$f$ [1/ms]')
        ax.set_ylabel(r'$S_r(f)$'); ax.set_title(f'Spectrum -- {name}')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Filled: the autocorrelation envelope, which carries the cycle. Open: pi*FWHM, which
    # carries the fixed point. Each is plotted only where it clears its own floor (drawn as a
    # dotted line) -- below that it measures the record length or the Welch bin, and a point
    # sitting on its floor is not a measurement that happens to be small.
    floor_psd, floor_acf = torch.pi / SEG_MS, 3.0 / (REC_MS / 3)
    for name, mk in zip(REGIMES, ('s', 'o')):
        d = torch.stack([res[(name, N)][:, col["decay"]] for N in ns])
        w = torch.stack([res[(name, N)][:, col["fwhm"]] for N in ns])
        m, se = d.mean(dim=1), d.std(dim=1) / d.shape[1] ** 0.5
        dp = torch.pi * w.mean(dim=1)
        # same criteria as the analysis block: the envelope is trustworthy where the orbit
        # dominates the noise, the width where the peak is wider than the Welch bin
        var_mf = float(res[(name, "mf")][col["var"]])
        v_n = torch.stack([res[(name, N)][:, col["var"]] for N in ns]).mean(dim=1)
        ok_a = (torch.isfinite(m) & (m > 0) & (var_mf > (v_n - var_mf).clamp_min(0)))
        ok_p = torch.isfinite(dp) & (dp > 2 * floor_psd)
        nsT = torch.tensor([float(n) for n in ns])
        if int(ok_a.sum()):
            p, a = power_fit(ns, m, ok_a)
            axs[2].errorbar(nsT[ok_a], m[ok_a], yerr=se[ok_a], fmt=mk, markersize=6,
                            label=f'{name}, envelope:  $N^{{{p:+.2f}}}$')
            axs[2].plot(nsT, a * nsT ** p, '-', lw=1, color=axs[2].lines[-1].get_color())
        if int(ok_p.sum()):
            p, a = power_fit(ns, dp, ok_p)
            axs[2].plot(nsT[ok_p], dp[ok_p], mk, mfc='none', markersize=7,
                        label=f'{name}, $\\pi\\,$FWHM:  $N^{{{p:+.2f}}}$')
            axs[2].plot(nsT, a * nsT ** p, '--', lw=1, color=axs[2].lines[-1].get_color())
    # only the Welch resolution is a real gate now -- the envelope's reach is set by the
    # autocorrelation's own noise, which a near-deterministic trace makes very small
    axs[2].axhline(2 * floor_psd, color='0.6', ls=':', lw=0.9)
    axs[2].annotate('Welch resolution', (ns[0], 2 * floor_psd), fontsize=7, color='0.4',
                    va='bottom')
    axs[2].set_yscale('log')
    axs[2].set_ylabel(r'linewidth  $D$  [1/ms]')
    axs[2].set_title('Damping does not move with $N$;\nphase diffusion falls as $1/N$')
    n_axis(axs[2])
    fig.suptitle('A quasi-cycle and a cycle look alike at one $N$ and differ in how they scale')
    fig.tight_layout()

    # 4. Entropy production. Two different statements, so two panels: on the cycle sigma has a
    # mean-field value to converge TO, and at the fixed point it has only zero, so what matters
    # there is the size of the error bar.
    fig, axs = plt.subplots(1, len(REGIMES), figsize=(12, 4.6), squeeze=False)
    for ax, name in zip(axs[0], REGIMES):
        sig = torch.stack([res[(name, N, "sigma")] for N in ns_epr])
        mf_s = res[(name, "mf_sigma")]
        ax.errorbar(ns_epr, sig[:, 0], yerr=sig[:, 1], fmt='o-', markersize=6, capsize=3,
                    label='spin network')
        ax.axhline(mf_s, color='k', ls=':', lw=1.2,
                   label=f'mean field  ({mf_s:.2e})')
        # zero belongs on the axis only at the fixed point, where it IS the prediction. On the
        # cycle it pushes the whole convergence into the top tenth of the panel.
        if abs(mf_s) < 10 * float(sig[:, 1].max()):
            ax.axhline(0.0, color='0.7', lw=0.8)
        ax.set_ylabel(r'$\sigma/\Delta_t$  [nats/ms per neuron]')
        ax.set_title(f'Entropy production -- {name}')
        n_axis(ax)
    fig.suptitle(r'EPR against the mean field: a limit to converge to on the cycle, '
                 r'and only zero at the fixed point')
    fig.tight_layout()

    rp.show()
