""" 1-D scan of the recurrent coupling J for one single-population (Ising) system: bifurcation
scatter, fixed-point branches with stability, spectrogram and mean entropy production. """

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import rdme.convergence as cvg
import rdme.fixed_points as fpts
import rdme.helpers as hlp
import rdme.plots as rp
from rdme.mean_field import RDMIsingModelBatch

# ── Parameters ────────────────────────────────────────────────────────────────

# Base parameter set, written out in full.
S1 = dict(
    E=0.0,
    beta=6.0, theta=0.5, tau_int=15.0, tau_ref=15.0, K_ref=15.0, nu0=0.04, t_abs=1.0)

HAZARD = "poisson_exact"   # "poisson_exact" | "poisson" | "bernoulli"

PARAMS: dict = S1 | dict(J=torch.linspace(0.0, 80.0, 150), dt=0.2, hazard=HAZARD)

TOL = 1e-2   # age-grid truncation tolerance (sets Q)

AXIS  = "J"                          # the swept parameter: the one PARAMS gives a vector
LABEL = 'Recurrent coupling  J'

# ms: equilibration, then recording; step counts follow dt
T_EQUI, T_MEAS = 10000.0, 15000.0

EQUI  = hlp.n_steps(T_EQUI, PARAMS["dt"], name="T_EQUI")
STEPS = hlp.n_steps(T_MEAS, PARAMS["dt"], name="T_MEAS")

N_LEAD = 4      # leading eigenvalues kept per fixed point
N_J_FP = 100    # J values to analyse; the cost is dominated by the eigendecomposition
N_SCAN = 401    # resolution of the sign scan that brackets the roots

# m_tot is cached whole (scatter + spectrogram); the rest only as time averages
MEANS = ("sigma_tot", "H_fwd_tot", "H_rev_tot")

# Burst-resolution criterion (cvg.classify_bursts): the continuum burst width w0, from the FWHM
# at dt, dt/2, dt/4, must span N_RES bins of dt. Validity only -- a resolved J can still be
# inaccurate at this dt. Widths are measured over the last T_CHK ms of equilibration.
T_CHK = 1000.0
N_RES = 10.0     # w0/dt at or above: resolved

TITLE = (f'S1  ·  hazard={HAZARD}  ·  dt={PARAMS["dt"]:g}  —  '
         + ',  '.join(f'{k}={PARAMS[k]:g}' for k in
                      ("E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "nu0", "t_abs")))


# ── Paths & switches ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parents[1] / "cache"
STEM      = Path(__file__).stem
CACHE_SIM = hlp.cache_path(STEM, dict(PARAMS, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                                      t_chk=T_CHK),
                           tag="traj", root=CACHE_DIR)
CACHE_CHK = hlp.cache_path(STEM, dict(PARAMS, t_equi=T_EQUI, t_meas=T_MEAS, tol=TOL,
                                      t_chk=T_CHK, osc_min=cvg.OSC_MIN),
                           tag="chk", root=CACHE_DIR)
CACHE_FP  = hlp.cache_path(STEM, dict(PARAMS, n_lead=N_LEAD, n_J=N_J_FP, n_scan=N_SCAN, tol=TOL),
                           tag="fp", root=CACHE_DIR)

run_sim   = True
run_chk   = True
run_fp    = True
run_plot  = True
overwrite = False  # the hash covers every parameter, so this is only for a change to the CODE


# ── Simulation ────────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if run_sim and (overwrite or not CACHE_SIM.exists()):

    mf = RDMIsingModelBatch(**PARAMS, tol=TOL, device=device)
    print(f"Running {mf.B} mean-fields for {T_EQUI + T_MEAS:g} ms "
          f"({EQUI + STEPS} steps at dt={PARAMS['dt']:g}, Q={mf.Q}) on {device}...")
    t0 = time.time()
    mf.forward(EQUI - hlp.n_steps(T_CHK, PARAMS["dt"], name="T_CHK"), pb=True)
    r_chk = cvg.watch_bursts(mf, T_CHK)
    traj = mf.entropy_trajectory(STEPS, pb=True, inplace=True)

    trunc = float(mf.truncation_error().max())
    assert trunc < TOL, f"truncation_error = {trunc:.2e} at Q={mf.Q} exceeds TOL={TOL:g}"
    print(f"Done in {time.time() - t0:.2f}s")

    amp, fwhm_ms = cvg.burst_fwhm(r_chk, PARAMS["dt"])

    out: dict = {"params": PARAMS, "t_equi": T_EQUI, "t_meas": T_MEAS,
                 "equi": EQUI, "steps": STEPS, "m_tot": traj["m_tot"].cpu(),
                 "osc_amp": amp, "fwhm_ms": fwhm_ms}
    out.update({f"{key}_{s}": getattr(traj[key], s)(dim=-1).cpu()
                for key in MEANS for s in ("mean", "std")})

    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(out, CACHE_SIM)
    print(f"Saved to {CACHE_SIM} ({CACHE_SIM.stat().st_size/2**20:.1f} MiB).")


# ── Burst-width check at dt/2, dt/4 ───────────────────────────────────────────
# The oscillating rows only, rerun to the same physical time at each finer step: with the
# main run's width that is the triple w0 is extrapolated from.

if run_chk and CACHE_SIM.exists() and (overwrite or not CACHE_CHK.exists()):

    d    = torch.load(CACHE_SIM, weights_only=False)
    rows = d["osc_amp"] >= cvg.OSC_MIN
    fwhm = torch.full((3,) + rows.shape, float("nan"), dtype=torch.float64)
    fwhm[0] = d["fwhm_ms"]

    for k in (1, 2):
        if not rows.any():
            break
        dt_k = PARAMS["dt"] / 2**k
        mf_k = RDMIsingModelBatch(**(PARAMS | dict(J=PARAMS[AXIS][rows], dt=dt_k)), tol=TOL,
                                  device=device)
        print(f"Width check: {int(rows.sum())} {AXIS} values at dt={dt_k:g} (Q={mf_k.Q})...")
        mf_k.forward(hlp.n_steps(T_EQUI - T_CHK, dt_k, name="T_EQUI - T_CHK"), pb=True)
        fwhm[k, rows] = cvg.burst_fwhm(cvg.watch_bursts(mf_k, T_CHK), dt_k)[1]

    torch.save({"fwhm_ms": fwhm, "dts": [PARAMS["dt"] / 2**k for k in range(3)]},
               CACHE_CHK)
    print(f"Width check saved to {CACHE_CHK}.")


# ── Fixed points ──────────────────────────────────────────────────────────────

if run_fp and (overwrite or not CACHE_FP.exists()):

    # on CPU: the fixed-point pass is tiny and LAPACK's eig has no GPU equivalent here
    mf_fp  = RDMIsingModelBatch(**PARAMS, tol=TOL, device="cpu")
    stride = max(1, mf_fp.B // N_J_FP)
    idx    = torch.arange(0, mf_fp.B, stride)

    print(f"Finding fixed points at {len(idx)} of {mf_fp.B} {AXIS} values (Q={mf_fp.Q})...")
    t0 = time.time()
    branches: dict = fpts.branch_scan(mf_fp, mf_fp.grid_axes[AXIS], idx, n_grid=N_SCAN, n_lead=N_LEAD)
    print(f"Done in {time.time() - t0:.2f}s, {len(branches['m_star'])} fixed points.")

    branches["params"] = PARAMS
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(branches, CACHE_FP)
    print(f"Fixed points saved to {CACHE_FP}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_SIM.exists():
        raise FileNotFoundError(f"{CACHE_SIM} not found. Run the simulation first.")

    print(f"Plotting {CACHE_SIM.name}.")
    d   = torch.load(CACHE_SIM, weights_only=False)
    fp  = torch.load(CACHE_FP, weights_only=False) if CACHE_FP.exists() else None
    chk = torch.load(CACHE_CHK, weights_only=False) if CACHE_CHK.exists() else None

    x, dt = PARAMS[AXIS].numpy(), PARAMS["dt"]
    if chk is not None:
        w0 = cvg.burst_limit(chk["fwhm_ms"])
    else:
        w0 = torch.full_like(d["fwhm_ms"], float("nan"))
    cls = cvg.classify_bursts(d["osc_amp"], w0, dt, n_res=N_RES)

    for c in cvg.BURST_CLASSES:
        if (cls == c).any():
            print(f"  {c:>11}: {(cls == c).sum():3d}  ({AXIS} {x[cls == c].min():.3g} – "
                  f"{x[cls == c].max():.3g})")

    figs = rp.sweep_figures(PARAMS[AXIS], d["m_tot"], dt=dt, label=LABEL,
                            banner=TITLE, fp=fp, means={k: d[f"{k}_mean"] for k in MEANS})

    for ax in figs["entropy"].axes:
        for c, color in (("unresolved", "0.75"), ("unconverged", "0.9")):
            ax.fill_between(x, 0, 1, where=(cls == c), step="mid", color=color, zorder=0,
                            transform=ax.get_xaxis_transform(), label=f"burst {c}")
    figs["entropy"].axes[0].legend()

    # burst width: the measured FWHM at each dt, its extrapolation, and the resolution line
    if chk is not None:
        fig, ax = plt.subplots(figsize=(11, 5))
        fig.suptitle(f'Burst width    —    {TITLE}', fontsize=13)
        osc = cls != "no_osc"
        for k, dt_k in enumerate(chk["dts"]):
            ax.plot(x[osc], chk["fwhm_ms"][k].numpy()[osc], "o-", ms=3, lw=1,
                    label=f"FWHM, dt={dt_k:g}")
        ok = (w0 > 0).numpy() & ((cls == "resolved") | (cls == "unresolved"))
        ax.plot(x[ok], w0.numpy()[ok], "k-", lw=2, label="w0 (Richardson)")
        ax.axhline(N_RES * dt, color="C3", ls="--", label=f"N_RES·dt = {N_RES:g}·{dt:g} ms")
        ax.set_yscale("log")
        ax.set(xlabel=LABEL, ylabel="FWHM  (ms)")
        ax.grid(which="both", alpha=0.4)
        ax.legend()
        fig.set_label("burst_width")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        figs["burst_width"] = fig

    rp.show()
