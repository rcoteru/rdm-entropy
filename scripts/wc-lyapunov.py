"""
Stage 2 of the interesting-dynamics search: the leading Lyapunov exponents.

Stage 1 (wc-search.py) screens every cell with trajectory statistics and can say whether the
activity settles, but not what it settles *onto*. This pass computes the top two Lyapunov
exponents on the same grid, which is what separates a locked cycle from quasiperiodicity from
chaos, and gives a continuous lambda_1 field rather than a masked one.

Two exponents, not more. lambda_1 decides chaos and the fixed/cycle split; lambda_2 separates
an invariant circle (lambda_1 ~ 0 > lambda_2) from a 2-torus (both ~ 0) and catches hyperchaos.
Nothing in the classification reads lambda_3 onward, and they are largely unresolvable anyway:
the exponents come in complex-conjugate pairs, which in a map are exactly degenerate, so within
a pair only the partial sum is well defined. Any cell that comes back chaotic is worth a
targeted re-run at larger k for a Kaplan-Yorke dimension.

Things that are easy to get wrong here, each of which cost a measurement to find out:

  This is a MAP, not a flow. In a flow a limit cycle has one zero exponent, the phase
  direction. In a map an attracting periodic orbit is a fixed point of its P-th iterate and
  *every* exponent is strictly negative. So lambda_1 < 0 does not separate a dead cell from an
  oscillating one, and the classification has to be joined against the stage-1 CV to supply
  that bit. A flow-style table would file every locked cycle with the dead cells.

  The zero of lambda_1 has two floors, not one. A finite window gives a statistical floor
  (~1e-4 /ms at 20000 steps). On top of that the age lattice pins a genuinely marginal phase
  direction, biasing it negative by an amount that vanishes with dt: measured lambda_1 on one
  oscillating cell runs -2.7e-4, -2.3e-4, -1.7e-4, -0.5e-4 /ms at dt = 0.4, 0.2, 0.1, 0.05. At
  the production dt both floors are of order 1e-4, hence TOL below. The consequence is a blind
  spot: a chaotic cell with lambda_1 below ~5e-4 /ms is not distinguishable from zero here, so
  "no chaos found" is a weaker statement than "no chaos exists".

  Tangents are propagated only during the measurement window. The tangent step costs several
  times a plain state step, so equilibrating with tangents attached is pure waste.

  A positive lambda_1 at dt = 0.2 is not evidence of chaos on its own. Every candidate this
  sweep produced was re-run at a finer age grid and a finer timestep. The age grid is fine:
  eps = 1e-3 (Qm = [346, 1382]) reproduces the production eps = 0.01 values to within their
  error bars. The timestep is not: at dt = 0.1 all four candidates flip sign, from
  +9.3e-4, +2.5e-3, +5.1e-4, +4.5e-4 to -2.7e-4, -6.5e-4, -7.1e-4, +1.5e-4 /ms. So the
  production dt resolves the *sign* of a near-zero lambda_1 only to about 1e-3 /ms, and
  anything this sweep flags as chaotic has to be confirmed at dt <= 0.1 before it is believed.
  As of this writing no cell on this grid survives that confirmation.

Output is cached per cell and joined to the stage-1 label, so cells that had not equilibrated
by stage 1 are marked rather than silently reported as attractor quantities.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import torch
import tqdm

import rdme.kernels as krn
import rdme.lyapunov as lyap
from rdme.batch import RDMWilsonCowanPoints, flatten_param_grid, unflatten_batch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[1] / "cache"
CACHE_LYAP_FILE = CACHE_DIR / f"{bname}_lyap.pt"
STAGE1_FILE     = CACHE_DIR / "wc-search_stats.pt"

run_sim   = True
run_plot  = True
overwrite = False

torch.set_default_dtype(torch.float64)

# Half-width of the band around zero that counts as "not distinguishable from zero", in /ms.
# Clears both the finite-window statistical floor and the dt-dependent lattice bias; see the
# module docstring. Reported in the cache so downstream work does not have to guess it.
TOL = 5e-4

# Stage-1 CV above which a cell is treated as actually varying in time. Must match wc-search.py.
CV_TOL = 1e-4

# How many standard errors a nonzero exponent must clear. The measured SE across this grid has
# median 5.1e-4 /ms -- the same size as TOL -- so this, not TOL, is what actually decides chaos.
N_SIGMA = 3.0


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # The grid is stage 1's, verbatim -- the whole point is to overlay the two.
    n = 13
    E_ext = torch.linspace(0.70, 1.20, n)
    w_EE  = torch.linspace(0.0,   4.0, n)
    g     = torch.linspace(0.0,  15.0, n)
    w_II  = torch.linspace(-10.0, 0.0, n)
    rho   = 2.0 ** 0.5

    E_ratio = 0.8
    beta    = 30.0
    theta   = 1.0
    tau_int_E, tau_int_I = 10.0, 40.0
    tau_ref = 3.0
    K_ref_E, K_ref_I = 0.5, 0.0

    dt       = 0.2
    equi     = 15000   # 3000 ms, state only -- no tangents attached yet
    warmup   = 2000    # tangent-basis alignment, logged but discarded
    steps    = 20000   # 4000 ms of accumulation; sets the ~1e-4 /ms statistical floor
    k        = 2
    cadence  = 100     # see lyapunov.benettin: the safe limit here is ~24000 steps
    n_blocks = 20      # block means for the error bars

    if CACHE_LYAP_FILE.exists() and not overwrite:
        print(f"Spectra already exist at {CACHE_LYAP_FILE}. Skipping.")
    else:
        axes_in = {"E_ext": E_ext, "w_EE": w_EE, "g": g, "w_II": w_II}
        flat, grid_shape, swept = flatten_param_grid(axes_in)
        B = flat["w_EE"].numel()

        Qm = [krn.q_from_tau(max(tau_int_E, tau_ref), dt, 0.01),
              krn.q_from_tau(max(tau_int_I, tau_ref), dt, 0.01)]
        D = 2 * sum(Qm)

        # Chunk from a memory model rather than copied from stage 1, which had no tangents.
        # The tangent step allocates roughly c_tan temporaries of (B, k, Q) on top of the basis
        # itself, and the state costs c_state of (B, Q); measured peak was 0.98 GiB at B=2197,
        # k=6, so without this a smaller dt (Qm doubles) or a larger k would silently OOM.
        c_tan, c_state, budget = 5, 5, 6 * 2**30
        chunk = min(n**3, int(budget / (8 * D * (c_tan * k + c_state))))
        n_chunks = (B + chunk - 1) // chunk

        print(f"Grid: {n}**4 = {B} cells, k={k}, Qm={Qm} (D={D}).")
        print(f"{n_chunks} chunks of up to {chunk}. "
              f"{equi} equilibration + {warmup} warmup + {steps} accumulation steps each.")
        print(f"Running on: {device}")

        t0 = time.time()
        parts = []
        gen = torch.Generator(device=device).manual_seed(0)
        for lo in tqdm.tqdm(range(0, B, chunk), desc="chunks"):
            sl = slice(lo, min(lo + chunk, B))
            gc = flat["g"][sl]
            # srm takes the weights in physical units -- the 1/dt the old call site applied
            # is internal now, since the input is built from the rate. hazard="synchronous"
            # is the sigmoid of the original model, so the numbers are unchanged.
            mf = RDMWilsonCowanPoints(
                E_ratio, flat["w_EE"][sl], gc*rho, -gc/rho, flat["w_II"][sl],
                flat["E_ext"][sl], flat["E_ext"][sl], beta, beta, theta, theta,
                tau_int_E, tau_int_I, tau_ref, tau_ref, K_ref_E, K_ref_I,
                dt=dt, hazard="synchronous", Qm=Qm, device=device)

            mf.forward(equi, pb=False)          # state only: tangents here would be wasted
            blocks = mf.init_tangent(k, generator=gen)
            mf.step_tangent(blocks)             # one step annihilates the 2M spurious modes
            lyap.orthonormalize(blocks)

            out = lyap.benettin(mf.step_tangent, blocks, n_steps=steps, dt=dt,
                                cadence=cadence, n_blocks=n_blocks, warmup=warmup)
            parts.append({f: getattr(out, f).cpu() for f in
                          ("lam", "se", "partial", "drift", "degenerate", "valid")})
            del mf, blocks
            if device == 'cuda': torch.cuda.empty_cache()

        res = {f: torch.cat([q[f] for q in parts]).reshape(*grid_shape, k)
               for f in parts[0]}
        print(f"Done in {time.time() - t0:.2f}s")

        res["axes"] = swept
        res["rho"], res["dt"], res["k"] = rho, dt, k
        res["tol"], res["cv_tol"] = TOL, CV_TOL
        res["T_ms"] = steps * dt      # in time, not steps: the floor is a per-ms quantity
        res["equi"], res["warmup"], res["cadence"], res["n_blocks"] = equi, warmup, cadence, n_blocks

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(res, CACHE_LYAP_FILE)
        print(f"Spectra saved to {CACHE_LYAP_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    for f in (CACHE_LYAP_FILE, STAGE1_FILE):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run wc-search.py then this script.")

    res = torch.load(CACHE_LYAP_FILE, weights_only=False)
    st1 = torch.load(STAGE1_FILE, weights_only=False)
    axes, dt, tol = res["axes"], res["dt"], res["tol"]

    lam = res["lam"]
    lam1 = lam[..., 0]

    # the bit the spectrum cannot supply: does the state actually move?
    moving = st1["cv"] > res["cv_tol"]
    # Per-cell error bars, not a scalar threshold: the SE varies by an order of magnitude
    # across the grid and is comparable to TOL, so a scalar cut counts every cell whose noise
    # lands positive as chaotic (138 cells here, against 3 that survive 3 sigma).
    label = lyap.classify_spectrum(lam, tol=tol, moving=moving, se=res["se"], n_sigma=N_SIGMA)

    # Cells stage 1 flagged as still decaying or growing had not reached an attractor by the
    # end of the same equilibration, so their exponents describe a transient. Mark rather than
    # drop: where they sit is itself informative.
    ratio = st1["std_2"] / st1["std_1"].clamp_min(1e-300)
    unequilibrated = moving & ((ratio < 0.9) | (ratio > 1.1))

    counts = {lyap.CLASS_NAMES[c]: int((label == c).sum()) for c in range(5)}
    print(f"\nCell counts by class ({label.numel()} total): {counts}")
    print(f"unequilibrated (stage-1 decaying/growing): {int(unequilibrated.sum())}")
    print(f"degenerate leading pair: {int(res['degenerate'][..., 0].sum())}")
    print(f"|lam|*dt >= 0.1 (lam/dt not a rate):      {int((~res['valid']).sum())}")
    se1 = res["se"][..., 0]
    print(f"\ntol = {tol:.1e} /ms (floor) over T = {res['T_ms']:.0f} ms; "
          f"SE of lam_1: median {se1.median():.1e}, 90th pct {se1.flatten().quantile(0.9):.1e}")
    print(f"the binding cut is {N_SIGMA:g}*SE, so chaos below ~{N_SIGMA*se1.median():.1e} /ms "
          f"is not detectable at this window")

    # Drift is a bias check the SE cannot make: a cell still converging can be statistically
    # significant and still wrong. Report it rather than folding it into the class.
    unconverged = chaotic_raw = (label == lyap.CHAOS)
    drifting = chaotic_raw & (res["drift"][..., 0].abs() > lam1)
    if int(drifting.sum()):
        print(f"of the chaotic cells, {int(drifting.sum())} still drift by more than lam_1 "
              f"and should be re-run longer before being believed")

    chaotic = label == lyap.CHAOS
    if chaotic.any():
        idx = torch.nonzero(chaotic)
        order = torch.argsort(lam1[chaotic], descending=True)
        print("\nChaotic cells by lambda_1 (re-run these at larger k for Kaplan-Yorke):")
        for q in order[:15]:
            ii = idx[q].tolist()
            coords = "  ".join(f"{a}={axes[a][x]:7.2f}" for a, x in
                               zip(["E_ext", "w_EE", "g", "w_II"], ii))
            print(f"  {coords} | lam1={lam1[tuple(ii)]:+.3e}  "
                  f"se={res['se'][tuple(ii) + (0,)]:.1e}  drift={res['drift'][tuple(ii) + (0,)]:+.1e}")
    else:
        print("\nNo chaotic cells above the detection floor.")

    # ── lambda_1 field, projected ─────────────────────────────────────────────
    # Diverging around zero: the sign is the question, so the midpoint is the meaningful
    # value, not the data's middle.
    lo, hi = lam1.min().item(), lam1.max().item()
    norm = TwoSlopeNorm(vmin=min(lo, -tol), vcenter=0.0, vmax=max(hi, tol))

    pairs = [(0, 1), (0, 2), (1, 2), (1, 3)]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    names = ["E_ext", "w_EE", "g", "w_II"]
    labels = {"E_ext": "$E$ (external drive)", "w_EE": "$w_{EE}$",
              "g": "$g$ (E-I loop gain)", "w_II": "$w_{II}$"}
    for ax, (a, b) in zip(axs.ravel(), pairs):
        others = tuple(d for d in range(4) if d not in (a, b))
        f = lam1.amax(dim=others)       # max, not mean: one chaotic cell must not be averaged away
        if a > b: f = f.T
        xa, ya = axes[names[b]], axes[names[a]]
        im = ax.imshow(f, origin='lower', aspect='auto', cmap='RdBu_r', norm=norm,
                       extent=(xa.min().item(), xa.max().item(), ya.min().item(), ya.max().item()))
        ax.set_xlabel(labels[names[b]]); ax.set_ylabel(labels[names[a]])
        fig.colorbar(im, ax=ax, label=r'$\max\ \lambda_1$  [/ms]')
    fig.suptitle(r'Leading Lyapunov exponent $\lambda_1$ (max over the collapsed axes)')
    fig.tight_layout()

    # ── classification slice ──────────────────────────────────────────────────
    per_cell = (label == lyap.CIRCLE).to(torch.float64).mean(dim=(0, 1))
    j, kk = [int(x) for x in torch.nonzero(per_cell == per_cell.max())[0]]

    cmap = ListedColormap(['#e8e8e8', '#2e6fb7', '#f2c14e', '#7b4ea3', '#b4453c'])
    norm2 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f'Attractor class at $g$={axes["g"][j]:.2f}, $w_{{II}}$={axes["w_II"][kk]:.2f}')
    im = ax.imshow(label[:, :, j, kk], cmap=cmap, norm=norm2, origin='lower', aspect='auto',
                   extent=(axes["w_EE"].min().item(), axes["w_EE"].max().item(),
                           axes["E_ext"].min().item(), axes["E_ext"].max().item()))
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(lyap.CLASS_NAMES)
    ax.set_xlabel(labels["w_EE"]); ax.set_ylabel(labels["E_ext"])
    fig.tight_layout()

    plt.show()
