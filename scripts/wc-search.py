"""
Stage 1 of the interesting-dynamics search: variance screening over the drive,
the self-excitation, the E-I loop gain and the self-inhibition.

Sweeps (E, w_EE, g, w_II), equilibrates, then asks the cheapest useful
question -- *does the activity stop moving?* -- from running trajectory
statistics alone. No Jacobians, no eigenvalues: the whole grid costs one
simulation, which is what makes it affordable to screen everything first and
spend the expensive diagnostics only on what survives.

Design notes, each of which came from a probe that went wrong first:

  The external drive E is an axis, because it selects the regime. At the
  nominal E = theta = 1.0 the drive alone puts the population at threshold and
  every synaptic weight is a perturbation on top of it -- recurrent excitation
  contributes ~10% of theta and recurrent inhibition ~1.5%. In that corner w_II
  and tau_I are numerically inert and the phase diagram collapses to a 1D
  threshold in w_EE. The live band is narrow (bursting at E = 0.9, silent by
  E = 0.8), so E is swept densely and everything else is read relative to it.

  w_EI and w_IE are replaced by loop gain and asymmetry. A 13**4 sweep over the
  raw weights put a clean diagonal boundary in the (w_EI, w_IE) plane: the
  oscillation dies on a curve of roughly constant *product*, which is what the
  Hopf determinant depends on (w_EI*w_IE*f'_E*f'_I). Sweeping both raw weights
  therefore spends 169 cells resolving approximately one direction. Here
      w_EI = g*rho,   w_IE = -g/rho,
  so g = sqrt(w_EI*|w_IE|) is the loop gain that the boundary actually tracks
  and rho = sqrt(w_EI/|w_IE|) is the asymmetry it is largely indifferent to.
  Only g is swept; rho is held fixed and is the obvious thing to vary next if
  the boundary turns out to care about it after all.

  Indicators are dt-invariant. m is a firing probability per age bin, not a
  rate, so peak-to-peak m shrinks with dt for the *same* physical trajectory --
  a fixed threshold on it would mean a different physical cutoff at each dt.
  Amplitudes are reported as rates (m/dt) and the classification runs off the
  coefficient of variation std(m)/mean(m), which is dimensionless. The
  separation is enormous (CV ~ 3e-8 at a fixed point vs ~1.5 on a cycle), so
  the threshold is not delicate.

  The recording window is split in half. A fixed point that is merely
  approached *slowly* also has nonzero variance over any finite window, and is
  the main false positive raw variance suffers from. A sustained attractor has
  the same spread in both halves; a decaying transient does not. In practice
  this band lands exactly on the bifurcation, which is what it should do --
  critical slowing down makes the transient longest there.

  Statistics are streamed, not stored. A 4D grid times T timesteps does not fit
  in memory; the running accumulators are O(B) instead of O(B*T), which is what
  lets the grid resolution be a free parameter.

  The batch comes from RDMWilsonCowanPoints, not RDMWilsonCowanBatch. The latter
  forms the outer product of *every* argument, so neither of the two axes here
  can go through it: E is two arguments (E_exc, E_inh) that have to covary, and g
  maps to both w_EI and w_IE. Passing either as a vector would silently give
  n**2 cells instead of n, with the off-diagonal ones quietly meaningless.

Output is a classification array plus the cached indicators, so stage 2 (return
maps, 0-1 test, Lyapunov) can pick up the flagged cells without re-running
anything.
"""

from itertools import combinations
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import tqdm

import rdme.kernels as krn
from rdme.batch import RDMWilsonCowanPoints, flatten_param_grid, unflatten_batch


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR       = Path(__file__).parents[1] / "cache"
CACHE_STAT_FILE = CACHE_DIR / f"{bname}_stats.pt"

run_sim   = True
run_plot  = True
overwrite = False   # overwrite existing cache file if True

torch.set_default_dtype(torch.float64)


# ── Classification thresholds ─────────────────────────────────────────────────

# Coefficient of variation below which a cell counts as a fixed point. Measured
# values are ~3e-8 at a settled fixed point and ~1.5 on a cycle, so anything in
# the middle of that gap works; this sits well clear of both.
CV_TOL = 1e-4

# std(second half) / std(first half). A sustained attractor sits at 1. Well below
# means still contracting; well above means still expanding, which in practice is
# a too-short equilibration rather than a genuine instability.
DECAY_LO, DECAY_HI = 0.9, 1.1

# Class codes, in the order used by the colormap below.
FIXED, DECAYING, SUSTAINED, GROWING = 0, 1, 2, 3
CLASS_NAMES = ["fixed point", "decaying", "sustained", "growing"]

AXES = ["E_ext", "w_EE", "g", "w_II"]
AXIS_LABELS = {"E_ext": "$E$ (external drive, $\\theta=1$)",
               "w_EE":  "$w_{EE}$ (E self-excitation)",
               "g":     "$g=\\sqrt{w_{EI}|w_{IE}|}$ (E-I loop gain)",
               "w_II":  "$w_{II}$ (I self-inhibition)"}


# ── Streaming statistics ──────────────────────────────────────────────────────

@torch.inference_mode()
def streaming_stats(mf, steps: int) -> dict[str, torch.Tensor]:
    """ Run `steps` updates, accumulating m_E statistics in O(B) memory.

    The excitatory population carries the oscillation and m_I follows it, so
    screening on m_E alone is enough at this stage.

    Sums are shifted by the first sample before accumulating. Without that,
    var = E[x^2] - E[x]^2 at a settled fixed point asks float64 to resolve a
    variance ~1e-15 of the mean square, which is its entire relative precision --
    the fixed-point CVs come back as numerical noise of indeterminate sign.
    Shifting makes the cancellation exact instead of catastrophic. """
    half = steps // 2
    n1, n2 = half, steps - half
    zeros = lambda: torch.zeros(mf.B, device=mf.device)

    s1, q1, s2, q2 = zeros(), zeros(), zeros(), zeros()
    lo = torch.full((mf.B,), float('inf'),  device=mf.device)
    hi = torch.full((mf.B,), float('-inf'), device=mf.device)
    shift = mf.m[:, 0].clone()

    for t in tqdm.tqdm(range(steps), desc="recording", leave=False):
        x = mf.m[:, 0]
        lo = torch.minimum(lo, x)
        hi = torch.maximum(hi, x)
        d = x - shift
        if t < half: s1 += d; q1 += d * d
        else:        s2 += d; q2 += d * d
        mf.update()

    # per-half moments, undoing the shift only where the absolute value is needed
    var = lambda q, s, n: (q / n - (s / n) ** 2).clamp_min(0.0)
    mean = (s1 + s2) / steps + shift
    return {
        "mean":  mean,
        "std":   var(q1 + q2, s1 + s2, steps).sqrt(),
        "std_1": var(q1, s1, n1).sqrt(),
        "std_2": var(q2, s2, n2).sqrt(),
        "amp":   hi - lo,
        # Slow drift of the mean catches transients that hold a roughly constant
        # spread while the whole trajectory slides -- the split-half ratio alone
        # is blind to those. The shift cancels in the difference.
        "drift": (s2 / n2 - s1 / n1).abs(),
    }


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Swept parameters. Ranges chosen from 1D probes and from the 13**4 raw-weight
    # sweep that preceded this one:
    #   E_ext -- the regime selector. At E = theta = 1.0 the drive alone holds the
    #            population at threshold; bursting survives to E = 0.9 and the
    #            activity is dead by E = 0.8, so the whole live band is within
    #            ~20% of theta and wants dense sampling, not a fixed value.
    #   w_EE  -- oscillation onset sits near w_EE ~ 1.15 (CV = 3e-8 at 0.5,
    #            CV = 1.04 at 1.5), so the window must start below 1. The ceiling
    #            is above the previous sweep's 3.0, where every top-amplitude cell
    #            piled up against the edge of the box.
    #   g     -- loop gain. The raw-weight sweep put the quench boundary on a
    #            diagonal of roughly constant w_EI*|w_IE| ~ 75-100, i.e. g ~ 9, so
    #            this range straddles it.
    #   w_II  -- a real axis once inhibition participates: the sustained fraction
    #            runs from ~0.55 at w_II = 0 to ~0.2 at w_II = -10.
    n = 13   # per axis; cost is n**4
    E_ext = torch.linspace(0.70, 1.20, n)
    w_EE  = torch.linspace(0.0,   4.0, n)
    g     = torch.linspace(0.0,  15.0, n)
    w_II  = torch.linspace(-10.0, 0.0, n)

    # Loop asymmetry, held fixed: w_EI = g*rho, w_IE = -g/rho. sqrt(2) reproduces
    # the nominal (w_EI, w_IE) = (2, -1) of the other wc- scripts at g = sqrt(2).
    # The Hopf determinant depends on the product, so this direction is expected to
    # be the weak one -- but that is an expectation, not a measurement, and it is
    # the first thing to put back on the grid if the boundary looks rho-dependent.
    rho = 2.0 ** 0.5

    # Fixed parameters, otherwise as in wc-EEII_ent_sweep.py. tau_int_I > tau_int_E
    # follows the 2D Wilson-Cowan Hopf condition
    #   w_EI*w_IE*f'_E*f'_I > (tau_E/tau_I)*(1 + w_II*f'_I)**2,
    # where slow inhibition lowers the bar the E-I loop gain must clear. K_ref is
    # this model's slow adaptation process -- the third timescale anything beyond a
    # limit cycle needs -- and its effect is carried entirely by the excitatory
    # population, so K_ref_I is left at zero. Raising tau_ref to make that process
    # genuinely slow was tried and does *not* produce sustained aperiodicity: the
    # elevated burst-height spread at tau_ref = 60-100 falls 4-7x between the first
    # and second half of the peak train, i.e. it is a longer transient, not chaos.
    E_ratio = 0.8
    beta    = 30.0
    theta   = 1.0
    tau_int_E, tau_int_I = 10.0, 40.0
    tau_ref = 3.0
    K_ref_E, K_ref_I = 0.5, 0.0

    dt      = 0.2
    equi    = 15000   # 3000 ms == 75 tau_I
    steps   = 4000    # 800 ms, ~50 periods at the ~16 ms burst period seen here

    if CACHE_STAT_FILE.exists() and not overwrite:
        print(f"Statistics already exist at {CACHE_STAT_FILE}. Skipping.")
    else:
        # The search axes are not the model's parameters, so the grid is built here
        # and mapped afterwards: E_ext drives both populations' external input, and
        # g sets w_EI and w_IE together.
        axes_in = {"E_ext": E_ext, "w_EE": w_EE, "g": g, "w_II": w_II}
        flat, grid_shape, swept = flatten_param_grid(axes_in)
        B = flat["w_EE"].numel()

        # One age grid for every chunk, sized from the slowest kernel per population.
        Qm = [krn.q_from_tau(max(tau_int_E, tau_ref), dt, 0.01),
              krn.q_from_tau(max(tau_int_I, tau_ref), dt, 0.01)]

        # Chunked by flat index: the full 4D batch would need several GiB of (p, S)
        # state before any intermediates, and chunking keeps the grid resolution a
        # free parameter rather than something bounded by VRAM.
        chunk = n ** 3
        print(f"Grid: {n}**4 = {B} cells, in {(B + chunk - 1)//chunk} chunks of {chunk}. "
              f"Qm = {Qm}. {equi+steps} steps each "
              f"({equi} equilibration + {steps} recording).")
        print(f"Running on: {device}")

        t0 = time.time()
        parts = []
        for lo in tqdm.tqdm(range(0, B, chunk), desc="chunks"):
            sl = slice(lo, min(lo + chunk, B))
            # The reparametrisation lives here, not in the factory: which coordinates
            # a search uses is a modelling choice, while zipping columns is plumbing.
            gc = flat["g"][sl]
            # srm takes the weights in physical units -- the 1/dt the old call site applied
            # is internal now, since the input is built from the rate. hazard="synchronous"
            # is the sigmoid of the original model, so the numbers are unchanged.
            mf = RDMWilsonCowanPoints(
                E_ratio, flat["w_EE"][sl], gc*rho, -gc/rho, flat["w_II"][sl],
                flat["E_ext"][sl], flat["E_ext"][sl], beta, beta, theta, theta,
                tau_int_E, tau_int_I, tau_ref, tau_ref, K_ref_E, K_ref_I,
                dt=dt, hazard="synchronous", Qm=Qm, device=device)
            mf.forward(equi, pb=False)
            parts.append({k: v.cpu() for k, v in streaming_stats(mf, steps).items()})
            del mf
            if device == 'cuda': torch.cuda.empty_cache()

        stats = {k: torch.cat([q[k] for q in parts]).reshape(grid_shape)
                 for k in parts[0]}
        print(f"Done in {time.time() - t0:.2f}s")

        # rates, so the numbers mean the same thing at any dt; CV is dimensionless
        stats["amp_rate"]  = stats.pop("amp") / dt
        stats["mean_rate"] = stats["mean"] / dt
        stats["cv"]        = stats.pop("std") / stats["mean"].clamp_min(1e-300)
        stats["drift"]     = stats["drift"] / dt
        stats["axes"]      = swept
        stats["rho"]       = rho
        stats["dt"], stats["steps"], stats["equi"] = dt, steps, equi

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(stats, CACHE_STAT_FILE)
        print(f"Statistics saved to {CACHE_STAT_FILE}.")


# ── Analysis & Plot ───────────────────────────────────────────────────────────

if run_plot:

    if not CACHE_STAT_FILE.exists():
        raise FileNotFoundError(f"{CACHE_STAT_FILE} not found. Run simulation first.")

    stats = torch.load(CACHE_STAT_FILE, weights_only=False)
    axes  = stats["axes"]

    cv    = stats["cv"]
    amp   = stats["amp_rate"]
    ratio = stats["std_2"] / stats["std_1"].clamp_min(1e-300)
    drift = stats["drift"]

    # ── Classification ────────────────────────────────────────────────────────
    # Order matters: the CV floor wins, because the split-half ratio of two
    # numerically-zero spreads is meaningless.
    moving = cv > CV_TOL
    label  = torch.full(cv.shape, FIXED, dtype=torch.long)
    label[moving & (ratio < DECAY_LO)] = DECAYING
    label[moving & (ratio > DECAY_HI)] = GROWING
    label[moving & (ratio >= DECAY_LO) & (ratio <= DECAY_HI)] = SUSTAINED
    # A cell whose mean is still sliding by more than a tenth of its own
    # peak-to-peak has not settled, whatever its spread ratio says.
    label[moving & (drift > 0.1 * amp)] = DECAYING

    counts = {CLASS_NAMES[c]: int((label == c).sum()) for c in range(4)}
    print(f"\nCell counts by class ({label.numel()} total): {counts}")

    sustained = label == SUSTAINED
    if sustained.any():
        # Rank survivors by amplitude -- these are the candidates stage 2 should
        # actually spend time on.
        idx   = torch.nonzero(sustained)
        order = torch.argsort(amp[sustained], descending=True)
        print("\nTop sustained-oscillation cells by amplitude:")
        for k in order[:15]:
            ii = idx[k].tolist()
            coords = "  ".join(f"{a}={axes[a][x]:7.2f}" for a, x in zip(AXES, ii))
            print(f"  {coords} | amp={amp[tuple(ii)]:.3e}/ms  "
                  f"rate={stats['mean_rate'][tuple(ii)]:.3e}/ms  CV={cv[tuple(ii)]:.3f}")
    else:
        print("\nNo sustained oscillations found -- widen the sweep.")

    # ── Pairwise projections ──────────────────────────────────────────────────
    # 4D can't be imshow'd. For each of the 6 axis pairs, marginalize the other
    # two: the fraction of cells that sustain says where in the plane the
    # oscillatory region lives, and how thick it is in the collapsed directions.
    pairs = list(combinations(range(4), 2))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    for ax, (a, b) in zip(axs.ravel(), pairs):
        others = tuple(d for d in range(4) if d not in (a, b))
        frac = sustained.to(torch.float64).mean(dim=others)
        if a > b: frac = frac.T     # mean() keeps ascending dim order; put a on y
        xa, ya = axes[AXES[b]], axes[AXES[a]]
        # Magnitude on a bounded 0-1 scale -> sequential, single hue, fixed limits
        # so all six panels are directly comparable.
        im = ax.imshow(frac, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       extent=(xa.min().item(), xa.max().item(), ya.min().item(), ya.max().item()))
        ax.set_xlabel(AXIS_LABELS[AXES[b]])
        ax.set_ylabel(AXIS_LABELS[AXES[a]])
        fig.colorbar(im, ax=ax, label='fraction sustained')

    fig.suptitle('Sustained-oscillation fraction, pairwise projections')
    fig.tight_layout()
    plt.show()
