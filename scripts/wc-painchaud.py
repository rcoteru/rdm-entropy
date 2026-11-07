"""
Painchaud, Doyon & Desrosiers (2022), arXiv:2204.00583, ported onto this model.

That paper puts the refractory fraction back into Wilson-Cowan as a dynamical variable rather
than slaving it to the active fraction, and gets two things the classical model cannot give:
a limit cycle in a *single* excitatory population (their 4.1) and chaos in a pair of *two*
excitatory populations (their 4.2, correlation dimension 2.17, lambda_1 ~ +0.16 per 1/gamma_1).
No inhibition anywhere. This script asks whether the same two things happen here, where
refractoriness is not a discrete state but the kernel R(u) = -K exp(-u/tau_ref) riding on the
age variable.

The translation, and what it cannot do
--------------------------------------
Their neuron is a three-state Markov chain 0 -> 1 -> i -> 0: a sensitive neuron activates at
rate alpha_J F_J(B_J), an active one decays to refractory at beta_J, a refractory one returns
to sensitive at gamma_J. F_J is a sigmoid of threshold theta_J and width s_J, and the input
B_J = sum_K c_JK A_K + Q_J is built from the ACTIVE FRACTIONS A_K, not from rates.

Here a spike is instantaneous -- there is no active state with a dwell time -- and a neuron's
field is h(u) = S(u) + R(u) - theta with S the filtered input. Four of the five parameter
groups carry over exactly, one does not, and one has no counterpart at all:

  Time unit.  Their rates are quoted in units of gamma_1, ours in 1/ms. The anchor is alpha:
  the maximal firing rate of a recovered neuron is alpha_J there and lam0 = 1/DT0_ANCHOR here, so
  alpha_1 = lam0 fixes

      T_u = alpha_1 * DT0_ANCHOR  ms per unit of 1/gamma_1,

  which at the DT0_ANCHOR = 0.5 asked for is T_u = 6.25 ms. Every one of their rates is divided by
  T_u to get 1/ms. Note the direction of the constraint: DT0_ANCHOR is not free once alpha_1 is, and
  a different DT0_ANCHOR is a different physical time unit, not a refinement.

  Weights.  A_K = r_K/beta_K relates their active fraction to a firing rate, so their
  dimensionless c_JK becomes our w_JK = c_JK / beta_K with beta_K in 1/ms -- the beta of the
  SOURCE population, which is the easy sign of the thing to get backwards. Thresholds and
  sigmoid widths then transfer unchanged, theta_J -> theta_J and beta_gain_J = 1/s_J, because
  w*r and B are the same dimensionless input. Q_J -> E_J, zero in both of their examples.

  alpha_2 != alpha_1 cannot be represented.  lam0 is a property of the model, not of a
  population, so both populations here saturate at the same maximal rate. Their pair has
  alpha_2/alpha_1 = 0.29. This is the one structural mismatch, and it is why the thresholds
  and the weight scale are swept rather than taken at face value: the second population's
  operating point has to be put back by hand.

  Refractoriness is soft here and hard there.  Their neuron cannot fire at all for a mean time
  1/beta_J + 1/gamma_J and then fires at full rate; ours is suppressed by a kernel that decays
  smoothly and never quite vanishes. Matching the mean dwell time directly does not work -- it
  was tried first and lands the single population on a low-rate fixed point with A ~ 0.03
  against their A* ~ 0.21, because the kernel's tail keeps suppressing long after the nominal
  dwell time is over. What is matched instead is the time at which the kernel stops mattering,
  i.e. falls below the sigmoid's own width 1/beta_gain:

      tau_ref_J = (1/beta_J + 1/gamma_J) * T_u / ln(K_ref * beta_gain_J).

  That is a calibration, not an identity, so tau_ref carries a sweep multiplier and K_ref is
  an axis in its own right.

  The age grid is sized by the RATE, not by the kernels.  q_from_tau truncates where the
  slowest kernel has decayed, which is the right rule when the firing rate is high enough that
  little mass survives to that age. It is not the rule here: these cells fire at ~0.06/ms, an
  interval of ~16 ms against a tau_ref of ~2.5 ms, so mass reaches the absorbing last bin and
  is frozen there with the refractory kernel at whatever value that age happens to give. The
  effect is not subtle -- at the nominal translation, U_max = 12 ms (the kernel rule at
  eps = 0.01) gives a FIXED POINT at r = 1.8e-2/ms, and U_max = 24 ms gives a limit cycle at
  r = 6.1e-2/ms with CV 1.65, converged to five digits against 48, 96 and 192 ms. So U_max is
  sized per cell by kernels.q_from_tau_and_rate, from a rate measured in a short pilot pass.
  Validated against a flat grid on the whole dt = 0.5 sweep: 99.4% of cells get the same label
  and the median rate agrees to 1e-5, on a mean Qm of 92 bins against 200, i.e. less than half
  the age grid. The disagreements are almost entirely in the step-artifact class, where the
  trajectory is a numerical 2-cycle and nothing is well defined anyway.

  tau_int has no counterpart.  Their input is instantaneous. Ours is filtered, and the filter
  cannot be switched off, so tau_int is held at a value small compared to every other time
  constant and is a convergence parameter rather than a model parameter.

What the sweep is over
----------------------
Not the translated point, which is a single cell and is on the wrong side of the mismatches
above. The axes are the four quantities the translation cannot pin:

  kappa       overall scale on the c matrix -- absorbs the alpha_2 mismatch and the rate
              deficit the soft kernel leaves behind. A single-population probe at the pop-1
              parameters puts the oscillatory band at w ~ 19-36 against a translated w = 16.7,
              so kappa ~ 1 is the lower edge rather than the centre.
  K_ref       refractory amplitude. Has no translated value at all: their refractory state is
              absolute, so any K large enough to suppress firing is "the same" model, and it
              is only the finite-K tail that makes it an axis.
  tau_scale   multiplier on the per-population tau_ref above. Carries their beta/gamma
              asymmetry (pop 1: 1.33, pop 2: 1.375 in units of 1/gamma_1 -- nearly equal,
              which is why one scalar multiplier is enough).
  cross       multiplier on the OFF-DIAGONAL c entries, 0.6 and 0.01. Their chaos is two
              self-oscillating populations coupled weakly and very asymmetrically; this is the
              axis that says how weakly, and the one with no reason to survive translation
              since it is a ratio of quantities the other three axes rescale.

Indicators
----------
Stage 1 of wc-search.py asks "does the activity stop moving", which separates a fixed point
from a cycle and nothing else. Chaos is a question about a cycle, so two indicators are added,
both streamed in O(B):

  Burst statistics.  A second pass over the same window thresholds the rate at the midpoint of
  its own range and emits one peak per excursion. The dispersion of the peak HEIGHTS and of the
  inter-peak INTERVALS is what separates their Fig. 7 (bottom) from their Fig. 12: a locked
  cycle repeats both to machine precision, an aperiodic orbit does not. Split in half like
  everything else here, so a slowly-decaying transient is not read as irregularity.

  Step-level alternation.  DT0_ANCHOR = 0.5 with dt = 0.5 puts lam0*dt = 1, where the async hazard
  coincides with the synchronous one and a saturated population can flip-flop between firing
  everything and firing nothing on alternate steps. That has CV ~ 1 and a huge amplitude and
  would otherwise be the loudest "oscillation" on the grid. It is a discretization artifact,
  not a solution, and the lag-1 autocorrelation identifies it exactly: at -1 the trajectory is
  a 2-cycle of the map. Cells flagged this way should be re-run at dt < DT0_ANCHOR before anything is
  believed about them.

A positive aperiodicity flag is a candidate, not a result, and stage 2 below is what turns one
into the other. The burst dispersion is necessary but very far from sufficient: a periodic
orbit whose period spans SEVERAL bursts repeats a pattern of unequal heights, which is exactly
what an irregular train looks like to a statistic that never compares a peak to the one a
period earlier. Only the spectrum separates the two.

What this found
---------------
Run as it stands, on a 9**4 grid:

  Stage 0 reproduces their section 4.1. Both translated populations oscillate on their own, by
  refractoriness alone, with no inhibition and no second population -- and the oscillatory band
  starts at kappa ~ 1, i.e. essentially at the nominal translation, for both. Population 2
  needs only kappa ~ 0.7. That is the one part of the port that comes out better than it had
  any right to.

  Stage 1 flags 878 of 6561 cells as aperiodic, against 577 periodic and 785 fixed points.

  Stage 2 confirms none of them. The twelve strongest candidates were run at dt = 0.5, 0.25 and
  0.125 (DT0_ANCHOR pinned at 0.5 throughout, so these are refinements of ONE model and not a walk
  across a family). At dt = 0.5 most have lambda_1 clearly negative -- down to -3.6e-2 /ms at
  the top-ranked cell, which therefore has a burst-height dispersion of 0.32 while being a
  perfectly periodic orbit, and is the multi-burst-period false positive above caught in the
  act. One cell comes back lambda_1 = +1.6e-2 at dt = 0.25 and -2.3e-4 at dt = 0.125. At the
  finest step every one of the twelve has |lambda_1| < 5e-4 /ms, within its own error bar,
  which puts them on limit cycles or invariant circles and not on strange attractors.

  So: their 4.1 ports over, their 4.2 does not -- or at least not here, on this grid, at this
  resolution. Given that lambda_1 cannot be resolved below ~1e-3 /ms, "no chaos found" remains
  a weaker claim than "no chaos exists", and their own lambda_1 = +0.16 per 1/gamma_1 is
  +2.6e-2 /ms in these units, which would be comfortably resolvable if it were here.

  A caveat that limits the whole sweep: 4286 of 6561 cells -- 65% -- are step artifacts. At
  dt = DT0_ANCHOR = 0.5 the async hazard is the synchronous one, and most of the strong-coupling half
  of the grid saturates into a 2-cycle of the map. The resolved region is the weak-coupling
  side, and a sweep that wants the other half has to be run at dt < DT0_ANCHOR from the start.
"""

# ── NOT RUNNABLE (2026-11-07) ─────────────────────────────────────────────────
#
# Ported to the route-B API mechanically and nothing more: `dt0` is gone, the dead hazard keys
# are mapped, and the model builds. It is NOT wired to a preset and has NOT been re-run, because
# the two-population presets it needs (`EI`, `EE-P`) are still `PENDING` in rdme.presets and
# their values are a vault decision that has not been made. Running it produces numbers that
# belong to no named parameter set, so do not.
#
# What the port changed, and why it is a change of MODEL and not a rename:
#   `dt0`         removed. There is no time quantum under route B; `nu0` is the rate unit of
#                 the Poisson reference and `t_abs` the physical dead time.
#   "async"    -> "poisson_exact". The route-B kernel with async's per-bin FORM is "bernoulli"
#                 (nu0*dt*sigma), not this one; the 2026-11-07 brief asks for the solver here
#                 instead, so the rate function changes from saturating to exponential.
#   "sync"/"synchronous"
#              -> "bernoulli", which at nu0*dt = 1 is the sigmoid bit for bit.
#
#   the TIME-UNIT ANCHOR is unresolved. This script fixes its physical time unit by
#                 T_u = alpha_1 * dt0, i.e. by identifying Painchaud's maximal rate with the
#                 old lam0 = 1/dt0 RATE CEILING. Route B has no rate ceiling: the hazard is
#                 exponential and the only cap is 1/t_abs. The arithmetic below is left
#                 EXACTLY as it was and the old dt0 survives only as DT0_ANCHOR, a number
#                 feeding T_u; it is no longer passed to any model. Choosing the new anchor
#                 is a vault decision and is what blocks the EE-P preset.
# ──────────────────────────────────────────────────────────────────────────────

import math
from itertools import combinations
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import tqdm

import rdme.kernels as krn
import rdme.lyapunov as lyap
import rdme.plots as rp
from rdme.rdm_batch import RDMWilsonCowanPoints, flatten_param_grid


# ── Cache paths ───────────────────────────────────────────────────────────────

bname = Path(__file__).stem
CACHE_DIR = Path(__file__).parents[1] / "cache"

# dt and DT0_ANCHOR are both in the cache name. DT0_ANCHOR because it selects the MODEL -- it fixes the time
# unit through T_u = alpha_1*DT0_ANCHOR, so two DT0_ANCHOR are two different ports of the paper and must not
# share a file. dt because it is the resolution, and the whole point of varying it is to put
# the two answers side by side rather than to overwrite one with the other.
_tag = lambda: f"{bname}_dt{dt:g}_dt0{DT0_ANCHOR:g}" + (f"_tabs{T_ABS:g}" if T_ABS else "")

run_probe = True    # stage 0: each translated population on its own
run_sim   = True
run_lyap  = True    # stage 2: Benettin spectra of the flagged candidates, at several dt
run_plot  = True
overwrite = False

torch.set_default_dtype(torch.float64)


# ── The paper's parameters ────────────────────────────────────────────────────
# Section 4.2, equations (17a)-(17c): two excitatory populations. Rates in units of gamma_1,
# c indexed [target][source] as their B_J = sum_K c_JK A_K.

P_ALPHA = (12.5, 3.6)
P_BETA  = (3.0,  8.0)
P_GAMMA = (1.0,  0.8)
P_THETA = (2.0,  0.84)
P_S     = (0.4,  0.2)
P_Q     = (0.0,  0.0)
P_C     = ((8.0, 0.6),
           (0.01, 14.0))


def translate(dt0_anchor: float, K_ref: float, kappa: float = 1.0,
              tau_scale: float = 1.0, cross: float = 1.0) -> dict:
    """ The paper's parameters in this model's units; see the module docstring for the
    derivation of each line.

    kappa scales the whole weight matrix, cross only its off-diagonal, and tau_scale the
    calibrated refractory time constants. All three are 1 at the nominal translation, which is
    a point on the grid rather than the answer. """
    T_u   = P_ALPHA[0] * dt0_anchor                           # ms per unit of 1/gamma_1
    beta  = [b / T_u for b in P_BETA]                         # 1/ms
    gamma = [g / T_u for g in P_GAMMA]
    bgain = [1.0 / s for s in P_S]

    # w[target][source] = c[target][source] / beta[source]: the source population's beta, since
    # it is the source's active fraction A_K = r_K/beta_K that the weight multiplies.
    scale = ((kappa, kappa * cross),
             (kappa * cross, kappa))
    w = [[P_C[j][k] * scale[j][k] / beta[k] for k in range(2)] for j in range(2)]

    # Dead time matched where the kernel falls below the sigmoid's width, not at its mean.
    tau_ref = [tau_scale * (1.0 / beta[j] + 1.0 / gamma[j]) / math.log(K_ref * bgain[j])
               for j in range(2)]

    return dict(T_u=T_u, beta_phys=beta, gamma_phys=gamma,
                # this model's names: w_XY is the X -> Y weight, i.e. w[target Y][source X]
                w_EE=w[0][0], w_IE=w[0][1], w_EI=w[1][0], w_II=w[1][1],
                beta_E=bgain[0], beta_I=bgain[1],
                theta_E=P_THETA[0], theta_I=P_THETA[1],
                E_exc=P_Q[0], E_inh=P_Q[1],
                tau_ref_E=tau_ref[0], tau_ref_I=tau_ref[1])


def translate_columns(kappa: torch.Tensor, K_ref: torch.Tensor,
                     tau_scale: torch.Tensor, cross: torch.Tensor) -> dict:
    """ `translate`, vectorised over flat (B,) search columns.

    The same arithmetic, not a second copy of the model: it is separate only because the search
    axes are not the model's parameters -- tau_ref is a function of two axes at once and each
    weight of two others -- so there is nothing for an outer product to do and the columns have
    to be zipped. This is why the sweep goes through RDMWilsonCowanPoints and not
    RDMWilsonCowanBatch; see that factory's docstring. """
    T_u   = P_ALPHA[0] * DT0_ANCHOR
    bphys = [b / T_u for b in P_BETA]
    bgain = [1.0 / s for s in P_S]
    return dict(
        w_EE = kappa * P_C[0][0] / bphys[0],
        w_IE = kappa * cross * P_C[0][1] / bphys[1],
        w_EI = kappa * cross * P_C[1][0] / bphys[0],
        w_II = kappa * P_C[1][1] / bphys[1],
        tau_ref_E = tau_scale * (1/P_BETA[0] + 1/P_GAMMA[0]) * T_u / torch.log(K_ref*bgain[0]),
        tau_ref_I = tau_scale * (1/P_BETA[1] + 1/P_GAMMA[1]) * T_u / torch.log(K_ref*bgain[1]),
    )


# The absolute refractory window is a model parameter now (t_abs on every constructor), not
# something this script patches into mf.R afterwards. The old note here argued that threading
# t_abs through RDMNetwork, RDMNetworkBatch and the factories "buys nothing this script needs";
# it buys two things:
#
#   kernels.refractory_kernel_abs blocked round(t_abs/dt) bins, one too many -- a neuron
#   entering bin 0 fires at the NEXT step, so n blocked bins buy a dead time of (n+1)*dt. The
#   realised dead time was T_ABS + dt and moved with dt. These runs therefore differ from the
#   cached ones by one bin.
#
#   The spin model has no R(u) to patch, so it had to be done a second way there, and its EPR
#   rebuilds the field without going through field(). One parameter, one implementation.
#
# R_abs is sized per element by kernels.abs_refractory_floor, which is the same bound the
# `abs_floor` helper here computed: with the window in place no population exceeds r = 1/T_ABS,
# so the drive is bounded by (incoming weights)/T_ABS + |Q| and the floor clears it by
# M_SUPPRESS/beta.


# ── Fixed parameters ──────────────────────────────────────────────────────────

# dt is the solver step. DT0_ANCHOR is the old time quantum, kept ONLY as the number that
# fixes the physical time unit through T_u = alpha_1 * DT0_ANCHOR below: the anchor identified
# Painchaud's maximal rate with the route-A ceiling lam0 = 1/DT0_ANCHOR, which route B does not have.
# UNRESOLVED -- a vault decision. It is no longer passed to any model.
dt, DT0_ANCHOR = 0.1, 0.5
HAZARD  = "poisson_exact"
# The old ceiling, kept because the saturation diagnostics below are written against it. It is
# NOT nu0: nu0 is the rate unit of the Poisson reference and is the repo's 1 /ms.
LAM0    = 1.0 / DT0_ANCHOR
NU0     = 1.0

# 1000 neurons per population, as in their Figs. 11 and 14. Structural only: N sets the
# N-weighted `activity` observable and nothing in the dynamics.
E_RATIO = 0.5

# No counterpart in their model, which has no synaptic filter. Small compared to the fastest
# translated time constant (tau_ref ~ 2 ms) so the filter is not a third timescale of its own.
TAU_INT = 1.0

K_REF_I_ON = True   # both populations refractory; theirs are, and the asymmetry is in tau

# Absolute refractory period, in ms, imposed as a hard floor on the first round(T_ABS/dt) age
# bins via the t_abs argument of kernels.refractory_kernel. Set to 0 to recover the plain
# exponential kernel.
#
# Without it the exponential kernel is a purely RELATIVE mechanism and a strong enough drive
# fires straight through it: on the dt = 0.1 sweep the median fixed point sat at 1264 Hz with
# an effective dead time of ~0.3 ms, against the 2.5-22 ms the kernel nominally imposes. That
# is not a rate any neuron sustains, and it is the corner that was pure step artifact at
# dt = 0.5 -- so two thirds of that grid was in a regime with no physiological counterpart.
#
# 1 ms is the short end of the cortical range (1-2 ms); with lam0 = 1/DT0_ANCHOR the two together cap
# the sustained rate at 1/(T_ABS + DT0_ANCHOR) = 667 Hz at the current DT0_ANCHOR = 0.5 ms. Both mechanisms
# contribute, which is worth stating rather than hiding: T_ABS is recovery, DT0_ANCHOR is how fast a
# recovered neuron can respond at all.
T_ABS = 1.0
# How far below threshold the field is held inside the window, in units of 1/beta. 40 puts the
# hazard at sigma(-40) ~ 4e-18, i.e. zero for every purpose here and still finite -- see
# kernels.abs_refractory_floor on why -inf is not an option.
M_SUPPRESS = 40.0

CACHE_STAT_FILE = CACHE_DIR / f"{_tag()}_stats.pt"
CACHE_LYAP_FILE = CACHE_DIR / f"{_tag()}_lyap.pt"

# Age grid. The kernel rule q_from_tau is the wrong rule here -- see the module docstring --
# so the grid is sized per cell from the MEASURED firing rate, via kernels.q_from_tau_and_rate.
# That needs a rate, which is an output, so a short pilot pass supplies one.
EPS_Q       = 1e-4     # tolerance on (mass beyond U_max) x (field error there)
U_MAX_PILOT = 100.0    # ms; the pilot's own grid, generous because its only job is a rate
PILOT_MS    = 2000.0   # settling before the pilot rate is read
PILOT_WIN   = 1000.0   # averaging window for it
# Ceiling, in ms. A cell that falls silent has no finite grid by the mass rule, and the cap is
# exactly where that is harmless: with no firing the kernels have long since saturated at the
# boundary, so the absorbing bin resums its tail exactly and the lumping commits no error.
U_MAX_CAP   = 200.0
# Peak device memory the production chunks may use for (p, S), in bytes. The chunk size is set
# from this and the chunk's own Qm rather than being a constant, since Qm now varies by an
# order of magnitude across the grid and a fixed cell count would OOM on the slow cells and
# waste the device on the fast ones.
STATE_BUDGET = 2 * 2**30


# ── Classification thresholds ─────────────────────────────────────────────────

CV_TOL = 1e-4                  # as wc-search.py: 3e-8 at a fixed point, >1 on a cycle
DECAY_LO, DECAY_HI = 0.9, 1.1  # std(second half)/std(first half)

# Lag-1 autocorrelation below this is a 2-cycle of the map, not a trajectory. Measured -1.000
# on the saturated cells and > -0.9 on every genuine burst train (whose period is tens of steps).
ALT_TOL = -0.90
# Firing rate within this fraction of lam0 is saturated, the other face of the same artifact.
SAT_FRAC = 0.95

# Dispersion of burst heights above which a train is called irregular. A locked cycle of this
# model repeats its peaks to ~1e-10; the threshold is far above that and far below the ~0.1
# their chaotic trajectory shows, so it is not a delicate number.
IRREG_TOL = 0.02

# Stage 2. Half-width of the band around zero that lambda_1 cannot be resolved inside, in /ms:
# a finite window gives a statistical floor and the age lattice pins the marginal phase
# direction, both of order 1e-4 here. N_SIGMA_LYAP is what actually decides, the measured
# standard errors being the same size as the tolerance. Both as in wc-lyapunov.py.
LYAP_TOL, N_SIGMA_LYAP = 5e-4, 3.0

FIXED, DECAYING, PERIODIC, APERIODIC, ARTIFACT = 0, 1, 2, 3, 4
CLASS_NAMES = ["fixed point", "transient", "periodic", "aperiodic", "step artifact"]

AXES = ["kappa", "K_ref", "tau_scale", "cross"]
AXIS_LABELS = {"kappa":     "$\\kappa$ (scale on $c$)",
               "K_ref":     "$K_{\\mathrm{ref}}$ (refractory amplitude)",
               "tau_scale": "$\\tau$-scale (on calibrated $\\tau_{\\mathrm{ref}}$)",
               "cross":     "cross-coupling scale (on $c_{12},c_{21}$)"}


# ── Streaming statistics ──────────────────────────────────────────────────────

@torch.inference_mode()
def streaming_stats(mf, steps: int) -> dict[str, torch.Tensor]:
    """ Pass 1: moments, range and lag-1 autocorrelation of the population-1 rate, in O(B).

    Sums are shifted by the first sample, as in wc-search.py: at a settled fixed point
    E[x^2] - E[x]^2 asks float64 to resolve a variance 1e-15 of the mean square, which is its
    whole relative precision, and the shift turns that cancellation from catastrophic to exact.
    The lag-1 product is accumulated on the shifted variable too, so the covariance it feeds is
    the same difference of large numbers made small. """
    half = steps // 2
    n1, n2 = half, steps - half
    zeros = lambda: torch.zeros(mf.B, device=mf.device)

    s1, q1, s2, q2, c1 = zeros(), zeros(), zeros(), zeros(), zeros()
    lo = torch.full((mf.B,), float('inf'),  device=mf.device)
    hi = torch.full((mf.B,), float('-inf'), device=mf.device)
    shift = mf.m[:, 0].clone()
    prev = None

    for t in tqdm.tqdm(range(steps), desc="pass 1", leave=False):
        x = mf.m[:, 0]
        lo = torch.minimum(lo, x)
        hi = torch.maximum(hi, x)
        d = x - shift
        if t < half: s1 += d; q1 += d * d
        else:        s2 += d; q2 += d * d
        if prev is not None: c1 += prev * d
        prev = d
        mf.update()

    var = lambda q, s, n: (q / n - (s / n) ** 2).clamp_min(0.0)
    s, q = s1 + s2, q1 + q2
    v = var(q, s, steps)
    # covariance over the steps-1 available lagged pairs, against the full-window moments;
    # the O(1/steps) inconsistency in the normalization is irrelevant at steps ~ 1e4
    cov = c1 / (steps - 1) - (s / steps) ** 2
    return {
        "mean":  s / steps + shift,
        "std":   v.sqrt(),
        "std_1": var(q1, s1, n1).sqrt(),
        "std_2": var(q2, s2, n2).sqrt(),
        "amp":   hi - lo,
        "lo":    lo,
        "hi":    hi,
        "drift": (s2 / n2 - s1 / n1).abs(),
        "acf1":  cov / v.clamp_min(1e-300),
    }


@torch.inference_mode()
def streaming_peaks(mf, steps: int, thr: torch.Tensor) -> dict[str, torch.Tensor]:
    """ Pass 2: one peak per excursion of the rate above `thr`, with the height and the
    interval since the previous peak accumulated per half-window, in O(B).

    Thresholded excursions rather than local maxima. A local-maximum test fires on every wiggle
    -- a burst with a shoulder contributes two peaks and the height dispersion picks up the
    shoulder rather than the irregularity -- whereas an excursion is emitted once, on the way
    down, and is insensitive to structure inside the burst. The threshold is the midpoint of
    the cell's own measured range, so it scales with the orbit instead of being a rate.

    A peak is attributed to the half in which it CLOSES, and the first peak of the run
    contributes a height but no interval, there being nothing before it. """
    half = steps // 2
    zeros = lambda: torch.zeros(mf.B, device=mf.device)
    b = lambda: torch.zeros(mf.B, dtype=torch.bool, device=mf.device)

    armed  = b()
    curmax = zeros()
    last   = torch.full((mf.B,), -1.0, device=mf.device)
    n  = [zeros(), zeros()]                       # peak counts, per half
    hs = [zeros(), zeros()]; hq = [zeros(), zeros()]   # height sums, sums of squares
    isn = [zeros(), zeros()]
    iss = [zeros(), zeros()]; isq = [zeros(), zeros()]  # interval sums, sums of squares

    for t in tqdm.tqdm(range(steps), desc="pass 2", leave=False):
        x = mf.m[:, 0]
        above = x > thr
        # track the running maximum only while above, restarting it on each up-crossing
        curmax = torch.where(above, torch.where(armed, torch.maximum(curmax, x), x), curmax)
        close = armed & ~above
        h = 0 if t < half else 1
        n[h]  += close
        hs[h] += torch.where(close, curmax, torch.zeros(()).to(curmax))
        hq[h] += torch.where(close, curmax * curmax, torch.zeros(()).to(curmax))
        gap = float(t) - last
        ok = close & (last >= 0)
        isn[h] += ok
        iss[h] += torch.where(ok, gap, torch.zeros(()).to(gap))
        isq[h] += torch.where(ok, gap * gap, torch.zeros(()).to(gap))
        last = torch.where(close, torch.full_like(last, float(t)), last)
        armed = above
        mf.update()

    def disp(ns, ss, qs):
        """ std/mean of a quantity accumulated as (count, sum, sum of squares). """
        nt, st, qt = ns[0] + ns[1], ss[0] + ss[1], qs[0] + qs[1]
        safe = nt.clamp_min(1.0)
        mu = st / safe
        sd = (qt / safe - mu * mu).clamp_min(0.0).sqrt()
        out = sd / mu.abs().clamp_min(1e-300)
        return torch.where(nt >= 4, out, torch.full_like(out, float('nan'))), mu, nt

    h_cv, h_mean, n_pk = disp(n, hs, hq)
    i_cv, i_mean, n_is = disp(isn, iss, isq)
    # Per-half height dispersion: an irregular train is irregular in both halves, a decaying
    # one is not, and that is the same question the std_2/std_1 ratio asks of the raw signal.
    def half_cv(j):
        safe = n[j].clamp_min(1.0)
        mu = hs[j] / safe
        sd = (hq[j] / safe - mu * mu).clamp_min(0.0).sqrt()
        return torch.where(n[j] >= 2, sd / mu.abs().clamp_min(1e-300),
                           torch.full_like(mu, float('nan')))
    return {"pk_n": n_pk, "pk_height": h_mean, "pk_height_cv": h_cv,
            "pk_height_cv_1": half_cv(0), "pk_height_cv_2": half_cv(1),
            "isi_n": n_is, "isi_mean": i_mean, "isi_cv": i_cv}


# ── Stage 0: the translated populations on their own ──────────────────────────

if run_probe:
    from rdme.rdm_batch import RDMNetworkBatch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("── Stage 0: each translated population alone ──────────────────────────")
    print(f"dt = {dt}, DT0_ANCHOR = {DT0_ANCHOR}, hazard = {HAZARD!r}, lam0 = {LAM0} /ms")
    nom = translate(DT0_ANCHOR, K_ref=10.0)
    print(f"Time unit T_u = {nom['T_u']} ms per 1/gamma_1 "
          f"(from alpha_1 = {P_ALPHA[0]} = lam0); alpha_2/alpha_1 = "
          f"{P_ALPHA[1]/P_ALPHA[0]:.3f} is NOT representable.")
    print(f"Nominal weights   w_EE={nom['w_EE']:.2f}  w_IE={nom['w_IE']:.3f}  "
          f"w_EI={nom['w_EI']:.4f}  w_II={nom['w_II']:.2f}")
    print(f"Nominal tau_ref   pop1={nom['tau_ref_E']:.2f} ms  pop2={nom['tau_ref_I']:.2f} ms "
          f"(at K_ref = 10)")

    # Each population with its own self-coupling only, over (kappa, K_ref): the question their
    # section 4.1 answers, and the prerequisite for 4.2 -- two populations that do not oscillate
    # on their own are not going to be coupled into chaos.
    nk, nK = 9, 7
    kap = torch.linspace(0.4, 2.8, nk)
    Kr  = torch.logspace(math.log10(2.0), math.log10(30.0), nK)
    for j in (0, 1):
        bg, th = 1.0 / P_S[j], P_THETA[j]
        T_u = P_ALPHA[0] * DT0_ANCHOR
        w0  = P_C[j][j] / (P_BETA[j] / T_u)
        td  = (1.0 / P_BETA[j] + 1.0 / P_GAMMA[j]) * T_u
        # Flattened by hand, and RDMNetworkBatch rather than RDMIsingModelBatch: the latter
        # takes the OUTER PRODUCT of its arguments, and tau_ref is a function of K_ref, so the
        # two have to be zipped. Through the factory this silently gives B**3 cells, most of
        # them off the intended sweep -- the same trap RDMWilsonCowanPoints exists to avoid.
        kk, KK = torch.meshgrid(kap, Kr, indexing='ij')
        kk, KK = kk.reshape(-1), KK.reshape(-1)
        tr = td / torch.log(KK * bg)
        # stage 0 keeps a fixed generous grid: it is 63 cells, the saving is not worth the
        # pilot, and U_MAX_PILOT is already past where these cells converge
        Qm = [max(krn.q_from_tau(max(tr.max().item(), TAU_INT), dt, 0.01),
                  math.ceil(U_MAX_PILOT / dt))]
        col = lambda t: torch.as_tensor(t).reshape(-1, 1).expand(kk.numel(), 1).clone()
        mf = RDMNetworkBatch(
            M=1, w=(kk * w0).reshape(-1, 1, 1), N=[1000], E=col(0.0), beta=col(bg),
            theta=col(th), tau_int=col(TAU_INT), tau_ref=tr.reshape(-1, 1),
            K_ref=KK.reshape(-1, 1), dt=dt, nu0=NU0, hazard=HAZARD, t_abs=T_ABS,
            Qm=Qm, device=device)
        mf.forward(int(4000 / dt), pb=False)
        st = streaming_stats(mf, int(1500 / dt))
        st = {k: v.cpu() for k, v in st.items()}
        cv  = (st["std"] / st["mean"].clamp_min(1e-300)).reshape(nk, nK)
        art = ((st["acf1"] < ALT_TOL) | (st["hi"] / dt > SAT_FRAC * LAM0)).reshape(nk, nK)
        live = (cv > CV_TOL) & ~art
        print(f"\n  population {j+1}: w_self(kappa=1) = {w0:.2f}, dead time {td:.2f} ms, "
              f"Qm = {Qm[0]}")
        print("    CV of r_1, '*' = step artifact;  rows kappa, cols K_ref")
        print("            " + " ".join(f"{k:7.1f}" for k in Kr))
        for i in range(nk):
            cells = " ".join(("      *" if art[i, c] else f"{cv[i, c]:7.3f}") for c in range(nK))
            print(f"    {kap[i]:5.2f} | {cells}")
        print(f"    oscillating and not an artifact: {int(live.sum())}/{live.numel()} cells")
        del mf
    print()


# ── Simulation ────────────────────────────────────────────────────────────────

if run_sim:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n = 9   # per axis; cost is n**4
    kappa     = torch.linspace(0.4, 2.8, n)
    K_ref     = torch.logspace(math.log10(2.0), math.log10(30.0), n)
    tau_scale = torch.linspace(0.3, 2.7, n)
    cross     = torch.logspace(math.log10(0.2), math.log10(5.0), n)

    # In MILLISECONDS, converted to steps below: a step count held fixed across dt would be a
    # different physical window at each resolution, which is the one thing a convergence check
    # must not let happen.
    equi_ms, rec_ms = 8000.0, 4000.0
    equi, steps = int(equi_ms / dt), int(rec_ms / dt)

    if CACHE_STAT_FILE.exists() and not overwrite:
        print(f"Statistics already exist at {CACHE_STAT_FILE}. Skipping.")
    else:
        axes_in = {"kappa": kappa, "K_ref": K_ref, "tau_scale": tau_scale, "cross": cross}
        flat, grid_shape, swept = flatten_param_grid(axes_in)
        B = flat["kappa"].numel()

        # The translation is a function of three of the four axes at once -- tau_ref depends on
        # both K_ref and tau_scale, and the weights on both kappa and cross -- so it is applied
        # to the flat columns here. RDMWilsonCowanPoints zips its arguments rather than taking
        # their outer product, which is exactly what that requires.
        cols = translate_columns(flat["kappa"], flat["K_ref"],
                                 flat["tau_scale"], flat["cross"])
        Kr, bgain = flat["K_ref"], [1.0 / s for s in P_S]
        T_u = P_ALPHA[0] * DT0_ANCHOR

        def build(idx, Qm):
            """ The batch for an arbitrary set of flat indices, on the given age grid.

            Indices rather than a slice: the production pass reorders the grid by how much age
            grid each cell needs, so the cells sharing a batch are no longer contiguous. """
            g = lambda k: cols[k][idx]
            mf = RDMWilsonCowanPoints(
                E_RATIO, g("w_EE"), g("w_EI"), g("w_IE"), g("w_II"),
                P_Q[0], P_Q[1], bgain[0], bgain[1], P_THETA[0], P_THETA[1],
                TAU_INT, TAU_INT, g("tau_ref_E"), g("tau_ref_I"),
                Kr[idx], Kr[idx] if K_REF_I_ON else 0.0 * Kr[idx],
                dt=dt, nu0=NU0, hazard=HAZARD, t_abs=T_ABS, Qm=Qm, device=device)
            return mf

        # ── Pass 0: pilot rates ───────────────────────────────────────────────
        # q_from_tau_and_rate needs a firing rate, which is an output of the very run it is
        # meant to size. The way out is that the dependence is one-sided and SAFE: a pilot on
        # too small a grid under-reports the rate -- in the worst case it reports a fixed point
        # where the converged answer is a cycle -- and a lower rate asks for MORE grid, never
        # less. So the pilot only has to settle, not to be right, and its errors cost time
        # rather than correctness.
        Qp = math.ceil(U_MAX_PILOT / dt)
        pilot_chunk = max(1, min(n ** 3, int(STATE_BUDGET / (8 * 4 * Qp))))
        rates = torch.empty(B, 2)
        print(f"Pilot: {B} cells at Qm = {Qp} (U_max = {U_MAX_PILOT:g} ms), "
              f"{PILOT_MS:g} + {PILOT_WIN:g} ms each.")
        for lo in tqdm.tqdm(range(0, B, pilot_chunk), desc="pilot"):
            idx = torch.arange(lo, min(lo + pilot_chunk, B))
            mf = build(idx, [Qp, Qp])
            mf.forward(int(PILOT_MS / dt), pb=False)
            acc, T = torch.zeros(len(idx), 2, device=device), int(PILOT_WIN / dt)
            for _ in range(T):
                acc += mf.rates()
                mf.update()
            rates[idx] = (acc / T).cpu()
            del mf
            if device == 'cuda': torch.cuda.empty_cache()

        # ── Per-cell age grid ─────────────────────────────────────────────────
        # Two kernels truncate, so two requirements per population: the refractory one, of
        # amplitude K_ref and time constant tau_ref, and the synaptic one, of amplitude |I| and
        # time constant tau_int. Both are measured against that population's gain, since what
        # the hazard sees is beta*h. The cell takes the largest of the four.
        inp = [(cols["w_EE"] * rates[:, 0] + cols["w_IE"] * rates[:, 1] + P_Q[0]).abs(),
               (cols["w_EI"] * rates[:, 0] + cols["w_II"] * rates[:, 1] + P_Q[1]).abs()]
        taur = [cols["tau_ref_E"], cols["tau_ref_I"]]
        Qcap = math.ceil(U_MAX_CAP / dt)
        Qreq = torch.ones(B, dtype=torch.long)
        for j in range(2):
            for i in range(B):
                r = max(rates[i, j].item(), 0.0)
                q = max(
                    krn.q_from_tau_and_rate(taur[j][i].item(), r, dt, eps=EPS_Q,
                                            gain=bgain[j], amplitude=max(Kr[i].item(), 1e-12)),
                    krn.q_from_tau_and_rate(TAU_INT, r, dt, eps=EPS_Q, gain=bgain[j],
                                            amplitude=max(inp[j][i].item(), 1e-12)))
                Qreq[i] = max(int(Qreq[i]), q)
        n_capped = int((Qreq > Qcap).sum())
        Qreq.clamp_(max=Qcap)

        # ── Production, in grid-homogeneous chunks ────────────────────────────
        # Sorted by requirement, because Qm is shared across a batch: left in grid order, one
        # slow cell would drag its whole chunk up to the cap, which is the "a batch is only as
        # cheap as its slowest member" caveat in RDMNetworkBatch. Sorting makes each chunk
        # nearly uniform, so the cheap cells actually get the small grid they asked for.
        order = torch.argsort(Qreq)
        print(f"Age grid from the measured rate (eps = {EPS_Q:g}): Qm per cell "
              f"min {int(Qreq.min())}, median {int(Qreq.median())}, max {int(Qreq.max())} "
              f"(U_max {Qreq.min()*dt:g}-{Qreq.max()*dt:g} ms); {n_capped} capped at {Qcap}. "
              f"A flat grid at the cap costs {Qcap * B / Qreq.sum().item():.1f}x more age grid.")
        print(f"{equi} equilibration + 2 x {steps} recording steps each "
              f"({equi_ms:g} + 2 x {rec_ms:g} ms at dt = {dt:g}).")
        print(f"Running on: {device}")

        t0 = time.time()
        out = {}
        pos, pbar = 0, tqdm.tqdm(total=B, desc="cells")
        while pos < B:
            # grow the chunk while the budget allows, charging every cell the Qm of the
            # largest in it -- which, sorted, is the last one taken
            lo, hi = pos, pos
            while hi < B and hi - lo < n ** 3:
                Q = int(Qreq[order[hi]])
                if hi > lo and 8 * 4 * Q * (hi - lo + 1) > STATE_BUDGET: break
                hi += 1
            idx = order[lo:hi]
            Q = int(Qreq[idx].max())

            mf = build(idx, [Q, Q])
            mf.forward(equi, pb=False)
            st = streaming_stats(mf, steps)
            # Pass 2 continues from where pass 1 stopped rather than re-equilibrating: the state
            # is on the attractor either way, and the peak statistics do not need the same window
            # as the moments, only one of the same length.
            st |= streaming_peaks(mf, steps, thr=0.5 * (st["lo"] + st["hi"]))
            for k, v in st.items():
                if k not in out: out[k] = torch.empty(B, dtype=v.dtype)
                out[k][idx] = v.cpu()
            del mf
            if device == 'cuda': torch.cuda.empty_cache()
            pbar.update(hi - lo)
            pos = hi
        pbar.close()

        stats = {k: v.reshape(grid_shape) for k, v in out.items()}
        stats["Qm_cell"] = Qreq.reshape(grid_shape)
        stats["pilot_rate"] = rates.reshape(*grid_shape, 2)
        print(f"Done in {time.time() - t0:.2f}s")

        # rates, so the numbers mean the same at any dt; CV and the dispersions are already
        # dimensionless and the ISI is converted from steps to ms
        stats["amp_rate"]  = stats.pop("amp") / dt
        stats["mean_rate"] = stats["mean"] / dt
        stats["max_rate"]  = stats["hi"] / dt
        stats["cv"]        = stats.pop("std") / stats["mean"].clamp_min(1e-300)
        stats["drift"]     = stats["drift"] / dt
        stats["isi_ms"]    = stats["isi_mean"] * dt
        stats["axes"]      = swept
        stats["dt"], stats["dt0_anchor"], stats["hazard"] = dt, DT0_ANCHOR, HAZARD
        stats["steps"], stats["equi"], stats["T_u"] = steps, equi, T_u
        stats["tau_int"], stats["E_ratio"] = TAU_INT, E_RATIO
        stats["eps_Q"], stats["U_max_cap"] = EPS_Q, U_MAX_CAP
        stats["t_abs"], stats["m_suppress"] = T_ABS, M_SUPPRESS

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(stats, CACHE_STAT_FILE)
        print(f"Statistics saved to {CACHE_STAT_FILE}.")


# ── Classification ────────────────────────────────────────────────────────────

if run_lyap or run_plot:

    if not CACHE_STAT_FILE.exists():
        raise FileNotFoundError(f"{CACHE_STAT_FILE} not found. Run the simulation first.")

    stats = torch.load(CACHE_STAT_FILE, weights_only=False)
    axes  = stats["axes"]

    cv    = stats["cv"]
    amp   = stats["amp_rate"]
    ratio = stats["std_2"] / stats["std_1"].clamp_min(1e-300)
    drift = stats["drift"]
    hcv   = stats["pk_height_cv"]
    hcv1, hcv2 = stats["pk_height_cv_1"], stats["pk_height_cv_2"]

    # Artifact first: a step-level 2-cycle has a large CV and a sustained spread, so every later
    # test would pass it, and none of them mean anything on a trajectory the map itself invents.
    artifact = (stats["acf1"] < ALT_TOL) | (stats["max_rate"] > SAT_FRAC / stats["dt0_anchor"])

    label = torch.full(cv.shape, FIXED, dtype=torch.long)
    moving = (cv > CV_TOL) & ~artifact
    settled = moving & (ratio >= DECAY_LO) & (ratio <= DECAY_HI) & (drift <= 0.1 * amp)
    label[moving] = DECAYING
    label[settled] = PERIODIC
    # Irregular in both halves, not just on average: a decaying burst train has a large overall
    # height spread and a small one within each half.
    irregular = settled & (hcv > IRREG_TOL) & (hcv1 > IRREG_TOL) & (hcv2 > IRREG_TOL)
    label[irregular] = APERIODIC
    label[artifact] = ARTIFACT

    counts = {CLASS_NAMES[c]: int((label == c).sum()) for c in range(5)}
    print(f"\nCell counts by class ({label.numel()} total): {counts}")

    if irregular.any():
        idx   = torch.nonzero(irregular)
        order = torch.argsort(hcv[irregular], descending=True)
        print("\nAperiodic candidates, by burst-height dispersion:")
        for k in order[:20]:
            ii = tuple(idx[k].tolist())
            coords = "  ".join(f"{a}={axes[a][x]:7.3f}" for a, x in zip(AXES, ii))
            print(f"  {coords} | hCV={hcv[ii]:.3f} ({hcv1[ii]:.3f}/{hcv2[ii]:.3f})  "
                  f"isiCV={stats['isi_cv'][ii]:.3f}  ISI={stats['isi_ms'][ii]:6.1f}ms  "
                  f"rate={stats['mean_rate'][ii]:.3e}/ms  n_pk={int(stats['pk_n'][ii])}")
        print("\nThese are candidates only: re-run them through wc-lyapunov.py and at "
              "dt < DT0_ANCHOR before calling any of them chaos.")
    else:
        print("\nNo aperiodic cells -- every sustained orbit repeats its bursts.")


# ── Stage 2: Lyapunov spectra of the candidates ───────────────────────────────

if run_lyap:

    # The flags, ranked, capped: the spectrum costs several times a state step per tangent
    # direction, so this is a confirmation pass over the top of the ranking and not a second
    # sweep. Ties in the ranking are broken by flat index, which is arbitrary but reproducible.
    N_CAND = 12
    # DT0_ANCHOR is held at its production value while dt is refined, so these are three resolutions
    # of ONE model. Refining DT0_ANCHOR with it would be a walk across a family of different models,
    # which is the mistake this model's two-timescale parametrisation exists to make visible.
    # The production dt and two refinements of it, always as a descending sequence from dt
    # itself: the question stage 2 answers is whether a sign survives refinement, so the
    # comparison has to start at the resolution stage 1 actually ran at.
    DTS = (dt, dt / 2)
    # Windows in ms, as in stage 1 -- a fixed step count would shrink the measurement window
    # with dt and inflate the error bars exactly where the answer is being checked.
    EQUI_MS, ACC_MS, WARM_MS = 8000.0, 10000.0, 1000.0
    k_lyap, N_BLOCKS, CADENCE = 2, 20, 100

    if not irregular.any():
        print("\nStage 2: nothing flagged, nothing to confirm.")
    elif CACHE_LYAP_FILE.exists() and not overwrite:
        print(f"\nSpectra already exist at {CACHE_LYAP_FILE}. Skipping.")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        idx   = torch.nonzero(irregular)
        order = torch.argsort(hcv[irregular], descending=True)[:N_CAND]
        sel   = idx[order]                                        # (N, 4) grid coordinates
        cand  = {a: axes[a][sel[:, i]] for i, a in enumerate(AXES)}

        print(f"\nStage 2: Benettin spectra for the top {len(sel)} candidates, "
              f"k={k_lyap}, at dt = {DTS} (DT0_ANCHOR = {DT0_ANCHOR} throughout).")
        res = {}
        for dt_r in DTS:
            cols = translate_columns(cand["kappa"], cand["K_ref"],
                                     cand["tau_scale"], cand["cross"])
            # U_max held fixed in MILLISECONDS, so the age grid is refined with dt rather than
            # being silently shortened by it -- refining dt at fixed Qm changes the truncation
            # age, which is a different model, not a finer solution of this one.
            Q = math.ceil(U_MAX_PILOT / dt_r)
            mf = RDMWilsonCowanPoints(
                E_RATIO, cols["w_EE"], cols["w_EI"], cols["w_IE"], cols["w_II"],
                P_Q[0], P_Q[1], 1/P_S[0], 1/P_S[1], P_THETA[0], P_THETA[1],
                TAU_INT, TAU_INT, cols["tau_ref_E"], cols["tau_ref_I"],
                cand["K_ref"], cand["K_ref"] if K_REF_I_ON else 0.0 * cand["K_ref"],
                dt=dt_r, nu0=NU0, hazard=HAZARD, t_abs=T_ABS, Qm=[Q, Q], device=device)

            mf.forward(int(EQUI_MS / dt_r), pb=False)  # state only; tangents here are waste
            gen = torch.Generator(device=device).manual_seed(0)
            blocks = mf.init_tangent(k_lyap, generator=gen)
            mf.step_tangent(blocks)                  # annihilates the 2M spurious modes
            lyap.orthonormalize(blocks)
            out = lyap.benettin(mf.step_tangent, blocks, n_steps=int(ACC_MS / dt_r), dt=dt_r,
                                cadence=CADENCE, n_blocks=N_BLOCKS,
                                warmup=int(WARM_MS / dt_r))
            res[dt_r] = {f: getattr(out, f).cpu() for f in ("lam", "se", "drift", "valid")}
            del mf, blocks
            if device == 'cuda': torch.cuda.empty_cache()

            print(f"\n  dt = {dt_r} (lam0*dt = {dt_r/DT0_ANCHOR:.3g}), Qm = {Q}")
            for j in range(len(sel)):
                l1, l2 = res[dt_r]["lam"][j, 0].item(), res[dt_r]["lam"][j, 1].item()
                e1 = res[dt_r]["se"][j, 0].item()
                coords = " ".join(f"{a}={cand[a][j]:6.3f}" for a in AXES)
                chaos = l1 > max(N_SIGMA_LYAP * e1, LYAP_TOL)
                print(f"    {coords} | lam1={l1:+.3e} +- {e1:.1e}  lam2={l2:+.3e}"
                      f"{'   CHAOS' if chaos else ''}")

        CACHE_DIR.mkdir(exist_ok=True)
        torch.save({"cand": cand, "sel": sel, "res": res, "dts": DTS, "dt0_anchor": DT0_ANCHOR,
                    "k": k_lyap, "U_max_ms": U_MAX_PILOT, "tol": LYAP_TOL}, CACHE_LYAP_FILE)
        print(f"\nSpectra saved to {CACHE_LYAP_FILE}.")
        print("A sign that does not survive the finest dt is a resolution artifact, not a "
              "result; see wc-lyapunov.py for why this model's lambda_1 behaves that way.")


# ── Plot ──────────────────────────────────────────────────────────────────────

if run_plot:

    # ── Pairwise projections ──────────────────────────────────────────────────
    # 4D can't be imshow'd. For each axis pair, marginalize the other two: the top row is where
    # sustained oscillation lives at all, the bottom where it is irregular.
    pairs = list(combinations(range(4), 2))
    fig, axs = plt.subplots(2, 6, figsize=(26, 8))

    for col, (a, b) in enumerate(pairs):
        others = tuple(d for d in range(4) if d not in (a, b))
        for row, (mask, name, cmap) in enumerate(
                [((label == PERIODIC) | (label == APERIODIC), 'sustained', 'viridis'),
                 (irregular, 'aperiodic', 'magma')]):
            frac = mask.to(torch.float64).mean(dim=others)
            if a > b: frac = frac.T      # mean() keeps ascending dim order; put a on y
            xa, ya = axes[AXES[b]], axes[AXES[a]]
            ax = axs[row, col]
            im = ax.imshow(frac, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1,
                           extent=(xa.min().item(), xa.max().item(),
                                   ya.min().item(), ya.max().item()))
            ax.set_xlabel(AXIS_LABELS[AXES[b]])
            ax.set_ylabel(AXIS_LABELS[AXES[a]])
            fig.colorbar(im, ax=ax, label=f'fraction {name}')

    fig.suptitle('Painchaud et al. (2022) 4.2 ported here: sustained (top) and '
                 'aperiodic (bottom) fractions, pairwise projections')
    fig.tight_layout()
    rp.show()
