# E–I presets

Two-population (E–I) presets for the mean field (RDM) and the spin network, and how they were
chosen. Work of 2026-11-19, branch `ei-presets`.

**The presets are placed by attractor behaviour along a transition, not by fixed fractions of
the Hopf coupling.** They feed the phase maps, so each one has to show the regime its name
says. Placing them by linear stability alone failed (see "What did not work").

## Model and conventions

- **Neuron:** Schwalger, Deger & Gerstner 2017, Fig. 5. The same neuron for E and I.
  - LIF with exponential escape noise: β = 0.4 /mV (Δ_u = 2.5 mV), θ = 15 mV,
    ν₀ = 0.01 /ms, t_abs = 4 ms, τ_int = 20 ms.
  - No relative refractoriness: K_ref = 0, so τ_ref is unused.
- **Weight naming:** `w_XY` means **X onto Y**. `RDMWilsonCowan` and `SpinWilsonCowan` store
  `[[w_EE, w_IE], [w_EI, w_II]]` with rows as targets. Every weight in the scripts carries its
  arrow, e.g. `w_IE=-1200.0,  # I→E`.
- **Units:** weights are in mV·ms and act on rates in 1/ms, so the stationary synaptic input
  is S = Σ w·r + E. Delta synapses on a leaky membrane give w = τ_m·C·J.
  - 960 = 20 ms × 160 inputs × 0.3 mV, and 1200 = 20 × 40 × 1.5 mV. These per-synapse values
    were recalled from the paper, not checked against its table.
  - At the presets, the inputs land at 12–18 mV against θ = 15 mV.
- **Runs:**
  - Numerics: `poisson_exact`, dt = 0.1 ms, silent start.
  - Times: T_equi = 1000 ms, T_meas = 2000 ms.
  - Spins: N = 20000, E_ratio = 0.8. In the mean field, E_ratio only weights the aggregate.

## The presets

All of them are **Fig. 5 as published** (w(E→E) = w(E→I) = 960, w(I→E) = w(I→I) = −1200
mV·ms, E_exc = E_inh = 24 mV), with only w(E→I) moved.

Along this cut the mean field has one Hopf, at **J_H = 885.98 mV·ms**: 19.91 Hz, with
r_E = 21.75 Hz and r_I = 18.69 Hz at the crossing. The fixed point is unstable **below** J_H.
- The cycle grows continuously from J_H.
- The silent start and a start 1 % off the fixed point agree at every point of the cut, so
  there's no bistability.
- The onset regime is narrow. r_E max passes the 1/t_abs = 250 Hz ceiling near 0.94·J_H, and
  r_max converges in dt only down to about 0.97·J_H. Hence presets 1–3 % from J_H.

| preset | w(E→I) | /J_H | mean-field state |
|---|---|---|---|
| `EI-async` (= `EI-fig5`) | 960 | 1.084 | stable fixed point, μ = −0.0365 /ms, 16.913 Hz both |
| `EI-onset` | 875 | 0.988 | small cycle, r_E 7.3–62 Hz, period 50.0 ms |
| `EI-cycle` | 865 | 0.976 | cycle, r_E 4.9–97 Hz, period 49.8 ms |
| `EI-burst` | 800 | 0.903 | bursts past the ceiling, r_E up to 678 Hz; **not dt-resolved** |

The presets are defined in `scripts/wc_mf_vs_sm.py`; the onset, cycle and burst presets are
copied into `scripts/ei_sigma_order.py`.

### Mean field against spins

Per-population rate r̄ (Hz) and entropy production σ (nats/neuron/s). Mean-field averages run
over whole cycles of r_E; spin averages over the whole 2000 ms.

| preset | MF r̄_E / r̄_I | spin r̄_E / r̄_I | MF σ_E / σ_I | spin σ_E / σ_I | period MF / spins (ms) |
|---|---|---|---|---|---|
| EI-async | 16.913 / 16.913 | 16.959 / 16.941 | −4e−13 (round-off) | 0.191 / 0.150 | n/a |
| EI-onset | 21.855 / 18.722 | 21.859 / 18.727 | 4.77 / 4.38 | 5.11 / 4.62 | 50.0 / 50.0 |
| EI-cycle | 21.957 / 18.737 | 21.933 / 18.712 | 10.24 / 8.58 | 9.60 / 8.08 | 49.8 / 50.0 |
| EI-burst | 23.308 / 18.977 | 23.425 / 18.985 | 52.4 / 34.0 | 53.0 / 34.2 | 47.0 / 23.5* |

\*The Fourier peak of the spin trace is the first harmonic (23.5 ≈ 47.0 / 2). The mean ISI
(1/r̄) is 45.8 ms for E and 53.4 ms for I at onset, against a 50 ms period.

### dt-convergence (`ei_sigma_order.py`, mean field, dt = 0.2 / 0.1 / 0.05)

| preset | period order | σ_E order | r_max,E (Hz) | H_rev remainder E / I (finer halving) | verdict |
|---|---|---|---|---|---|
| EI-onset | 1.00 | 0.86 | 57.6 / 62.2 / 65.0 | −0.23 / +0.057 | resolved |
| EI-cycle | 0.90 | 0.77 | 87.0 / 97.4 / 104.3 | −0.19 / +0.161 | resolved |
| EI-burst | out of range | 0.34 | 428 / 678 / 1021 | +1.06 / +1.90 | **unresolved** |

**The resolution rule.** A preset is unresolved if any of these holds:
- the period's observed order is outside [0.5, 3] or its differences change sign;
- for any population, the H_rev remainder at the finer halving, dH_rev − r̄·ln 2, exceeds
  **5 % of r̄·ln 2** (`REM_TOL`).

The 5 % threshold is for the presets. **It will probably be lowered for the paper.**

For calibration, from the E1 reference:
- The resolved E1 points have small negative remainders.
- E1-30-strong has +10.2 against r̄·ln 2 ≈ 33 (about 30 %), and E1-15-strong has +427.
- EI-onset and EI-cycle carry a small positive remainder in I, 0.4 % and 1.2 % of r̄·ln 2,
  which halves with each halving of dt. A strict sign rule would flag them.

## Collapse check (Fig. 5 as published)

With identical neurons and equal weight rows, both populations receive the same input, so the
mean field must collapse onto one population with J = 960 − 1200 = −240 mV·ms (preset
`I1-fig5-collapsed` in `scripts/ising_mf_vs_sm.py`).
- **Mean field:** r_E = r_I = 16.913 Hz = the single population. σ is round-off (about
  1e−13), 0 cycles, ISI CV 0.3548.
- **Spins:** the single population gives 16.913 Hz with σ = 0.0018 nats/s. The E–I network
  gives 16.96 / 16.94 Hz with σ = 0.19 / 0.15 nats/s, about 100× larger. **Not yet
  explained.** A first check would be how it scales with N.

## What did not work

1. **The originally specified family:** w(I→I) = 0, E_exc = E_inh = 24, w(E→I) from 0 to 3840.
   - There is no crossing. I, driven by E_inh alone (≈ 24.5 Hz), silences E (r_E ≈ 0.003 Hz),
     and that state is stable throughout.
   - A saddle and an unstable focus exist only for w(E→I) < 10.1 and vanish in a saddle-node.
   - Same at E_inh = 22, 20 and 18.
2. **E_inh lowered to 15** (w(I→I) = 0): one Hopf at J_H = 448.52 (20.01 Hz), unstable below.
   - Presets at 2·J_H, 0.9·J_H and 0.75·J_H all reached the **same large burst cycle**:
     period ≈ 27 ms, peaks of 1000–6600 Hz.
   - 2·J_H is bistable: the fixed point is stable and holds when started there.
   - No small cycle exists even at 0.99·J_H, which jumps to the burst within about 4 s. This
     looks subcritical; no continuation of the cycle branch has been done.
   - Both oscillating presets were unresolved in dt.
3. **Linear stability as the placement rule.** Fractions of J_H say nothing about which
   attractor a run reaches or how wide the onset regime is. On the Fig. 5 cut, 0.9·J_H and
   0.75·J_H already lie in the unresolved bursts.

## How they were found

1. **Coarse attractor map** (mean field, GPU batch via `RDMWilsonCowanBatch`; about 10 s per
   126 cells).
   - Grid: E_inh ∈ {0, 4, …, 24} × w(E→I) ∈ {0, 240, …, 1920} × w(I→I) ∈ {0, −1200}.
   - Two starts per cell: silent, and 1 % off the lowest-r_E fixed point.
   - Result: no small cycle anywhere with w(I→I) = 0, and bistable edges along the boundary.
     With w(I→I) = −1200 the transition is single-valued.
2. **Fine cuts along w(E→I)** through the softest spots:
   - w(I→I) = 0, E_inh = 8: bistable over two windows, rejected.
   - w(I→I) = −1200, E_inh = 20: gradual, onset near 990.
   - **w(I→I) = −1200, E_inh = 24 (Fig. 5): gradual, onset near 886, no bistability.**
     Chosen, because it runs through the published point.
3. **J_H by bisection on the leading growth rate**, then r_max, r̄ and the period at dt = 0.2 /
   0.1 / 0.05 for w(E→I) from 884 down to 770. The presets were picked from those.
4. **Bifurcation scan** over the whole cut (`ei_bifurcations.py`, w(E→I) from 0 to 3840, 81 points):
   - One fixed point everywhere.
   - One crossing: a Hopf at 885.970 (19.907 Hz), unstable below.
   - At w(E→I) = 0 the fixed point is an unstable focus (r_E = 112.0 Hz, μ = +0.271 /ms).
     Above J_H it is stable up to 3840 (μ between −0.052 and −0.056 /ms), and the
     r·t_abs = 0.5 cap is never reached.

## Tooling: `rdme.stability`

Everything about fixed points and their stability lives in `src/rdme/stability.py`, one section
per question. The stationary closed forms and the overlap map F are in `rdme.equations`.

| question | tool | device |
|---|---|---|
| fixed points of one system | `RDMNetwork.fixed_points()`: Newton from a lattice over the physical box [0, dt/t_abs]^M, plus the exact 1-D starts (`reduced_guesses`) for M = 2 with w(I→I) = 0; bracketing for M = 1 | CPU |
| fixed points of a batch | `RDMNetwork.fp_sweep(chunk=...)`: Newton in chunks that halve on out-of-memory (`run_chunked`) | GPU |
| leading modes: μ, frequency | `RDMNetwork.leading_modes(m)` / `stability.leading_modes`, ARPACK | CPU |
| growth rate where ARPACK fails | `RDMNetwork.leading_growth(m)` / `stability.leading_growth`, power iteration, one system or a batch | GPU for batches |
| bifurcations along a cut | `scan_cut` (batched fixed points + growth rates) and `find_bifurcations` (Hopf, period-doubling and branch points bisected with ARPACK; saddle-nodes on the count) | GPU, then CPU |
| branch diagram, M = 1 | `branch_scan` | CPU |
| which attractor a run reaches | `classify_attractor`, `compare_starts` on batched runs | GPU (the runs) |
| full spectrum | `jacobian_eigvals`: the dense oracle, for tests only | CPU |

**Devices** (measured on an RTX 4070 SUPER, E–I state D ≈ 8900):

| operation | CPU | GPU |
|---|---|---|
| single-system JVP (ARPACK steps) | 0.22 ms | 0.44 ms |
| batched JVP, 81 systems | 2.2 ms | 0.43 ms |
| batched Newton, 81 × 36 starts | 78 s | ~1 s (27 × 36: 0.4 s, 0.66 GiB) |

**Timings.** The whole bifurcation scan of the Fig. 5 cut (81 values, the bisection to 0.01) now
takes 49 s, against about 2 h with dense spectra.

- **`leading_modes`** gives the leading eigenvalues of the linearized one-step map,
  matrix-free.
  - Returns μ = ln|λ|/dt (1/ms), ω = arg λ/dt, and a warm start `v`.
  - Checked against the dense spectrum (`tests/test_stability.py`).
  - **k stays small:** below the few isolated leading modes sits a band of about Q
    near-degenerate eigenvalues, and k = 6 already reaches into it.
  - **tol is explicit:** ARPACK's default tolerance (machine precision) caused the stalls.
- **Where ARPACK fails.** Deep on the stable side the leading modes are nearly degenerate in
  modulus (two pairs within about 1e−4 /ms, more just below). ARPACK fails there at any k, ncv
  or tol, and `ArpackNoConvergence` means *unresolved*, not *stable*.
- **`leading_growth`** takes over there.
  - Power iteration; μ is the least-squares slope of the cumulative log norm. The rate
    converges where no eigenvector does. No frequency.
  - Within 2e−5 /ms of ARPACK where ARPACK converges, and within 5e−4 of the dense spectrum at
    a degenerate point.
- **Tried and dropped:**
  - ARPACK on the propagator Jᵖ: slower at every p.
  - Dense fallbacks: replaced by `leading_growth`.
- **The real fix for degenerate spectra** is root-finding on the characteristic equation
  det(I − W·Ĝ(z)) = 0, mode by mode. Not done.

## Scripts

| script | what it does |
|---|---|
| `scripts/wc_mf_vs_sm.py` | one preset, mean field against spins: conventions table per population, rate and σ against time, the attractor in (r_E, r_I) with the fixed points, age distributions |
| `scripts/ei_bifurcations.py` | the bifurcations along w(E→I), via `stability.scan_cut` and `find_bifurcations` |
| `scripts/ei_sigma_order.py` | dt-order of the oscillating presets; tables and phase profiles in `scripts/figures/EI-sigma_order/` |
| `scripts/ising_mf_vs_sm.py` | holds `I1-fig5-collapsed` for the collapse check |

## Open

- Why the spin σ at EI-async/Fig. 5 is about 100× that of the collapsed single population.
- Whether the E_inh = 15 Hopf is subcritical (needs a continuation of the cycle branch).
- The H_rev threshold for the paper (5 % here).
- The characteristic equation, for spectra where the leading modes are degenerate.
