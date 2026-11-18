# Units audit and regime map of the population model

Status as of 2026-11-17. Covers the units audit of the RDM population model, the stripped-limit analytic checks, and a first regime map of the full model. Nothing in `src/` was changed except a reporting helper. No preset or default was changed.

## Summary

| Item | Result |
|---|---|
| (a) J multiplies the rate r (1/ms), not m = r·Δt | Pass |
| (b) No rate or ν₀ in Hz | Pass |
| (c) Filter has unit gain at every Δt | Pass |
| (d) θ, K, E, J·r share one unit, β its inverse | Pass |
| 3a. J = 0 rate equals σ(b)/t_abs | Pass, first order in Δt |
| 3b. Fixed points solve x = σ(Gx + b), cusp at (4, −2) | Pass on fixed points. The high-rate branch oscillates; see below |
| 3c. Invariance under h → c·h, β → β/c and the (ν₀, θ) gauge | Pass, 1e-14 |
| 4. Full model, τ_m/t_abs = 5 and 10 | One fixed point everywhere in range. Two-state threshold moves past G = 15. Stable under Δt halving |
| 4. Full model, τ_m/t_abs = 1 | **Not resolved.** 330–441 cells per panel change regime between Δt = 0.1 and 0.05 ms |

## 1. Units audit

**(a) J multiplies the rate.** The input is built in one place, `I = w @ r + E` ([equations.py:36-62](../src/rdme/equations.py#L36-L62)). Every caller passes r = m/dt:

- `equations.total_input` ([equations.py:419-422](../src/rdme/equations.py#L419-L422)), used by the driver ([mean_field.py:354](../src/rdme/mean_field.py#L354)) and the Jacobian map ([mean_field.py:509](../src/rdme/mean_field.py#L509))
- fixed points ([fixed_points.py:160](../src/rdme/fixed_points.py#L160))
- tangent map ([lyapunov.py:65](../src/rdme/lyapunov.py#L65))
- spins ([spin_model.py:199](../src/rdme/spin_model.py#L199), [spin_model.py:209](../src/rdme/spin_model.py#L209))

Confirmed numerically: the 3b cusp stays at G ≈ 4 for Δt = 0.1, 0.05 and 0.025 ms. Older scripts in `scripts/old/` may hard-code J/dt and need checking by hand.

**(b) No Hz.** ν₀ defaults to 1/ms ([mean_field.py:143](../src/rdme/mean_field.py#L143)). It enters only as ν₀·e^{βh} ([hazards.py:221-223](../src/rdme/hazards.py#L221-L223)) and as λ·dt with dt in ms ([hazards.py:269](../src/rdme/hazards.py#L269)). The only "Hz" in `src/` is a comment ([equations.py:276](../src/rdme/equations.py#L276)).

**(c) Unit gain.** α = 1 − e^{−Δt/τ_m} ([equations.py:319-321](../src/rdme/equations.py#L319-L321)) and S ← αI + (1−α)S ([equations.py:80](../src/rdme/equations.py#L80)). The fixed point is S = I at every Δt.

- The reset S(0) = 0 is hard-coded ([equations.py:76](../src/rdme/equations.py#L76)) with no switch. Step 3 disabled it with a subclass in a scratch script.
- Input enters S one step late. This is O(Δt) and does not move fixed points.
- `synaptic_kernel` ([equations.py:149-154](../src/rdme/equations.py#L149-L154)) is not unit-gain at finite Δt: it sums to ≈ 1 + Δt/2τ. It has no callers; left as is.

**(d) One unit for h.** h = S + R − θ ([equations.py:427](../src/rdme/equations.py#L427)), R = −K·e^{−u/τ_ref} ([equations.py:308](../src/rdme/equations.py#L308)), and J carries [h]·ms. β only appears as β·h. The absolute-refractory floor R_abs = S_max − θ + 40/β ([equations.py:255](../src/rdme/equations.py#L255)) is in the same unit, so it rescales with h. It is a soft floor that holds the hazard near e^{−40}.

## 2. Dimensionless parameters

`presets.dimensionless(p, r_star=None)` and `presets.dimensionless_line(...)` ([presets.py](../src/rdme/presets.py)) report:

- G = βJ/t_abs
- L = ln(ν₀t_abs) − βθ
- βE, and b = βE + L
- τ_m/t_abs, τ_ref/τ_m, βK
- given a run's r*: x* = r*·t_abs and βJ·r* = G·x*

Presets, for reference. Most sit far outside the ranges studied here:

| Preset | G | L | b | τ_m/t_abs | τ_ref/τ_m | βK |
|---|---|---|---|---|---|---|
| S1-sat | 320 | −4 | −4 | 10 | 2.5 | 200 |
| D2 | 240 | −20 | 20 | 15 | 0.067 | 0 |
| S1/fp | 60 | −3 | −3 | 15 | 1 | 90 |

**Configs used.** All runs use β = 1, ν₀ = 1/ms and t_abs = 1 ms (also 2 ms in 3a), so the dimensionless values are the swept coordinates themselves:

| Check | G | b | L | τ_m/t_abs | τ_ref/τ_m | βK | Reset |
|---|---|---|---|---|---|---|---|
| 3a | 0 | −8 … 8, step 0.5 | −3.5 | 1 | 1 | 0 | off |
| 3b | 0 … 8, step 0.25 | −6 … 2, step 0.25 | −3.5 | 1 and 5 | 1 | 0 | off |
| 3c stripped | 6 | −3 | −3.5 | 1 | 1 | 0 | off |
| 3c full | 6 | −1 | −3.5 | 5 | 1 | 5 | on |
| 4 | −25 … 15, step 0.5 | −16 … 16, step 0.5 | −11, −7, −3.5 | 1, 5, 10 | 1 | 0, 5 | on |

In 3a, L plays no role because there is no reset and J = 0. Per-cell x* and βJ·r* for step 4 (189,540 rows) are in `check4_cells.csv`, which the audit scripts regenerate.

## 3. Stripped limit (K = 0, no reset)

**3a. J = 0.** Started at the library's stationary state and run 20 ms. The rate matches the discrete renewal formula 1/(t_abs + Δt/(e^{λΔt} − 1)) to 5e-16 and stays put to 1e-12. Error against the continuum σ(b)/t_abs, worst over b, for t_abs = 1 ms:

| Δt (ms) | max rel. error | bound Δt/(2t_abs) | ratio per halving |
|---|---|---|---|
| 0.125 | 4.9e-2 | 6.3e-2 | |
| 0.0625 | 2.6e-2 | 3.1e-2 | 1.88 |
| 0.03125 | 1.4e-2 | 1.6e-2 | 1.90 |
| 0.015625 | 7.1e-3 | 7.8e-3 | 1.93 |

t_abs = 2 ms with the same Δt/t_abs gives identical numbers, as it should.

**3b. J ≠ 0.** Runs start from x = 0.001 and from x = 0.99, run 1500 ms, and are averaged over the last 200 ms.

- Wherever a run settles, it sits on the discrete root of x = f_Δt(Gx + b) to within 1e-6.
- Two fixed points first appear at G = 4.5, b = −2.25. That is the cusp (4, −2) to grid resolution: at G = 4.25 the window is only 0.04 wide in b, under the 0.25 grid.
- Every disagreement with the continuum count lies on a fold.

![Stripped-limit regimes](figures/units_audit/check3b_regimes.png)

**Finding: the high-rate branch oscillates, with period ≈ t_abs.** This is a synchronous dead-time mode, and it is not converged in Δt.

- At τ_m/t_abs = 5, Δt = 0.1 → 0.05 → 0.025 ms:
  - peak amplitude 0.09 → 2.2 → 5.1 /ms
  - oscillating cells 201 → 290 → 333 of 1089
- At τ_m/t_abs = 1 it reaches down to G ≈ 1.25.
- The coarse step damps it, so Δt = 0.1 ms hides a state that the continuum model has.

**3c. Invariances.** Rescaling h by c and β by 1/c (c = 0.1, 10), and the gauge move (ν₀, θ) → (ν₀e^{βδ}, θ + δ) with δ = −1, 2, change r(t) by at most 1.1e-14 (relative) over 300 ms. This holds for both the stripped and the full model.

## 4. Full model (reset on, then K)

**Method.** Both starting points (x = 0.001 and 0.99) are placed at the library's stationary state for that rate. Runs last 1500 ms with stats over the last 300 ms, at Δt = 0.1 and 0.05 ms. τ_ref/τ_m = 1, because the brief gives no value. Each cell gets one of these labels:

| Label | Rule (x units, t_abs = 1) |
|---|---|
| oscillation | half peak-to-peak > 5e-3 |
| two states | the two starts' means differ by > 1e-3 |
| not converged | drift between window halves > 1e-5, no oscillation |
| saturated | both starts' means > 0.995 |

The saturated cells fire once per t_abs. Their ripple depends smoothly on the starting point and never drifts (tested at two cells, Δt = 0.05 and 0.025). They are one state with a frozen phase pattern, not two attractors.

![Full model, dt = 0.05 ms](figures/units_audit/check4_regimes_dt0.05.png)

**τ_m/t_abs = 5 and 10.** These panels hold one fixed point everywhere in G ∈ [−25, 15], b ∈ [−16, 16], for all L and βK. Not a single cell changes between Δt = 0.1 and 0.05. The two-state threshold therefore moves **beyond G = 15**.

A plausible reason: with the reset, a neuron fires at age ≈ t_abs, when S has only reached I·(1 − e^{−t_abs/τ_m}) ≈ 0.1–0.2·I. A partial count of stationary states put the first window near G ≈ 48.5 for τ_m/t_abs = 5, L = −11, βK = 0 (Δt = 0.1). That count was stopped, did not check stability, and lies outside the requested range.

**τ_m/t_abs = 1.** First G with two stable fixed points, from the dynamics:

| βK | L | Δt = 0.1 | Δt = 0.05 | cells changed |
|---|---|---|---|---|
| 0 | −3.5 | 8.0 (b = −3.0) | 8.0 (b = −3.0) | 441 |
| 0 | −7 | 9.5 (b = −3.0) | 9.5 (b = −3.0) | 349 |
| 0 | −11 | 12.5 (b = −3.5) | 15.0 (b = −5.0) | 337 |
| 5 | −3.5 | 11.5 (b = −3.5) | 10.0 (b = −3.0) | 330 |
| 5 | −7 | 13.0 (b = −3.5) | none ≤ 15 | 358 |
| 5 | −11 | none ≤ 15 | none ≤ 15 | 411 |

These thresholds are not reliable. Most cells that change regime switch between oscillation and fixed point. The oscillation period changes with Δt:

- At Δt = 0.1, the dominant periods are near 0.5 ms or on the grid scale (2 bins).
- At Δt = 0.05, they are ≈ t_abs, with amplitudes up to 3.7 /ms, about twice as large.

At Δt = 0.1, the oscillations at negative G (inhibition) have grid-scale periods and vanish at Δt = 0.05, so they are numerical. Reporting the τ_m/t_abs = 1 boundaries would need Δt ≤ 0.025 ms with its own halving check, which was not run.

![Regime changes between dt = 0.1 and 0.05](figures/units_audit/check4_dt_diff.png)

The Δt = 0.1 map is in [figures/units_audit/check4_regimes_dt0.1.png](figures/units_audit/check4_regimes_dt0.1.png).

## Open questions

- Does the dead-time oscillation of the high branch survive Δt → 0 with a finite amplitude? It grows at every halving so far, in both the stripped and the full model.
- τ_ref/τ_m = 1 was assumed in step 4. A different ratio changes the βK = 5 panels.
- Where exactly the two-state threshold lands for τ_m/t_abs = 5 and 10 is open beyond "> 15".
- The audit scripts live in a scratch directory and are not in the repo. Move them to `scripts/` if these results need to be reproduced.
