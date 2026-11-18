# Sizing the age grid

How the number of age bins `Q` is chosen, why, and what it replaced.

## The problem

The RDM age axis is finite: bins `0 … Q-1`, with ages up to `U_max = (Q-1)·dt`. The last bin
is **absorbing**: every cohort older than `U_max` is lumped there and fires at the field of age
`U_max`, not at its own field `h(u) = S(u) + R(u) − θ`. A cohort that old should have its
refractory kernel closer to 0 and its synaptic term closer to the filtered input. So the lumped
bin fires at the wrong rate, unless the field has stopped changing by `U_max`.

`Q` therefore has to be large enough that the field has converged by `U_max`, and no larger:
cost is linear in `Q`.

## The rule

One parameter, `tol`, sets `Q` for every model the library builds (`RDMNetwork` and all its
builders; default `tol = 1e-2`):

> `U_max` is the smallest age window for which the truncation misassigns at most a fraction
> `tol` of each population's firing, at every step.
> Then `Q = ceil(U_max / dt) + 1`.

This comes from a bound and needs no fitting. It depends only on the model's parameters and
`tol`, not on `dt`, on a pilot run, or on which attractor the run lands on.

### The bound

For any cohort of age `u ≥ U ≥ t_abs`, the error in its field is the sum of two terms.

**Refractory term.** `R(u) = −K_ref e^{−u/τ_ref}` rises monotonically to 0, so

    |R(u) − R(U)| ≤ K_ref e^{−U/τ_ref}.

**Synaptic term.** Two cohorts share their first `U/dt` filter taps, so the gap is the
filter's tail, `Σ_{k≥Q} α(1−α)^{k−1} I(t−k)`. The input `I = E + Σ_b w_ab r_b` has no
instantaneous bound: one synchronous burst can put a whole population into a single step. The
**dead time** does bound it on average, because no neuron of population `b` fires twice
within `t_abs,b`. Taking the tail window by window at its largest weight, and using
`α/dt ≤ 1/τ`:

    |S(u) − S(U)| ≤ A e^{−U/τ_int},   A = |E| + Σ_b |w_ab| / (τ_int (1 − e^{−t_abs,b/τ_int})).

No `dt` is left in this bound, so it holds at every step size.

**Field to firing.** Write `Δ(U) = K_ref e^{−U/τ_ref} + A e^{−U/τ_int}`. Every hazard
satisfies `Φ(h+δ)/Φ(h) ∈ [e^{−β|δ|}, e^{β|δ|}]`:

- exactly, for `poisson`;
- by concavity of `1 − e^{−x}`, for `poisson_exact`;
- by `σ(x+δ)/σ(x) ≤ e^{|δ|}`, for `bernoulli`.

The boundary bin's share of the firing is at most 1, so

    |misassigned firing| / total firing  ≤  expm1(β Δ(U)).

`equations.age_window` solves `expm1(βΔ(U)) = tol` for `U` by bisection, per population. A
batch shares one grid, so it takes the largest `U` of any population anywhere in it.

### In the code

| Where | What |
|---|---|
| `equations.drive_bound` | `A`, per population |
| `equations.field_tail` | `Δ(U)` |
| `equations.age_window` | `U_max` from `tol` |
| `RDMNetwork(..., tol=1e-2)` | `Q = ceil(U_max/dt) + 1`, at least the dead-time bins + 1 |
| `RDMNetwork.truncation_error()` | the same bound at the current state, with the boundary bin's actual firing share in place of 1. It is `≤ tol` whenever `Q` came from `tol`. |

`Q=` is still accepted as an explicit override. It is meant for putting two models on one grid
(comparing a batch element bit for bit with its single system), and for models with
`t_abs = 0`. Those have no rate ceiling and so no finite bound: `age_window` refuses them, and
`truncation_error()` reports `inf`.

## What there was before

Five ways to choose `Q` coexisted:

| Method | Used by | Basis | Problem |
|---|---|---|---|
| `q_from_tau(τ, dt, eps)` as the model default: `U = −τ ln eps` | every model not given `Q` | the kernel's *normalised* tail | Ignores the amplitude `K_ref` and the gain `β`. The real field error is `β·K_ref·e^{−U/τ}`, not `e^{−U/τ}`. At S1 it gives 69 ms, where the dynamics collapses (rate error 86–100 %). |
| `q_from_rate`, `q_from_tau_and_rate` | `wc-painchaud` | tail mass `e^{−rU}` × field error | Needs the firing rate, which is an output of the run, so a pilot pass is needed. The mass factor is only a heuristic bound. |
| `convergence.fit_age_grid(build, tol)` | most S1 scripts | doubles `Q` with full-length pilot runs until `truncation_error() < tol` | Expensive: each doubling is a whole `equi + meas` pass at full batch size. The result depends on the attractor and on the pilot length, and it inherits the flaw in `truncation_error` below. |
| Held constants (`QM = 1566`, `U_MAX = 200`, `U_MAX = 138.5`, "fit at one `dt`, hold `U_max`") | `S1-2Dscan_betaJ_batch`, `S1-hopf_dt`, `S1-epr_bench`, `S1-dt_cost`, … | an old fit, frozen | No longer tied to the code that produced them. `QM = 1566` was fitted before the 2026-11-13 rebuild. |
| `q_from_tau(..., 0.01)` called directly | `wc-lyapunov`, `wc-search` | as the default | As the default. |

### The flaw in the old `truncation_error`

The old check was `p[-1] · |Φ(I(t) − θ) − Φ(h[-1])| / m`. It compared the boundary field with
`I(t) − θ`, but the infinite-age synaptic term is the filtered input, which differs from
`I(t)` whenever the input moves. On a limit cycle that gap does not shrink as `Q` grows, so
the only way to pass was to drain the boundary mass, by doubling `Q`.

Its docstring argued the check was rigorous because `h(u)` is monotone in age. That argument
holds only at fixed points.

Measured at a bursting S1 cell (`J = 60`, `β = 15.25`, CV ≈ 10, `dt = 0.5`), against a
`Q = 1400` reference:

| `Q` (U_max) | true rate error | old `truncation_error` |
|---|---|---|
| 278 (139 ms) | 4.1e-2 | 1.4e+2 |
| 417 (208 ms) | 3.5e-4 | 1.4e+2 |

At fixed points the old check was accurate. Near silence, for example, it read `3.0e-2`
against a true `3.0e-2`.

## How they compare

**Grid sizes,** S1, `poisson_exact`, `dt = 0.5`:

| Setting | Before | Now |
|---|---|---|
| default, `J = 17` | `q_from_tau`: 69 ms, Q = 139, wrong attractor | 148 ms, Q = 298 at `tol = 1e-2` |
| (β, J) map, 81 × 81, `tol = 5e-2` | `fit_age_grid`: Q = 1566 (783 ms), held as a constant | Q = 312 (155.5 ms) |
| (β, J) map, `tol = 1e-2` | fit abandoned after more than 30 min | Q = 360 (179.5 ms) |
| any `dt` | refit, or `U_max` held by hand | the same `U_max`; only its rounding to bins changes |

**Does the bound hold?** `tests/test_age_window.py` takes a trajectory on a reference grid
(`tol = 1e-12`). At every step it lumps the cohorts beyond the `tol`-sized window into one
bin, which is exactly one step of the truncated model, and records the worst relative firing
error. At `tol = 1e-2`:

| S1 cell | worst error / tol |
|---|---|
| cycle `J = 17`, `poisson_exact` / `poisson` / `bernoulli` | ≤ 1e-13 |
| bursting `J = 60`, `β = 15.25` | 0.18 |
| near-silent `J = 10`, `β = 20` | 0.57 |

The bound held in every case. It is tight (within 2×) where the boundary bin holds most of the
mass, which is the regime that sets `Q`. It is loose on healthy cycles, where little mass ever
reaches the boundary. That looseness is the price of not needing a run.

## Entropy production

The σ calculation (`epr.py`) has two boundaries, and handles both explicitly:

- **Forward lumped bin.** The future hazards of the merged cohorts are clamped at age `Q-1`,
  so cohorts merged there share them, and the sliding recursions stay exact through the merge.
- **Reverse censoring bin.** The time-to-next-spike axis stops at `K-1 = Q-1`. There the
  reverse synaptic term takes its `K-2` value, which is valid once it has settled.

In both bins, the only approximation is that lumped mass is evaluated at the boundary hazard.
That is the field error `Δ(U)` again, times β. **Neither bin needs to be discarded or treated
differently.** Discarding one would break normalisation and bias σ.

This is measured, not proven. `test_the_window_bounds_the_entropy_production` runs the σ
calculation a `tol`-sized model would use (lumped marginal, `Q`-long window) against the
long-grid one, on one reference trajectory, so only the two boundaries can differ:

| `tol` | bursting (`J = 60`, `β = 15.25`) | cycle (`J = 17`) |
|---|---|---|
| 1e-1 | 7.7e-4 | 3e-8 |
| 1e-2 | 6.8e-5 | 0 |
| 1e-4 | 2.2e-7 | 0 |

The values are relative σ errors. They stay 100× or more below `tol` and shrink with it. On a
resolved cycle, σ averaged over whole cycles follows the rate error from start to finish
(3e-7 at `tol = 1e-1`).

**Grid-scale bursts are a `dt` problem, not a `Q` problem.** At the bursting cell, nearly the
whole population fires within about two time steps, at `dt = 0.5`, `0.25` and `0.125` alike,
so the burst width follows the step. σ per cycle then varies from cycle to cycle with where
the burst lands between steps: by 1.9 % within the reference run itself at `dt = 0.5`, while
the spike count per cycle varies by 0.1 %. Any small change to the run, a change of `Q`
included, re-rolls that variation. So averaged σ there can move by about 1e-3 between grids
whose rates agree to 3e-7. The `dt → 0` check is the guard for this, and such a cell does not
pass it.

## Limits

- **It bounds the error per step, not its long-time consequence.** A perturbation of size
  `tol` can be amplified near a bifurcation or on a chaotic attractor. On every cell measured,
  the error in the mean rate sat at or below `tol`, but that is an observation, not part of
  the bound.
- **Relative σ error is not controlled near a fixed point.** There σ → 0 while `H_fwd` and
  `H_rev` stay finite, so only the absolute error is small (of order `tol × H`).
- **It needs a dead time.** Without `t_abs` a source's rate is unbounded, so no window is
  bounded. Pass `Q` explicitly and read `truncation_error()` as `inf`.
- **It can overshoot where little mass reaches the boundary,** by up to about `τ·ln(1/share)`,
  where `share` is the boundary bin's share of the firing. At S1 that is a few tens of ms.

## Status (2026-11-16)

Library and tests are migrated, and all fast and slow tests pass. The tests include the bound
on firing at every step and the σ check above, which is marked slow. The scripts still call
`fit_age_grid`, `q_from_*` or `eps=`, or hold `QM` / `U_MAX`, and have not been migrated yet.
Caches keyed on `tol` will mean a different grid once they are.
