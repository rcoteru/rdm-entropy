# rdm-entropy: coding-agent instructions

Code for P1, *Irreversibility in Integrate-and-Fire Neural Populations*: refractory density method (RDM) + mean-field entropy production rate (EPR) for SRM₀ populations, and microscopic spin simulations to check it.

## Scope: stay in this repo

- Work only inside `~/code/rdm-entropy`. **Never write to the Obsidian vault** (`~/code/research-notes`); results are copied across by hand.
- Your output is code, figures and **result dumps** (below). No interpretation, framing or literature. What the numbers *mean* for the paper is decided in the vault.
- Commit only when asked. The repo may be mid-refactor: never stash, reset or revert changes you didn't make.

## Layout

| Path | What |
|---|---|
| `src/rdme/` | Library: `rdm.py`, `rdm_batch.py`, `rdm_single.py` (RDM), `spin_model.py` (microscopic), `hazards.py`, `kernels.py`, `epr.py`, `fixed_points.py`, `lyapunov.py` |
| `scripts/` | One script per study. Naming: `<system>-<study>.py` |
| `cache/` | Hashed `.pt` caches written by scripts |
| `scripts/dumps/` | Result dumps, one Markdown file per run (spec below) |
| `scripts/figures/` | Figures, auto-saved by scripts |
| `tests/` | Tests |

## Model conventions (non-negotiable)

- **`dt0` vs `dt`.** `dt0` is the model's time quantum (λ₀ = 1/`dt0`), part of the model and fixed within a study. `dt` is the solver step and defaults to `dt0`. Both are **keyword-only**. **Never pass `dt0=dt`**: refining `dt` with `dt0` dragged along walks through different models, it doesn't converge to one.
- **`t_abs`** (absolute refractory window) defaults to `dt0`. Leave it there unless the study is about it.
- **Couplings act on rates:** $I = \sum_b w^{ab} r^b + E$, with $r = m/dt$ in 1/ms. Pass `J`, never `J/dt`.
- **Reporting rule:** every number is either labelled `dt = dt0` (the discrete model) or comes with a `dt → 0` check, preferably Richardson-extrapolated, reporting the `dt` pair it came from and the observed order. Nothing in between.
- **Units:** time in ms, rates in 1/ms, σ in nats/ms (σ/dt per step → per ms).
- **Durations in ms, never in steps.** Equilibration, recording and any trimmed window are set as physical durations (`T_EQUI`, `T_MEAS`, `SKIP_MS`, …) and converted with `helpers.n_steps(ms, dt)`. A step count means a different window at every `dt`, so refining `dt` on a fixed step count silently shortens what is measured, and two scripts at different `dt` cannot be compared by the numbers at the top of them. The duration, not the step count, goes in the cache key and in the dump. Plot time axes are in ms for the same reason (`rdme.plots` takes `dt` and labels in ms; a figure without `dt` falls back to steps).
- **EPR:** report σ. The forward/reverse terms H and H̃ diverge as rate·ln(1/dt); report them only as a control (increment per halving → r ln 2).
- **Oscillatory states:** average over whole cycles (between upward mean-crossings), not over a fixed window.
- **Age grid:** size it by the tolerance (truncation-error bound), not by a fixed Q. An undersized grid can give a different attractor.
- **Step artifact:** at strong drive and `dt = dt0`, check lag-1 autocorrelation of the rate. −1 means the whole population fires in alternate bins; refine `dt`.

## Hazard

**Default hazard: TBD** (to be set after the 2026-11-04 meeting; most likely route B, the Poisson prior: `poisson` = the MaxCal model kernel at each bin, `exp_surv` = exact-survival solver for the same continuum). Until then, every script states its hazard explicitly. Never mix hazards within one comparison. The hazard is **not** part of a preset; it is its own axis.

## Parameter presets

Every run starts from a **named preset** plus explicit overrides. No unnamed parameter sets: if a run differs from a preset in anything other than the swept axes and listed overrides, it is a new preset and gets added here and in `src/rdme/presets.py`.

`src/rdme/presets.py` holds them as `PRESETS: dict[str, dict]` of model kwargs, including `dt0` and `t_abs`. This table is the spec; keep the two in sync. A script builds its run with one call:

```python
PRESET = "S1"
HAZARD = "poisson"       # stated per script: the hazard is its own axis, not part of a preset

PARAMS = pre.params(PRESET, hazard=HAZARD, J=torch.linspace(0, 20, 300), dt=0.2)
banner = pre.banner(PRESET, PARAMS)      # "S1 · hazard=poisson · dt=0.2 (dt0=1) — E=0, β=6, …"
```

- `params(name, *, hazard, **overrides)`: the preset, the hazard (**required** — there is no `DEFAULT_HAZARD` until the convention is fixed) and the overrides. An override that is neither a model parameter of that preset nor a solver key (`dt`, `Qm`, `eps`, `device`, `N`, `ic`) raises: a typo is not something to find in the figure, and a genuinely new parameter means a new preset.
- `t_abs` is stored as `None`, the model's own default (= `dt0`), so overriding `dt0` carries the window with it.
- Named points (`POINTS`) are a preset plus the coupling that makes it a system: `params("S1/fp")` is `S1` with J = 10.
- `diff(name, params)` returns the `overrides:` field for a dump's frontmatter, built from the dict that built the model rather than retyped; `banner(name, params)` is the figure line CLAUDE.md asks for.
- `PENDING` holds the two-population names that are **not encoded yet** (`EI`, `EE-P`); asking for one raises with the reason.

Wired so far: the four `ising_1Dscan_*.py` scripts. Everything else still carries its own dict.

| Preset | Pops | Parameters | Notes |
|---|---|---|---|
| `S1` | 1 | E = 0, β = 6, θ = 0.5, τ_int = 15, τ_ref = 15, K_ref = 15, `dt0` = 1, `t_abs` = `dt0` | Main one-population system (formerly "Ising set"). Self-sustained activity, no drive. Named points: `S1/fp` (J = 10). Cycle point to be re-chosen under the new hazard (its Hopf moves to J ≈ 14.4–14.6 under route B; J = 17 is the candidate) |
| `S1-sat` | 1 | E = 0, β = 8, θ = 0.5, τ_int = 10, τ_ref = 25, K_ref = 25, `dt0` = 1, J = 40 | Formerly "low-β point". The rate-ceiling test (peaks at 100% λ₀ under the sigmoid kernel). Old runs used `dt0` = 0.5 and 1 inconsistently; use `dt0` = 1 |
| `D1` | 1 | E = 0.9, β = 30, θ = 1, τ_int = 20, τ_ref = 3, K_ref = 0, `dt0` = 0.5 | Legacy driven high-β set. Only to reproduce old figures |
| `EI` | 2 | E = 1, β = 30, θ = 1, τ_int = 20, τ_ref = 3, K_ref = 0, E_ratio = 0.8, w_IE = −6, w_II = −1, `dt0` = 1 | From `wc-EEEI_sweep.py`; **to be confirmed**. `wc-mf_vs_sm.py` has two other E–I sets: reconcile or name them |
| `EE-P` | 2 | Painchaud et al. 2022 mapping, built in `wc-painchaud.py`, `dt0` = 0.5 | Painchaud cell. The τ_m mapping may change |

Known violations to fix: `wc-EEII_sweep.py` and `wc-lyapunov.py` pass `dt0=dt`; `ising_2Dscan_EJ.py` uses θ = 1.0 (not `S1`); `ising_mf-vs-sm.py` uses three unnamed combinations; `ising_1Dscan_beta.py` still says "Hopf at J ~ 14.5" (it is 12.06 at `dt = dt0` with the sigmoid kernel); `wc-EEEI_sweep.py` mentions the removed `"sync"` hazard.

## Figures

Scripts auto-save figures to `scripts/figures/<YYYY-MM-DD>_<script>_<slug>.png`. Filenames must be unique across the repo (Obsidian resolves embeds by filename). Titles or captions state preset, hazard and `dt`.

The saving is `rdme.plots.show()`, which every script calls in place of `plt.show()`: it saves every open figure and then shows them, so no run leaves its figures only on the screen. The directory is the running script's own folder plus `figures/` (override with `$RDME_FIG_DIR`), the date and script stem come from the run, and the slug is the figure's label (`fig.set_label(...)`), else its suptitle, else a lone axes' title. A re-run on the same day overwrites. PNG here; the paper's figures are drawn in `paper/` and saved as PDF there.

## Result dumps

One file per run in `scripts/dumps/<YYYY-MM-DD>_<script>_<slug>.md`, written after the run finishes.

Dumps are **not committed** -- they are local run records, like the figures beside them, and travel to the vault by hand. `scripts/dumps/` and `scripts/figures/` are both gitignored. What the `commit:` field in the frontmatter pins down is the code the run used, which is what makes the dump reproducible without the dump itself being in the history.

**A dump is never rewritten.** A re-run writes a new dump with `supersedes:`. The only edit allowed on an old dump is its frontmatter: `status: stale` (or `bug`) and `superseded_by:`.

### Frontmatter

```yaml
---
script: scripts/S1-dt_conv.py
commit: 0ab4453          # append "-dirty" if the working tree had uncommitted changes
date: 2026-11-05
preset: S1
overrides: {J: 17}       # everything that differs from the preset, except the swept axes
sweep: {dt: [1, 0.5, 0.25, 0.125]}
hazard: exp_surv
N: mf                    # "mf", or the list of N for spin simulations
seeds: [0, 1]
tolerance: 1e-10         # age-grid tolerance
cache: [cache/S1-dt_conv_a6421594.pt]
figures: [2026-11-05_S1-dt_conv_rate.png]
status: measured         # measured | partial | stale | bug
supersedes: []
superseded_by: []
---
```

### Body

1. **Question**: one line, what the run answers.
2. **Reproduce**: the exact command.
3. **Results**: tables with units. Every number carries its `dt` (or "dt → 0, Richardson from (dt₁, dt₂), order p"). Embed figures as `![[<figure filename>]]`.
4. **Checks**: convergence orders, numerical floors, the H control (−r ln 2 per halving), grid tolerance, lag-1 autocorrelation where relevant, MF vs spins agreement if both ran.
5. **Caveats**: what is not converged, what is near a floor, what differs from earlier runs (link the old dump by filename).
6. **Follow-ups**: checkboxes, each with exactly one tag: `- [ ] #run ...` (numerical work) or `- [ ] #code ...` (code chores).

**Status values** (same legend as the vault): `measured` = converged, reproducible from a clean commit under current conventions. `partial` = not converged, not systematic, or from a dirty tree. `stale` = conventions or code changed since. `bug` = known wrong. A dirty-tree run is at best `partial`.

No interpretation, no literature, no "this shows that the paper…". State what was computed and how well it is converged.
