# rdm-entropy: coding-agent instructions

Code for P1, *Irreversibility in Integrate-and-Fire Neural Populations*: refractory density method (RDM) + mean-field entropy production rate (EPR) for SRM₀ populations, and microscopic spin simulations to check it.

## Principles

1. **State assumptions, don't silently pick one.** If a request has more than one reasonable reading, or conflicts with a convention below, say so and ask rather than guessing.
2. **Minimum code for the task.** No speculative features, abstractions for single-use code, configurability nobody asked for, or error handling for scenarios that can't happen.
3. **Touch only what the task needs.** Don't refactor or "improve" adjacent code, comments or formatting; match existing style; remove only the imports/variables your own change orphaned, and mention (don't delete) unrelated dead code you notice.
4. **Define done before starting.** Multi-step or bug-fix tasks get a verification step stated up front (the test that reproduces the bug, the check that proves the fix) and run before reporting the task complete.

## Scope: stay in this repo

- Work only inside this repository. **Never write to the notes vault** that tracks this project (a separate, private repo); results are copied across by hand. `agents/briefs/` (below) is the one channel from the vault into this repo — read it, never write back through it.
- Your output is code, figures and **result dumps** (below). No interpretation, framing or literature. What the numbers *mean* for the paper is decided in the vault.
- Commit only when asked. The repo may be mid-refactor: never stash, reset or revert changes you didn't make.

## Layout

| Path | What |
|---|---|
| `src/rdme/` | Library: `rdm.py`, `rdm_batch.py`, `rdm_single.py` (RDM), `spin_model.py` (microscopic), `hazards.py`, `kernels.py`, `epr.py`, `fixed_points.py`, `lyapunov.py` |
| `scripts/` | One script per study. Naming: `<system>-<study>.py` |
| `cache/` | Hashed `.pt` caches written by scripts |
| `scripts/figures/` | Figures, auto-saved by scripts |
| `agents/dumps/` | Result dumps, one Markdown file per run (spec below) |
| `agents/briefs/` | Briefs dropped in from the vault: decisions the vault made that this repo can't see (prior/parameter/convention choices). Read the newest one before starting if present, and write a dump when you are done with it, measurement or not (see Result dumps) |
| `tests/` | Tests. Split fast/slow: `pytest` runs the fast suite (~40 s), `pytest -m slow` the convergence studies and large-N spin runs (~50 s). No test reads a cache; every oracle is a closed form, a second representation, or a literal in the test file |

## Executing a multi-step brief

A brief with several independent runs or checks is executed by delegation, not inline: dispatch one subagent per step, sequentially (never in parallel — one machine, and numerical runs contend for the same cores/GPU). Keep delegating until the brief's checklist is done.

- Each subagent gets one step: the relevant brief section, the preset/hazard/overrides it needs, and what "done" means for that step. It runs the script, debugs itself if it fails, and writes its own result dump (or, for a non-measurement step, says so) before reporting back.
- It returns a short status to you, not its working — pass/fail, headline numbers, the dump filename, anything that needs a decision. You never see its internal debugging; that is the point, since it is what would otherwise fill your context with logs and stack traces across every step.
- You hold the brief and the running state across steps (what's done, what each step left behind, what the next one needs); a subagent does not. Pass forward whatever the next step depends on explicitly — a subagent starts with nothing but what you give it.
- You write the closing report once every step is dispatched, pointing at the dumps rather than restating their contents.

## Model conventions (non-negotiable)

- **`nu0`, `t_abs`, `dt`.** There is no time quantum and no `dt0` (route B, 2026-11-07). `nu0` is the **rate unit** of the Poisson reference measure, fixed at 1 ms⁻¹: it is not identifiable on its own, since (ν₀, θ) → (ν₀e^{βδ}, θ+δ) leaves Φ unchanged, so only ν₀e^{−βθ} is a parameter. `t_abs` is the **physical dead time** in ms, default 1 ms, and under an exponential hazard it is the *only* thing that caps the realised rate (at 1/`t_abs`); `t_abs = 0` has to be asked for explicitly and means no ceiling. `dt` is the **solver step** and nothing else — it is **required**, with no default, and is keyword-only along with the rest.
- **`dt` must divide `t_abs`.** `kernels.n_abs_bins` refuses a `dt` above `t_abs` or one that does not divide it: a window that rounds moves with the step, which is the one thing an absolute refractory period must not do. With `t_abs` = 1 ms the admissible steps are 1, 1/2, 1/4, … — which is what the (`dt`, `dt`/2) Richardson pairs walk down.
- **Couplings act on rates:** $I = \sum_b w^{ab} r^b + E$, with $r = m/dt$ in 1/ms. Pass `J`, never `J/dt`.
- **Reporting rule:** every number comes with a `dt → 0` check. The default is **Richardson extrapolation from a coarse pair (`dt`, `dt`/2)** with `poisson_exact`, reporting the pair it came from and the observed order. There is no second case: the discrete model it used to be contrasted with no longer exists. Run the extrapolation on `poisson_exact` only — `poisson`'s per-bin error changes sign at threshold, so Richardson on it fails on an attractor that crosses one; quote `poisson` at a stated `dt`, or extrapolate the solver and compare.
- **Step validity:** `poisson` needs c = ν₀·`dt` < 1 (strictly); `bernoulli` needs ν₀·`dt` ≤ 1; `poisson_exact` has no arithmetic cap, and its step is set by accuracy alone — the local relative error is λ`dt`/2 at the peak rate (`hazards.step_error`, `hazards.check_accuracy`), read off the attractor rather than from ν₀.
- **Units:** time in ms, rates in 1/ms, σ in nats/ms (σ/dt per step → per ms).
- **Durations in ms, never in steps.** Equilibration, recording and any trimmed window are set as physical durations (`T_EQUI`, `T_MEAS`, `SKIP_MS`, …) and converted with `helpers.n_steps(ms, dt)`. A step count means a different window at every `dt`, so refining `dt` on a fixed step count silently shortens what is measured, and two scripts at different `dt` cannot be compared by the numbers at the top of them. The duration, not the step count, goes in the cache key and in the dump. Plot time axes are in ms for the same reason (`rdme.plots` takes `dt` and labels in ms; a figure without `dt` falls back to steps).
- **EPR:** report σ. The forward/reverse terms H and H̃ diverge as rate·ln(1/`dt`) against a counting measure, so report them **relative to the Poisson reference** (`epr.poisson_reference_entropy`), which is finite as `dt` → 0 and is the model's own reference measure under route B. Record both: raw (as the control — increment per halving → r ln 2) and referenced. σ is unchanged by the referencing, both halves subtracting the same offset.
- **Oscillatory states:** average over whole cycles (between upward mean-crossings), not over a fixed window.
- **Age grid:** size it by the tolerance (truncation-error bound), not by a fixed Q. An undersized grid can give a different attractor.
- **Step artifact:** at strong drive and a coarse `dt`, check lag-1 autocorrelation of the rate. −1 means the whole population fires in alternate bins; refine `dt`. `max_step_prob()` is the companion diagnostic — the realised max Φ along a trajectory, which is what says whether the step resolves an unbounded hazard.

## Hazard

**Route B, the Poisson reference, is the model** (decided 2026-11-07). Three kernel keys, hard-renamed with no aliases:

| Key | Φ per bin | rate | What it is |
|---|---|---|---|
| `poisson` | σ(βh + ℓ), ℓ = logit(c), c = ν₀·`dt` | ν₀e^{βh} | MaxCal against the reference at **every** bin width: the model of binned data |
| `poisson_exact` | 1 − exp(−ν₀e^{βh}`dt`) | ν₀e^{βh} | Exact survival for the **same** continuum: the solver, and the **default hazard** |
| `bernoulli` | ν₀·`dt`·σ(βh) | ν₀σ(βh) | Route A, the legacy construction. Comparison and equivalence claims only — never the hazard of a measurement |

`poisson` and `poisson_exact` share a rate function *as the same object*, which is how one reads off that they are one model under two discretization maps. `poisson_exact` is the default (`hazards.DEFAULT_HAZARD`) because its per-bin error is single-signed and its convergence monotone, which is what a `dt → 0` limit is taken with.

Every script still states its hazard explicitly — it is its own axis, not part of a preset, and the dump's frontmatter has to carry it. Never mix hazards within one comparison.

## Parameter presets

Every run starts from a **named preset** plus explicit overrides. No unnamed parameter sets: if a run differs from a preset in anything other than the swept axes and listed overrides, it is a new preset and gets added here and in `src/rdme/presets.py`.

`src/rdme/presets.py` holds them as `PRESETS: dict[str, dict]` of model kwargs, including `nu0` and `t_abs`. This table is the spec; keep the two in sync. A script builds its run with one call:

```python
PRESET = "S1"
HAZARD = "poisson"       # stated per script: the hazard is its own axis, not part of a preset

PARAMS = pre.params(PRESET, hazard=HAZARD, J=torch.linspace(0, 20, 300), dt=0.25)
banner = pre.banner(PRESET, PARAMS)      # "S1 · hazard=poisson · dt=0.25 · t_abs=1 ms — E=0, β=6, …"
```

- `params(name, *, hazard, **overrides)`: the preset, the hazard (falls back to `hazards.DEFAULT_HAZARD`) and the overrides. **`dt` is required**: there is no quantum to default to, so a run that has not named its step has not said how it was solved. An override that is neither a model parameter of that preset nor a solver key (`dt`, `Qm`, `eps`, `device`, `N`, `ic`) raises: a typo is not something to find in the figure, and a genuinely new parameter means a new preset.
- `nu0` and `t_abs` are stored as numbers (1 ms⁻¹ and 1 ms), in physical units, and do not move when `dt` does.
- Named points (`POINTS`) are a preset plus the coupling that makes it a system: `params("S1/fp")` is `S1` with J = 10.
- `diff(name, params)` returns the `overrides:` field for a dump's frontmatter, built from the dict that built the model rather than retyped; `banner(name, params)` is the figure line CLAUDE.md asks for.
- `PENDING` holds the two-population names that are **not encoded yet** (`EI`, `EE-P`); asking for one raises with the reason.

Wired so far: every `S1*` script in `scripts/`. The `wc-*.py` scripts are not, and cannot be until the two-population presets land — they carry a header saying so and are not runnable.

| Preset | Pops | Parameters | Notes |
|---|---|---|---|
| `S1` | 1 | E = 0, β = 6, θ = 0.5, τ_int = 15, τ_ref = 15, K_ref = 15, `nu0` = 1, `t_abs` = 1 | Main one-population system (formerly "Ising set"). Self-sustained activity, no drive. Named points: `S1/fp` (J = 10). Cycle point to be re-chosen from a converged scan. The Hopf is at **J_c = 11.534 ± 0.012** under `poisson_exact`, as a `dt → 0` limit (11.4646 under `poisson`, 12.3525 under `bernoulli`); J = 17 is the candidate cycle point, 5.47 past onset. The J ≈ 14.4–14.6 quoted here before was **route-A `poisson`**, never a route-B prediction, and 12.06 was the legacy sigmoid kernel |
| `S1-sat` | 1 | E = 0, β = 8, θ = 0.5, τ_int = 10, τ_ref = 25, K_ref = 25, `nu0` = 1, `t_abs` = 1, J = 40 | Formerly "low-β point". The rate-ceiling test. Under route B the ceiling is the dead time, not the kernel, so the realised rate should sit at 1/`t_abs` |
| `D1` | 1 | E = 0.9, β = 30, θ = 1, τ_int = 20, τ_ref = 3, K_ref = 0, `nu0` = 1, `t_abs` = 1 | Legacy driven high-β set. **Legacy-only, no live caller**: the `paper/fig_*.py` that used it are frozen and reproduce from `378bf2a^`. Kept as the record of those figures' parameters. Their runs were route A at `dt0` = 0.5, i.e. `bernoulli` at `dt` = 0.5 with `nu0` = 2 — state both |
| `D2` | 1 | E = 1, β = 40, θ = 0.5, τ_int = 15, τ_ref = 1, K_ref = 0, `nu0` = 1, `t_abs` = 1, J = 6 | Driven, no refractory kernel, strong gain; `dt` = 0.1, since β = 40 is stiff. Adopted 2026-11-07 from `S1-mf_vs_sm.py`'s inline `noref_driven_strong`, which matched no other preset. **The name is provisional** — it was this port's choice, not a vault decision |
| `EI` | 2 | E = 1, β = 30, θ = 1, τ_int = 20, τ_ref = 3, K_ref = 0, E_ratio = 0.8, w_IE = −6, w_II = −1 | From `wc-EEEI_sweep.py`; **to be confirmed**. `wc-mf_vs_sm.py` has two other E–I sets: reconcile or name them |
| `EE-P` | 2 | Painchaud et al. 2022 mapping, built in `wc-painchaud.py` | Painchaud cell. The mapping needs a new time-unit anchor under route B — it was anchored on the rate ceiling, which route B does not have |

**Ported 2026-11-07.** The eleven `ising_*.py` scripts are now `S1*` under `<preset>-<study>.py`, wired to `presets.params`, on route-B kernel keys, with the preset name in every cache key. The violations recorded before the rename are closed with them: `wc-EEII_sweep.py`/`wc-lyapunov.py` no longer pass `dt0=dt`; `S1-2Dscan_EJ.py` takes θ = 0.5 from the preset (which **moves the plane** — do not read the new map against the old); `S1-mf_vs_sm.py`'s four inline sets resolve to `S1` + overrides, `S1-sat` + overrides, `S1-sat` exactly and the new `D2`; `wc-EEEI_sweep.py`'s `"sync"` is gone.

Still open: the `wc-*.py` scripts keep their names and are **not runnable** — they are ported mechanically (no `dt0`, live hazard keys, `is_flow` in place of `dt < dt0`) but not wired to a preset, because `EI` and `EE-P` are `PENDING`. They get their `<preset>-<study>.py` name when those land. The Painchaud time-unit anchor (`T_u = α₁·dt0`, i.e. the route-A rate ceiling) is left exactly as it was, as `DT0_ANCHOR`, and flagged unresolved in both `wc-painchaud*.py`.

## Figures

Scripts auto-save figures to `scripts/figures/<YYYY-MM-DD>_<script>_<slug>.png`. Filenames must be unique across the repo (Obsidian resolves embeds by filename). Titles or captions state preset, hazard and `dt`.

The saving is `rdme.plots.show()`, which every script calls in place of `plt.show()`: it saves every open figure and then shows them, so no run leaves its figures only on the screen. The directory is the running script's own folder plus `figures/` (override with `$RDME_FIG_DIR`), the date and script stem come from the run, and the slug is the figure's label (`fig.set_label(...)`), else its suptitle, else a lone axes' title. A re-run on the same day overwrites. PNG here; the paper's figures are drawn in `paper/` and saved as PDF there.

## Result dumps

One file in `agents/dumps/P1_<YYYY-MM-DD>_<script>_<slug>.md`. The `P1_` prefix is what the vault files it under; keep the filename identical on both sides, since `superseded_by:` and every cross-dump citation is by filename.

**When to write one.** Two triggers, and the second is the one that gets forgotten:

1. **Every measurement run**, after it finishes.
2. **Every session that worked from a brief**, when that work is done -- *including a session that ran no measurement at all*. An API port, a test audit, a round of fixes, a commit: if a brief sent you here, the dump is how the result travels back, and it is written **before** your closing report, not instead of it. The report is a pointer to the dump plus whatever needs a decision; it is not a substitute, and the dump is not something to produce afterwards when asked. If the brief's checklist is only partly done, the dump is still written now, with the rest under Follow-ups -- one dump per brief-session, not one per finished brief.

A non-measurement dump uses the same frontmatter and body, with the run-shaped fields filled honestly rather than dropped: `script:` is the command that stands in for the run (`pytest tests`, or the commit alone), `preset: n/a`, `overrides: {}`, `sweep: {}`, `figures: []`, and `status:` judged by the same rule (on whether the result can be regenerated from the repo, not on whether the tree was clean). Results is what changed and what now holds; Checks is how you know it (the suite, a regression check, a hand check). `P1_2026-11-07_api-B_commit-blockers-landed.md` is the worked example.

Dumps are **not committed** -- they are local run records, like the figures they embed, and travel to the vault by hand (`agents/` -- dumps and briefs -- is gitignored, and so is `scripts/figures/`). What the `commit:` field in the frontmatter pins down is the code the run used, which is what makes the dump reproducible without the dump itself being in the history.

**A dump is never rewritten.** A re-run writes a new dump, and the only edit allowed on the old one is its frontmatter: `status: stale` (or `bug`) and `superseded_by: [<the new dump's filename>]`. Set it in the same session that writes the replacement -- the vault reads that field to mark every claim resting on the old dump **[stale]**, so a replacement that leaves it empty silently keeps a dead number alive.

### Frontmatter

```yaml
---
script: scripts/S1-dt_conv.py   # or the stand-in command, for a dump that ran no measurement
commit: 0ab4453          # append "-dirty" if the tree was dirty; correct it to the commit that lands the code
date: 2026-11-05
brief: agents/briefs/P1 2026-11-05 route-B API and re-runs.md   # omit if not run from a brief
preset: S1               # "n/a" when no model was run
overrides: {J: 17}       # everything that differs from the preset, except the swept axes
sweep: {dt: [1, 0.5, 0.25, 0.125]}
hazard: poisson_exact
nu0: 1                   # rate unit, 1/ms
t_abs: 1                 # dead time, ms -- dt must divide it
N: mf                    # "mf", or the list of N for spin simulations
seeds: [0, 1]
tolerance: 1e-10         # age-grid tolerance
cache: [cache/S1-dt_conv_a6421594.pt]
figures: [2026-11-05_S1-dt_conv_rate.png]
status: measured         # measured | partial | stale | bug
superseded_by: []        # set on the OLD dump when a re-run replaces it, never on this one
---
```

### Body

1. **Question**: one line, what the run answers. If run from a brief, name it (`brief:` in the frontmatter) rather than restating its decisions here.
2. **Reproduce**: the exact command.
3. **Results**: tables with units. Every number is a `dt → 0` value and says where it came from: "Richardson from (dt₁, dt₂), order p". Embed figures as `![[<figure filename>]]`.
4. **Checks**: convergence orders, numerical floors, the H control (+r ln 2 per halving on the raw pair) alongside the Poisson-referenced H, grid tolerance, the λ`dt`/2 step bound at the peak rate, lag-1 autocorrelation where relevant, MF vs spins agreement if both ran.
5. **Caveats**: what is not converged, what is near a floor, what differs from earlier runs (link the old dump by filename).
6. **Follow-ups**: checkboxes, each with exactly one tag: `- [ ] #run ...` (numerical work) or `- [ ] #code ...` (code chores).

**Status values** (same legend as the vault): `measured` = converged, reproducible from the repo under current conventions. `partial` = not converged, not systematic, or exploration-grade (a single `dt`, a loosened tolerance, a screen nothing passed). `stale` = conventions or code changed since. `bug` = known wrong.

**A dirty tree is not a ground for `partial`** (vault decision, 2026-11-07, retiring the earlier rule here). The test is whether the run can be *regenerated* from the repo, not whether it had an audit trail at the moment it ran. Record `-dirty` on the `commit:` field as the honest statement of what was in the tree, and **correct that field to the commit that lands the code once it is committed** -- an uncorrected `-dirty` commit, or one the scripts have moved past, is what the vault reads as `stale`. Genuine grounds for `partial` still apply and are stated on their own terms.

No interpretation, no literature, no "this shows that the paper…". State what was computed and how well it is converged.
