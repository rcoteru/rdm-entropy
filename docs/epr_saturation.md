# Entropy production in the saturated regime

Status as of 2026-11-16. Covers the mean-field EPR clamp bug, its fix, the evidence for it, the open question it exposed, and the dtype policy decided the same day.

## The bug (fixed 2026-11-16, uncommitted)

The mean-field EPR (`src/rdme/epr.py`) computed ln(1−Φ) and ln Φ from a hazard Φ clamped at `1 − max(1e-12, eps_dtype)`. That capped the per-bin cost of a "silent under a saturated reverse hazard" event at −ln(1e-12) ≈ 27.6 nats in float64, and ≈ 16 nats in float32.

The fix reads both logs from each kernel's exact closed-form branches, `Hazard.logphi` and `Hazard.log1m_phi` in `hazards.py`. For `poisson_exact`, ln(1−Φ) = −λ·dt. The spin-model EPR estimator always used the exact branches.

`bernoulli_log1m_phi` now uses log1p(−Φ) for ν₀dt < 1, where it is the more accurate form. The old algebraic form is kept only at ν₀dt = 1.

## Effect

S1 preset, `poisson_exact`, dt = 0.5 ms, σ per ms:

| J | after fix | before fix |
|---|---|---|
| ≤ 30 | unchanged (J=30: 0.524) | same |
| 45 | 37.3 | 0.81 |
| 60 | 3715 | 0.65 |

The old values were set by the clamp, not by the model. Capping the exact log at 27.6 reproduces them digit for digit. The reverse hazard reaches λ·dt ≈ 1.3×10⁸ per bin at J = 60.

## Verification

- **Spins vs mean field, N = 16000**, after the fix:

  | J | spins | mean field |
  |---|---|---|
  | 30 | 0.526 ± 0.028 | 0.524 |
  | 45 | 37.1 ± 1.9 | 37.3 |
  | 60 | 3692 ± 164 | 3715 |

  Before the fix the mean field was 45× to 5700× too low at J ≥ 45. The existing spin test only covered J = 10 and 17, so this went unnoticed.
- σ is now independent of the clamp. Float32 matches float64 to about 1e-5 relative.
- The sliding observer equals the joint-distribution oracle to < 1e-14. In-place and out-of-place runs are bit-identical.

## Open physics question (unresolved)

In the saturated regime σ/ms does not converge as dt → 0. Each halving of dt adds a roughly constant amount, so σ grows like ln(1/dt):

| J | dt = 0.5 | 0.25 | 0.125 |
|---|---|---|---|
| 45 | 37.3 | 57.0 | 74.4 |
| 60 | 3716 | 6808 | 9907 |

The mechanism: when λ_rev·dt ≫ 1, Φ_rev is pinned at 1, so the spike term's log ratio becomes ln(λ_f·dt) instead of the continuum ln(λ_f/λ_r). Resolving it would need dt below 1/λ_max (about 1e-8 ms), which is infeasible. The exponential rate is unbounded, so the reverse process makes the observed silences astronomically unlikely.

Not yet decided: how the paper treats this regime. The options are effectively infinite σ, a different observable, or a restricted parameter range.

Rule for now: check the maximum λ·dt before quoting σ, and treat saturated σ as dt-dependent.

## Consequences for results

- Any cached EPR at saturating parameters from before 2026-11-16 is wrong.
- The caches don't track code changes, so the saturated runs must be cleared and rerun. Likely affected scripts: `S1-sat-*`, `S1-2Dscan_betaJ*` (J up to 60), `S1-sat-2Dscan_KrefTauref`.
- Any claim built on σ at J ≥ 45 (S1) needs revisiting.

## Related decision: dtype policy (same day)

The library now owns its precision:

- **Mean field:** float64 through a `dtype` keyword on `RDMNetwork` and its factories, independent of torch's global default.
- **Spin model:** float32 state with float64 random draws and accumulators. Its results are bit-identical to before.
- **Scripts:** no longer call `torch.set_default_dtype`.
- **Tests:** `tests/test_dtype.py` checks that both models ignore the global default.

The mean field stays float64 because σ = H_rev − H_fwd with H around 3e-2 leaves a float32 floor of a few 1e-9 per bin. That floor is above σ near fixed points and near onset.

Most scripts' parameter grids are still float32-rounded (for example β = 1.2375000477) so the existing cache hashes keep matching. Changing that is pending a decision.
