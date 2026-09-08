# Changelog

## Unreleased

### Scientific scope

- Reclassify the current pipeline as a scientifically flawed legacy
  compatibility computation with no validated detection capability. Its fit and
  diagnostics do not detect dynamics, strategic interaction, regimes, cycles,
  ESS in data, or domain effects.
- Distinguish the contemporaneous payoff-field regression from the future
  mixture-motion estimator. NMF remains an exploratory reconstruction tool, not
  a validated selector of replicator coordinates.

### Math/Algebra

- Correct payoff-operator canonicalization to left-centering only:
  `A = M_Z^(k) @ A_raw`. This preserves the replicator field under the payoff
  equivalence `A ~ A + 1 c.T`; right-centering is no longer imposed.
- The v0.1.1 both-side-centering entry below records historical released
  behavior and is explicitly superseded by this correction. It is not the
  current canonicalization rule.

### API

- Add `centering_projector`, `canonicalize_payoff_operator`, and
  `replicator_field` geometry helpers.
- Return the ridge estimate as `A_raw` from both operator estimators while
  keeping public `A` as the canonical representative satisfying `1.T @ A == 0`.
- Re-export `project_out_identities`, `value_gradient_payoffs`,
  `growth_payoffs`, and `info_gain_payoffs` from the package root.
- Return `ss_res` and `ss_tot` with the legacy payoff-field regressions. Their
  in-sample `R2` is now `NaN` when the centered target has zero sum of squares,
  where the ratio is undefined, instead of reporting a spurious perfect fit.
- Preserve `ridge` as the supported regularization keyword; the removed
  `lambda_` keyword is not restored.

### Equilibria

- Separate symmetric Nash checks, ESS certification, and local replicator
  stability in `find_ESS` records.
- Add three-state `ess_status`, orthonormal tangent bases, symmetric-part
  eigenvalues, transverse invasion eigenvalues, and an independently reported
  hyperbolic stability status. Pure supports now use a zero-dimensional tangent.
- Retain `is_ess` as an alias for `ess_status == "certified"`; it is not a
  complete non-ESS decision when the status is `"undetermined"`.

### Deprecations

- Deprecate `surrogate_ess_frequency` with `DeprecationWarning`. Its frequency
  is an uncalibrated legacy descriptive diagnostic, not a p-value or test.

### Documentation, notebooks, and repository

- Recast the README, mathematical background, domain notes, and notebooks as
  documentation of compatibility computations. Remove motion-estimation,
  NMF-coordinate, calibrated-inference, causal, and domain-detection overclaims.
- Document the capital-labor circular-shift degeneracy and the time-constant
  value-gradient payoff construction instead of presenting an invalid null test.
- Make the fast notebook runner execute committed notebook content directly and
  consolidate notebook validation into the main CI workflow.
- Add repository hygiene checks and remove tracked virtual environments, Python
  caches, generated TeX/PDF products, transient CSV outputs, editor settings,
  working prompts, and obsolete notebook-mutation/bootstrap scripts.

## v0.1.1 — 2025-09-01

### API

- Rename regularization arg **`lambda_` -> `ridge`** in:
  - `ts2eg.core.estimate_A_from_series`
  - `ts2eg.extensions.estimate_A_from_series_weighted`
  - `ts2eg.extensions.surrogate_ess_frequency`
- Keep `lambda_common` (I-projector weight) unchanged.

### Math/Algebra

- Enforce **both-side centering** of payoff operator:
  `A = M_Z^(k) @ A @ M_Z^(k)`. This historical v0.1.1 behavior is superseded
  by the Unreleased left-centering correction above.
- **Robust ridge solve**: use `np.linalg.solve(...)`; **fallback** to
  `np.linalg.pinv(...)` on `LinAlgError`.

### Stability / Guards

- Guard `sum(weights) > 0` in weighted estimator.

### Examples / Docs / CI

- Demo: NumPy 2.0 compat (`np.ptp(X, ...)`), minor cleanup.
- README/docs: `VAR = Vector Autoregression`; link
  `docs/ts2eg_math_background.tex`.
- CI: run on PRs and on pushes to `main` only.
- Packaging: stop tracking `src/ts2eg.egg-info/`.

### Notes

- This release intentionally **drops** the `lambda_` keyword; callers must pass
  `ridge=...`.
