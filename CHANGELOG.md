# Changelog

## Unreleased
### Math/Algebra
- Correct payoff-operator canonicalization to left-centering only:
  `A = M_Z^(k) @ A_raw`. This preserves the replicator field under the payoff
  equivalence `A ~ A + 1 c.T`; right-centering is no longer imposed.

### API
- Add `centering_projector`, `canonicalize_payoff_operator`, and
  `replicator_field` geometry helpers.
- Return the ridge estimate as `A_raw` from both operator estimators while
  keeping public `A` as the canonical representative satisfying `1.T @ A == 0`.

## v0.1.1 — 2025-09-01
### API
- Rename regularization arg **`lambda_` → `ridge`** in:
  - `ts2eg.core.estimate_A_from_series`
  - `ts2eg.extensions.estimate_A_from_series_weighted`
  - `ts2eg.extensions.surrogate_ess_frequency`
- Keep `lambda_common` (I-projector weight) unchanged.

### Math/Algebra
- Enforce **both-side centering** of payoff operator: `A = M_Z^(k) @ A @ M_Z^(k)`.
- **Robust ridge solve**: use `np.linalg.solve(...)`; **fallback** to `np.linalg.pinv(...)` on `LinAlgError`.

### Stability / Guards
- Guard `sum(weights) > 0` in weighted estimator.

### Examples / Docs / CI
- Demo: NumPy 2.0 compat (`np.ptp(X, ...)`), minor cleanup.
- README/docs: `VAR = Vector Autoregression`; link `docs/ts2eg_math_background.tex`.
- CI: run on PRs and on pushes to `main` only.
- Packaging: stop tracking `src/ts2eg.egg-info/`.

### Notes
- This release intentionally **drops** the `lambda_` keyword; callers must pass `ridge=...`.
