# ts2eg agent instructions

## Purpose

`ts2eg` is research software for mapping time-indexed observations to explicitly
specified evolutionary-game objects. Correct identification and calibrated
validation take precedence over preserving an attractive pipeline narrative.

## Read before changing code

1. `docs/ARCHITECTURE.md`
2. `docs/design/STATISTICAL_CONTRACT.md`
3. `docs/design/VALIDATION_MATRIX.md`
4. The active plan in `docs/exec-plans/TS2EG_V2_REDESIGN.md`

Treat those files as the source of truth. Keep this file short; put detailed
mathematics and decisions in the referenced documents.

## Non-negotiable mathematical rules

- For a `k x k` payoff operator, use
  `M = I - 11.T/k` and the canonical representative `B = M @ A`.
- Never impose right-centering `A @ M` or double-centering `M @ A @ M` as an
  identification normalization. Right-centering changes the replicator field in
  general.
- Payoff equivalence is `A ~ A + 1 c.T`. Strategy relabeling is by simultaneous
  row/column permutation. A sign reversal is not an equivalence.
- Compare recovered operators through canonical operators and/or their
  replicator vector fields, not raw entrywise equality.
- Keep aggregate and relative payoff channels separate. Do not silently discard
  the aggregate channel when an application concerns creation or destruction.
- ESS search is descriptive. Do not use ESS presence as a significance statistic
  or attach a p-value to it.

## Statistical contract

- Distinguish at least three modes:
  1. observed mixtures: `x(t)` is supplied;
  2. known basis: a fixed observation model is supplied and mixtures are inferred;
  3. latent basis: loadings, mixtures, and dynamics are estimated jointly.
- Only the observed-mixture mode may be treated as inferentially stable until the
  later modes pass the validation ladder.
- Fit preprocessing transforms and latent bases on training data only. Apply the
  frozen transform and basis to validation/test data.
- If model selection is used, repeat the complete selection rule inside every
  surrogate or bootstrap replicate.
- Keep training, selection, and final test sets distinct for inferential runs.
- Predictive payoff induction is association. Do not call it rent, extraction,
  causal influence, welfare, or appropriation without an external causal design.
- Statistical smoke tests in CI do not establish calibration. Calibration and
  power claims require the slow validation suite and reported Monte Carlo
  uncertainty.

## Observation and latent-state discipline

- Do not learn a basis on one scaling of `X` and infer mixtures on another.
- Bundle every fitted preprocessing transform with the basis/model that uses it.
- Do not normalize NMF coefficients to the simplex unless the observation model
  makes that normalization valid. State whether an amplitude term is present.
- Use centered log-ratio motion for interior mixtures when fitting replicator
  dynamics. State and test the treatment of zeros and boundary states.

## Change discipline

- Implement one milestone from the active execution plan per pull request.
- Do not combine normalization fixes, a new latent estimator, notebook rewrites,
  and significance claims in one patch.
- Preserve the public API where practical; otherwise add a deprecation path and
  document the behavioral change.
- Add a regression test for every mathematical bug fixed.
- Do not edit generated notebook outputs by hand.

## Required checks

Run from the repository root:

```bash
python -m pip install -e .
pytest -q
MPLBACKEND=Agg TS2EG_CI=1 python scripts/run_notebooks.py --tag fast --timeout 120
```

For changes affecting inference or simulation, also run the relevant commands in
`docs/design/VALIDATION_MATRIX.md`. Report exactly what was run and any failures.
