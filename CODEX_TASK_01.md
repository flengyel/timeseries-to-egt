# Codex task 1: operator normalization and guardrails

Work in `flengyel/timeseries-to-egt` on the current `main` branch. Read `AGENTS.md`
and the referenced design documents first.

## Goal

Correct the payoff-operator normalization and add mathematical regression tests.
Do not attempt the new latent-state estimator in this task.

## Required changes

1. Add a small geometry module with:
   - `centering_projector(k)`;
   - `canonicalize_payoff_operator(A)` returning `M @ A`;
   - `replicator_field(A, x)`;
   - validation and clear docstrings.
2. In both unweighted and weighted operator estimators:
   - retain the raw ridge estimate as `A_raw`;
   - set public `A` to `M @ A_raw`;
   - do not multiply on the right by `M`;
   - document `1.T @ A == 0` as the canonical condition;
   - do not assert or advertise `A @ 1 == 0`.
3. Add regression tests showing:
   - invariance under `A -> A + 1 c.T`;
   - invariance under left canonicalization;
   - failure of right-centering invariance on a seeded random example;
   - strategy-permutation equivariance;
   - sign reversal is not an equivalence.
4. Update README, mathematical background, docstrings, and changelog only where
   necessary to describe the corrected normalization.
5. Do not change ESS significance, NMF, notebooks, or surrogate selection except
   for minimal adjustments required to keep the existing suite running.

## Acceptance

Run:

```bash
python -m pip install -e .
pytest -q
MPLBACKEND=Agg TS2EG_CI=1 python scripts/run_notebooks.py --tag fast --timeout 120
```

In the final report, list touched files, summarize the mathematical correction, and
include exact test results. Do not claim that operator recovery or statistical power
has been established.
