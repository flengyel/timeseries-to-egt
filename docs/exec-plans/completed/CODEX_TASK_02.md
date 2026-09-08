# Codex task 2: retire ESS inference and repair descriptive equilibrium diagnostics

Baseline commit: `77568de25492eeb87aae8d39f06758ad3b4bfb43`
(`fix payoff operator canonicalization`).

Work in `flengyel/timeseries-to-egt`. Read `AGENTS.md` and every design
document it references before changing code.

## Goal

Complete Milestone 2 of `docs/exec-plans/TS2EG_V2_REDESIGN.md`:

1. remove every inferential claim based on ESS presence or surrogate ESS
   frequency; and
2. repair the equilibrium routine so that Nash equilibrium, ESS certification,
   and local dynamic stability are not conflated.

Do **not** implement the motion estimator, latent-state estimator, new surrogate
test, basis selection, or aggregate-channel redesign in this task.

## Required changes

### 1. Retire ESS-presence inference

- Remove from `README.md`, `docs/ts2eg_math_background.tex`, demos, and API
  documentation all claims that:
  - ESS presence is a test statistic;
  - surrogate ESS frequency supplies a calibrated p-value;
  - an observed ESS with low surrogate frequency establishes competitive
    interaction.
- Replace those claims with a prominent statement:
  - ESS and Nash results are descriptive properties of a fitted operator;
  - the former ESS-frequency procedure is uncalibrated and must not be used for
    inference;
  - no replacement inferential test is introduced by this milestone.
- Keep `surrogate_ess_frequency` temporarily for compatibility, but:
  - emit `DeprecationWarning`;
  - document that it is a legacy descriptive diagnostic only;
  - do not expose or describe its output as a p-value or hypothesis test.
- Update `CHANGELOG.md`.
- Mark Milestone 1 completed at commit `77568de` and Milestone 2 active in
  `docs/exec-plans/TS2EG_V2_REDESIGN.md`.

### 2. Separate Nash, ESS certification, and local stability

Preserve `find_ESS` for compatibility, but revise its return records and
documentation.

For every candidate support `J`:

1. Solve and check the symmetric Nash conditions as before.
2. Let
   \[
   A_s=(A_{JJ}+A_{JJ}^{T})/2.
   \]
3. Construct an **orthonormal** basis `Q` for
   \[
   T_J=\{z\in\mathbb R^{|J|}:\mathbf 1^Tz=0\}.
   \]
   For a pure support, `Q` has shape `(1, 0)` and the within-support tangent
   condition is vacuous.
4. Compute the eigenvalues of
   \[
   Q^T A_s Q.
   \]
5. Treat the usual strict-off-support/negative-tangent condition as an ESS
   certificate:
   - `ess_status = "certified"` when all unused pure strategies are strictly
     worse against the candidate and the symmetric part is negative definite on
     `T_J`;
   - `ess_status = "rejected"` when the within-support quadratic condition
     fails;
   - `ess_status = "undetermined"` when unused pure strategies tie and the
     simple certificate is therefore incomplete.
6. Retain `is_ess` only as a backward-compatible alias for
   `ess_status == "certified"`. State explicitly that `False` is not a complete
   non-ESS decision when `ess_status == "undetermined"`.

Do not call a Jacobian test an ESS test.

### 3. Report local dynamic stability separately

If the existing Jacobian diagnostic is retained:

- use an orthonormal tangent basis;
- treat the pure-support within-support tangent as zero-dimensional;
- report it under a separate key such as
  `is_locally_asymptotically_stable`;
- include transverse invasion eigenvalues for unused strategies;
- do not infer ESS from this flag and do not use it as a significance statistic.

If a complete and mathematically justified local-stability repair cannot be
finished within this task, remove the claim and return the underlying
eigenvalues without a stability verdict. Do not guess.

### 4. Regression tests

Add tests for at least:

- a strict pure ESS;
- invariance of Nash, ESS certification, and stability diagnostics under
  `A -> A + 1 c.T`;
- a pure-strategy example with an off-support tie that is marked
  `undetermined`, not falsely certified;
- neutral rock-paper-scissors: full-support Nash, not ESS-certified;
- a standard mixed ESS such as Hawk-Dove;
- an orthonormal tangent basis (`Q.T @ Q == I`);
- correct zero-dimensional tangent handling for pure supports;
- `surrogate_ess_frequency` emits `DeprecationWarning`;
- no README or mathematical-background text describes ESS presence as a
  calibrated hypothesis test.

Do not use the random-matrix ESS base rate as a unit-test oracle.

### 5. Compatibility and scope

- Preserve the public import path `ts2eg.find_ESS`.
- It is acceptable to add clearer helpers or a new descriptive function and
  make `find_ESS` a compatibility wrapper.
- Do not alter NMF, payoff induction, operator estimation, or notebook data
  generation except where wording must change to remove ESS inference.
- Do not claim operator recovery, calibrated size, power, or causal
  identification.

## Acceptance checks

Run from the repository root:

```bash
python -m pip install -e .
pytest -q
MPLBACKEND=Agg TS2EG_CI=1 \
  python scripts/run_notebooks.py --tag fast --timeout 120
```

Also run a repository text search and report every remaining occurrence of:

```text
ESS p-value
ESS indicator
reject H0
surrogate significance
evidence for strategic interaction
```

Each remaining occurrence must either be removed or explicitly identified as a
historical/deprecated statement.

## Final report

Report:

- touched files;
- exact API changes and deprecations;
- the mathematical ESS certificate implemented;
- how off-support ties are represented;
- exact test and notebook results;
- remaining limitations.

Do not begin Milestone 3.
