# ts2eg v2 redesign execution plan

## Status

Proposed. Execute one milestone per pull request. Do not skip validation gates.

## Milestone 1: lock the geometry

- add canonical left-centering helpers and replicator-field tests;
- remove right-centering from current estimators;
- return raw and canonical operators explicitly;
- correct documentation of payoff equivalence;
- add a changelog entry.

Exit condition: all geometry invariance tests pass; no code path presented as
canonical applies right-centering.

## Milestone 2: retire ESS inference and repair ESS description

- mark ESS frequency and ESS-presence p-values deprecated;
- remove inferential claims from README, docs, and demos;
- fix pure-support and tangent-space calculations in `find_ESS`;
- retain ESS only as a descriptive property of a fitted operator.

Exit condition: known-game and payoff-equivalence tests pass.

## Milestone 3: oracle-mixture motion estimator

- implement centered-log-ratio motion estimation;
- return canonical operator, diagnostics, and held-out score;
- add true-mixture generators for null, symmetric, antisymmetric, mixed, and
  handicap cases;
- diagnose sign with vector-field recovery, not matrix correlation.

Exit condition: oracle recovery succeeds across operator classes at low noise and
fails gracefully near boundaries.

## Milestone 4: nulls and calibration for observed mixtures

- define distinct no-coupling and zero-antisymmetric hypotheses;
- implement appropriate bootstrap/surrogate generators in log-ratio coordinates;
- add a slow Monte Carlo calibration harness with confidence intervals;
- no ESS statistic.

Exit condition: documented empirical size and monotone power for the oracle mode.

## Milestone 5: known-basis measurement model

- introduce a fitted preprocessing object;
- infer held-out mixtures using a frozen loading matrix;
- remove scaling inconsistencies between NMF fitting and mixture inference;
- quantify measurement-error attenuation.

Exit condition: known-basis validation passes without leakage.

## Milestone 6: latent dynamic representation (experimental)

- specify and prototype a joint objective for observation reconstruction and
  centered-log-ratio dynamic residual;
- normalize loadings and state the remaining identifiability class;
- use training/validation/test partitions;
- treat NMF-seed selection as exploratory until full nested validation passes.

Exit condition: synthetic latent-basis recovery improves over reconstruction-only
selection and has no hidden test-set use.

## Milestone 7: full-pipeline inference

- repeat payoff induction, preprocessing, basis fitting, `k` selection, operator
  fitting, and scoring inside every null replicate;
- report computational cost and Monte Carlo uncertainty;
- expose an inferential API only after size/power gates pass.

## Milestone 8: dual payoff/motion and aggregate channels

- return aggregate and relative payoff channels together;
- distinguish payoff-field estimation from motion-field estimation;
- add an agreement diagnostic between the two canonical operators;
- document predictive versus causal interpretations.

## Milestone 9: exclusion-game benchmark

- add a ground-truth generator or adapter with known exclusion, transfer,
  destruction, and dynamic effects;
- test recovery of each channel separately;
- connect to the capital-labor notebook only after the synthetic recovery gates pass.
