# Validation matrix

## CI-fast mathematical tests

### Geometry

- `field(A, x) == field(A + 1 c.T, x)` for random `A,c,x`.
- `field(A, x) == field(M @ A, x)`.
- At least one seeded random case verifies
  `field(A @ M, x) != field(A, x)`.
- Strategy permutation equivariance.
- A sign reversal changes the field except in degenerate cases.

### Equilibria

- strict pure ESS is invariant under `A -> A + 1 c.T`;
- rock-paper-scissors full-support Nash is not classified as ESS in the neutral case;
- a standard mixed ESS is recognized;
- tangent restriction uses an orthonormal basis;
- pure supports use the zero-dimensional within-support tangent correctly.

### No leakage

- preprocessing fitted on training data is unchanged when held-out values change;
- a fixed basis is used to infer held-out mixtures;
- no factorization is fitted on validation/test data.

## Statistical smoke tests

These detect gross regressions but do not establish calibration.

- oracle mixtures, low-noise random canonical operators, correct sign;
- recovery after strategy permutation;
- symmetric, antisymmetric, mixed, and strategy-handicap generators;
- operator error worsens when noise is increased substantially;
- aggregate channel separates transfer from destruction in stipulated generators.

## Slow calibration suite

Run outside ordinary CI and save machine-readable results.

### Oracle mixtures

For `k=2..5`, multiple operator classes, lengths, and noise levels:

- canonical operator recovery;
- vector-field recovery;
- empirical size under a genuine no-coupling generator;
- power against symmetric and antisymmetric alternatives separately.

Use at least hundreds of realizations per reported size/power cell and report Monte
Carlo confidence intervals.

### Known basis

- infer held-out mixtures against a fixed true loading matrix;
- report mixture error after permutation;
- quantify attenuation in operator recovery.

### Latent basis

- compare reconstruction-only selection with joint reconstruction/dynamics selection;
- select `k` and hyperparameters on validation only;
- repeat selection inside every null replicate;
- do not expose an inferential p-value until size and power are acceptable.

## Exclusion-game recovery

Generate known exclusion variables, aggregate destruction, relative transfers, and
canonical dynamics. Test separately whether the pipeline recovers:

- aggregate destruction;
- relative payoff advantage;
- dynamic operator changes;
- externally identified intervention effects.

A time-series game fit alone is not permitted to identify the blocker map.
