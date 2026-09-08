# ts2eg architecture

## Status and thesis

The repository is at the Milestone-2 closeout. The implemented package remains
a scientifically flawed legacy compatibility surface for constructing payoff
fields, fitting a contemporaneous payoff-field regression, and computing
properties of the resulting fitted matrix. It has no validated detection
capability. In particular, its fit and diagnostics do not detect dynamics,
strategic interaction, regimes, cycles, ESS in data, or domain effects.

The package does not yet implement the inferential motion-estimation pipeline in
the version-2 roadmap. That roadmap separates payoff construction, measurement,
evolutionary-dynamics estimation, and statistical validation so each layer can
eventually be validated independently. The target modules below describe
Milestone 3 and later work; they are not current public capabilities.

## Current compatibility behavior

### Geometry

`ts2eg.geometry` currently provides the centering projector, canonical payoff
operator, and replicator vector field. For a `k x k` operator,

```text
M = I - 11.T / k
B = M @ A
```

is the canonical representative. Only left-centering is an identification
normalization. Right- or double-centering is not imposed because it changes the
replicator field in general.

### Legacy payoff-field regression

`estimate_A_from_series` and its weighted extension compute the contemporaneous
field fit

```text
Gc ~= A_raw @ Xk
```

with a no-intercept ridge regression and return `A = M @ A_raw`. The routines
normalize the columns of `S` and re-infer each simplex vector `Xk[:, t]` from
`X[:, t]` by projected gradient. They do not consume an NMF coefficient matrix
and contain no amplitude term. Observations passed for re-inference therefore
need the same preprocessing scale used to fit `S`.

Before fitting, the legacy path player-centers the payoff field and discards its
aggregate component. The weighted variant changes that player-space centering
only; it does not weight reconstruction or mixture inference. This path cannot
identify aggregate creation or destruction.

The returned `R2` is an in-sample algebraic diagnostic relative to the zero
payoff field, not a held-out score or detection statistic. It is undefined and
returned as `NaN` when the centered target field has zero sum of squares.

This computation is not a mixture-motion estimator, does not show that mixtures
follow replicator dynamics, and supports no calibrated inferential or causal
conclusion. NMF remains a reconstruction tool, not a validated selector of
replicator coordinates.

### Payoffs, equilibria, and surrogates

The static, predictive, growth, information-gain, and value-gradient payoff
builders provide stipulated or associational numerical payoff fields. They have
no validated domain-effect detection interpretation. Predictive payoffs do not
establish rent, extraction, welfare, appropriation, or causal influence without
an external causal design.

`find_ESS` computes Nash, ESS, and local-stability properties of a supplied or
fitted operator; it does not detect ESS in observed data. A result may remain
undetermined in degenerate cases. ESS presence is not a significance statistic.
`surrogate_ess_frequency` remains a deprecated, uncalibrated legacy computation
and must not be presented as a p-value or hypothesis test.

## Identification tiers

1. `observed_mixtures`: mixtures are supplied by the application. This is the
   first mode targeted for inferential support, after its validation gates pass.
2. `known_basis`: a fixed observation model is supplied and held-out mixtures
   are inferred with frozen preprocessing and loadings.
3. `latent_dynamic`: loadings, mixtures, and dynamics are estimated jointly;
   this remains experimental until the validation ladder passes.
4. `causal_application`: an external intervention design and defensible
   exclusion assumptions are added. No lower tier implies this tier.

The current legacy payoff-field estimators do not implement the planned
observed-mixture motion estimator and therefore do not satisfy tier 1 merely
because they return simplex-valued `Xk`.

## Milestone-3-and-later target layers

### Observed-mixture dynamics

Input: application-supplied `x(t)` in the simplex.

- centered-log-ratio motion with an explicit boundary policy;
- motion-based canonical-operator estimation;
- held-out dynamic scoring;
- descriptive equilibrium analysis;
- null and power experiments defined by the statistical contract.

These capabilities become inferential only after their validation gates pass.

### Domain payoff fields

- domain-specified payoff series;
- predictive payoff builders explicitly labeled predictive;
- aggregate and relative channels returned together;
- optional comparison of payoff-field and motion-field operators.

### Measurement models

Known-basis mode will use train-only preprocessing, a fixed loading matrix,
held-out mixture inference, and reconstruction/mixture diagnostics. Latent-basis
mode remains experimental until its validation ladder passes.

### Simulation and validation

- generators with known mixtures and payoff operators;
- independent/null, symmetric, antisymmetric, mixed, handicap, and regime-change
  alternatives;
- canonical-operator and vector-field recovery metrics;
- Monte Carlo calibration and power reports with uncertainty.

## Target module layout

```text
src/ts2eg/
  geometry.py
  dynamics.py
  payoffs.py
  measurement.py
  equilibria.py
  surrogates.py
  simulation.py
  validation.py
  legacy.py
```

Only `geometry.py` from this conceptual split exists as a dedicated module at
the Milestone-2 closeout. The exact future file split may change, but the
conceptual separation may not.
