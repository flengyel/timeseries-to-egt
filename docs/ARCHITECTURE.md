# ts2eg v2 architecture

## Thesis

The present pipeline combines four logically different tasks: payoff construction,
latent representation learning, evolutionary-dynamics estimation, and statistical
testing. Version 2 separates them so that each layer can be validated independently.

## Stable public layers

### 1. Geometry

`ts2eg.geometry`

- player and strategy centering projectors;
- canonical payoff operator `B = M @ A`;
- replicator vector field;
- operator alignment under strategy permutations and positive time scaling;
- aggregate/relative payoff decomposition.

### 2. Observed-mixture dynamics

`ts2eg.dynamics`

Input: `x(t) in Delta_k` supplied by the application.

- centered log-ratio motion;
- motion-based operator estimation;
- held-out dynamic scoring;
- descriptive equilibrium analysis;
- null and power experiments defined in the statistical contract.

This is the first inferentially supported mode.

### 3. Domain payoff fields

`ts2eg.payoffs`

- domain-specified payoff series;
- predictive payoff builders, explicitly labeled predictive;
- aggregate and relative channels returned together;
- optional comparison between payoff-field and motion-field operators.

### 4. Measurement models

`ts2eg.measurement`

Known-basis mode:

- train-only preprocessing;
- fixed loading matrix;
- held-out mixture inference;
- reconstruction and mixture diagnostics.

Latent-basis mode is experimental until the validation ladder passes.

### 5. Simulation and validation

`ts2eg.simulation`, `ts2eg.validation`

- generators with known mixtures and payoff operators;
- independent/null, symmetric, antisymmetric, mixed, handicap, and regime-change
  alternatives;
- operator and vector-field recovery metrics;
- Monte Carlo calibration and power reports.

## Compatibility layer

The current functions may remain temporarily, but their interpretation changes:

- `estimate_A_from_series` becomes a payoff-field regression and returns a
  left-centered canonical operator; it is not a motion estimator.
- `find_ESS` remains descriptive after its tangent-space implementation is fixed.
- `surrogate_ess_frequency` is deprecated as an inferential API.
- NMF remains available as a reconstruction tool, not as a validated method for
  selecting replicator coordinates.

## Data tiers

1. `observed_mixtures`: strongest identification; no latent basis.
2. `known_basis`: measurement error is evaluated against known/fixed loadings.
3. `latent_dynamic`: joint observation/dynamics estimation; experimental.
4. `causal_application`: adds an external intervention design and exclusion
   variables. No lower tier implies this tier.

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

The exact file split may change, but the conceptual separation may not.
