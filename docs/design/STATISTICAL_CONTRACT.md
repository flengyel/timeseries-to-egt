# Statistical contract

## 1. Estimand

For mixtures `x(t) in Delta_k`, the continuous replicator equation is

\[
\dot x_i=x_i\big((Ax)_i-x^TAx\big).
\]

Let

\[
M_k=I_k-\frac1k\mathbf 1\mathbf 1^T.
\]

The dynamically identified canonical operator is

\[
B=M_kA,
\qquad \mathbf 1^TB=0.
\]

The transformation `A -> A + 1 c.T` is payoff-equivalent. Right-centering is not.

For interior mixtures define centered log-ratio coordinates

\[
z(t)=M_k\log x(t).
\]

Then

\[
\dot z(t)=B x(t).
\]

A discrete estimator may use

\[
\frac{z(t+1)-z(t)}{\Delta t}\approx Bx(t),
\]

with approximation error stated and tested.

## 2. Distinct hypotheses

Do not collapse these hypotheses:

- `H0-coupling`: no cross-strategy replicator coupling beyond the specified
  baseline process;
- `H0-antisymmetric`: the antisymmetric component of the canonical operator is
  zero while symmetric interaction may remain;
- `H0-stability`: a particular fitted rest point is not dynamically stable;
- `H0-causal-exclusion`: an intervention on an exclusion variable has no effect.

They require different null generators and different statistics.

## 3. Statistics

Primary dynamic statistics are continuous:

- held-out motion log likelihood or residual improvement over a baseline;
- held-out vector-field `R^2` with an explicitly defined denominator;
- canonical operator or vector-field recovery error in simulation;
- nested-model improvement for symmetric versus full operators.

ESS presence is not a test statistic.

## 4. Aggregate and relative channels

For player-level payoff vector `v(t)`, return

\[
\bar v(t)=\pi^Tv(t),
\qquad
v_Z(t)=v(t)-\bar v(t)\mathbf 1.
\]

The aggregate channel measures common creation/destruction under the chosen payoff
definition. The relative channel measures advantage relative to the weighted mean.
Neither substitutes for the other.

## 5. Operator comparison

Compare `B_hat` with `B_true` only after:

1. left canonicalization;
2. simultaneous strategy permutation;
3. optional positive time-scale fitting when the observation interval is not fixed.

Never identify `B` with `-B`. Prefer held-out vector-field error to raw matrix
correlation.

## 6. Selection and inference

A valid selected-model test uses three partitions:

- training: preprocessing, basis fitting, and operator fitting;
- validation: seed, hyperparameter, and strategy-count selection;
- test: one final score.

Every null replicate repeats the entire pipeline, including preprocessing, payoff
induction, basis fitting, selection, and final scoring.

## 7. Causal language

Forecast-error reduction and information gain are predictive quantities. Causal
claims require a separately specified intervention, natural experiment, randomized
variation, or structural identification argument.
