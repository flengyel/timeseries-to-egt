# SOCIOLOGY: Exploratory Social Payoff Fields

This note records social input and payoff conventions that can exercise the
current legacy `ts2eg` compatibility path. It does not recommend that path as a
detector for coalitions, collective action, online governance, diffusion, or
organizational allocation.

The legacy pipeline is scientifically flawed as a detection procedure and has no
validated ability to detect social dynamics, interactions, regimes, cycles, ESS
in data, stable coalitions, or intervention effects.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator: supplied mixtures are transformed to centered log-ratio increments and
used to fit a canonical motion operator. Nothing in the current compatibility
path is a substitute for that estimator.

> **Current status.** NMF supplies reconstruction factors, not validated social
> strategies or replicator coordinates. `estimate_A_from_series` fits a
> contemporaneous payoff field rather than mixture motion, and its `R2` is
> in-sample. Nash, ESS-certificate, and local-stability records are descriptive
> properties of that fitted operator. No social-science null test is validated at
> this milestone.

## Observation design

- Rows may represent groups, tactics, roles, or allocation channels. This choice
  is substantive and should not be inferred from matrix shape alone.
- Let `X` have shape `(N, T)`. Counts, shares, and normalized intensities have
  different interpretations; document every normalization.
- `nmf_on_X` provides nonnegative reconstruction factors. The legacy estimator
  ignores the returned `H`, normalizes the loading matrix, and independently
  re-infers simplex coefficients. Interpreting those factors as coalitions,
  tactics, or roles requires external evidence.
- Fit preprocessing and the basis on a common scale. Any predictive evaluation
  must fit them on training data and freeze them before evaluation.

## Candidate payoff definitions

### Per-capita growth

For strictly positive counts of members, votes, or events, one possible
descriptive payoff is

\[
v_i(t)=\frac{\log n_i(t+1)-\log n_i(t)}{\Delta t}.
\]

This quantity describes growth under the chosen counting process. It does not by
itself measure utility, social welfare, or causal advantage.

### Predictive ablation

For a prespecified future outcome `J_future`,
`info_gain_payoffs(X, J_future, ridge, window)` computes leave-one-feature-out
ridge MSE differences. The current implementation fits and scores on the same
global or rolling sample. Its output is an in-sample predictive association, not
information-theoretic information or a causal contribution to the outcome.

Cost-aware variants may subtract an explicitly defined cost before centering.
Do not combine payoff definitions within one reported fit or select a definition
because it gives a preferred equilibrium record.

## Aggregate and relative channels

For player-level payoffs,

\[
v(t)=\bar v(t)\mathbf 1+v^Z(t),\qquad
v^Z(t)=v(t)-\bar v(t)\mathbf 1.
\]

The aggregate and mean-zero relative channels answer different questions. The
legacy estimator discards the aggregate channel. It therefore cannot assess
aggregate mobilization, welfare, organizational production, or destruction.
Mean centering does not establish a zero-sum social interaction.

## Legacy descriptive operator

The current estimator fits

\[
G_c\approx A X_k
\]

at the same times; it does not fit changes in `X_k`. Thus `A` does not establish
an observed replicator motion law. The symmetric and antisymmetric parts of `A`
are algebraic summaries of the fitted payoff field. They do not establish a
scalar social objective, arms race, or rotational cycle.

`find_ESS(A)` reports descriptive equilibrium and local-linearization records.
It skips singular KKT systems and may omit continua. ESS records do not establish
stable coalitions, optimal moderation portfolios, robust role blends, or social
regimes.

## Minimal descriptive example

```python
from ts2eg import (
    estimate_A_from_series,
    find_ESS,
    growth_payoffs,
    nmf_on_X,
)

# counts: strictly positive N x T observations
v = growth_payoffs(counts, dt=1.0)
X = counts / counts.sum(axis=0, keepdims=True)

S, _ = nmf_on_X(X, k=4, iters=300, seed=1)
est = estimate_A_from_series(S, X, v, k=4, ridge=1e-2)

A = est["A"]
in_sample_R2 = est["R2"]
records = find_ESS(A)
certified = [r for r in records if r["ess_status"] == "certified"]
```

This code executes the legacy descriptive path only. Circular shifts, block
bootstraps, target permutations, rolling fits, and changes in equilibrium records
have no calibrated inferential interpretation here. Elections, policy changes,
or protests support causal conclusions only through a separately specified
research design.

Use aggregated or de-identified data when possible, and report weak,
undetermined, and negative descriptive results without converting them into
hypothesis-test conclusions.
