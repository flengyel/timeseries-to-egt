# CYBERSECURITY: Exploratory Detector-Portfolio Payoff Fields

This note records security input and payoff conventions that can exercise the
current legacy `ts2eg` compatibility path. It does not recommend that path as a
detector for detector portfolios, deception assets, or response workflows.

The legacy pipeline is scientifically flawed as a detection procedure and has no
validated ability to detect security dynamics, interactions, campaigns, regimes,
cycles, ESS in data, or operational effects.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator: supplied mixtures are transformed to centered log-ratio increments and
used to fit a canonical motion operator. Nothing in the current compatibility
path is a substitute for that estimator.

> **Current status.** The path is descriptive. NMF supplies reconstruction
> factors, not validated replicator coordinates. `estimate_A_from_series` fits a
> contemporaneous payoff field rather than mixture motion, and its `R2` is
> in-sample. ESS and local-stability records describe the fitted operator only.
> No cybersecurity null test or control policy is validated at this milestone.

## Observation design

- Rows may represent detector or rule families, deception-asset classes, or
  response-workflow components.
- Let `X` have shape `(N, T)`. If columns are normalized shares of alerts,
  compute, or analyst time, record that normalization as part of the observation
  model.
- `nmf_on_X` may summarize `X` by nonnegative reconstruction factors. The legacy
  estimator ignores the returned `H`, normalizes the loading matrix, and
  independently re-infers simplex coefficients. Calling these factors
  operational strategies requires external interpretation; NMF reconstruction
  alone does not justify it.
- Use the same fitted preprocessing and scale for basis learning and membership
  inference. A predictive evaluation must fit preprocessing and the basis on
  training data only.

## Predictive payoff construction

For a future operational target `J_future`, such as a prespecified incident-cost
or service-level measure,

```python
v = info_gain_payoffs(X, J_future, ridge=1e-2, window=168)
```

computes leave-one-feature-out ridge MSE differences on each fitting window. It
does not compute information-theoretic information, and the current helper does
not score a frozen model on held-out data. Its values are in-sample predictive
associations, not causal effects of detectors, rules, or response actions.

Strictly align each target at horizon `H` with information available at the
prediction time. Include false-positive, analyst-time, and compute costs only
through an explicitly defined target.

## Aggregate and relative channels

For player-level payoffs,

\[
v(t)=\bar v(t)\mathbf 1+v^Z(t),\qquad
v^Z(t)=v(t)-\bar v(t)\mathbf 1.
\]

The aggregate channel and mean-zero relative channel answer different questions.
The legacy estimator discards the aggregate channel and fits only the relative
one. It therefore cannot measure the total benefit or cost created by a detector
portfolio. Mean centering does not establish a zero-sum security interaction.

## Legacy descriptive operator

The estimator fits

\[
G_c\approx A X_k
\]

at the same times. It does not fit changes in `X_k`, so `A` is not an estimated
motion law or a validated controller. Its symmetric and antisymmetric parts are
algebraic summaries of the payoff-field fit. A large antisymmetric norm does not
establish attack-defense rotation or operational cycles.

`find_ESS(A)` reports descriptive equilibrium and local-linearization records.
It skips singular KKT systems and may omit continua of equilibria. ESS
persistence, rolling changes in `A`, or changes in its skew ratio do not
establish campaign changes or operational regimes.

## Minimal descriptive example

```python
from ts2eg import (
    estimate_A_from_series,
    find_ESS,
    info_gain_payoffs,
    nmf_on_X,
)

# X: N x T nonnegative usage shares; J_future: a prespecified aligned target
v = info_gain_payoffs(X, J_future, ridge=1e-2, window=168)
S, _ = nmf_on_X(X, k=4, iters=300, seed=1)
est = estimate_A_from_series(S, X, v, k=4, ridge=1e-2)

A = est["A"]
in_sample_R2 = est["R2"]
records = find_ESS(A)
certified = [r for r in records if r["ess_status"] == "certified"]
```

This code executes the legacy descriptive path only. Circular row shifts,
calendar permutations, attack-free comparisons, and rolling fits are possible
exploratory summaries, but no statistic based on them is calibrated here.
Rule changes or red-team exercises support causal conclusions only through a
separately specified intervention design. A controller would additionally
require independently validated motion dynamics and control-safety analysis.

See `notebooks/CYBER_DEMO.ipynb` for a stipulated synthetic demonstration. It
does not validate an empirical security analysis.
