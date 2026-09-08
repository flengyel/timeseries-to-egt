# FINANCE: Exploratory Corporate Payoff Fields

This note records financial input and payoff conventions that can exercise the
current legacy `ts2eg` compatibility path. It does not recommend that path as a
detector for public-company financial series.

The legacy pipeline is scientifically flawed as a detection procedure and has no
validated ability to detect financial dynamics, interactions, regimes, cycles,
ESS in data, management quality, or allocation effects.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator: supplied mixtures are transformed to centered log-ratio increments and
used to fit a canonical motion operator. Nothing in the current compatibility
path is a substitute for that estimator.

> **Current status.** NMF is an exploratory reconstruction method, not a
> validated selector of replicator coordinates. `estimate_A_from_series` fits a
> contemporaneous payoff field, not mixture motion, and returns an in-sample
> `R2`. Nash, ESS-certificate, and local-stability records describe only the
> fitted operator. No financial null or causal test is validated at this
> milestone.

## Observation design

Rows may represent consistently defined line items such as labor compensation,
COGS, SG&A, R&D, capital expenditure, working-capital change, or payouts. Record
units, signs, restatements, and the decision to use levels, changes, or shares.
Avoid silently including ratios that are deterministic functions of other rows.

Compositional shares are collinear by construction. Ridge coefficients then
depend on parameterization, scaling, and regularization and should not be
interpreted as invariant marginal effects.

## Accounting identities

`project_out_identities(X, C)` removes components in the row space of homogeneous
linear constraints `C`. This may be useful for diagnosing bookkeeping
redundancy, but the projection can create negative values and destroy column-sum
normalization.

Do not silently learn an NMF basis from such projected data. `nmf_on_X` clips
negative inputs, whereas membership inference later uses the supplied observation
matrix. Applying those operations to differently transformed data creates a
measurement mismatch. Use a separately specified nonnegative observation matrix
for reconstruction and pass that same matrix to the membership estimator.

## Predictive coefficient payoff

For a prespecified future target `J_future`,

```python
v = value_gradient_payoffs(
    X_model,
    J_future,
    model="ridge",
    ridge=1e-3,
    standardize=True,
)
```

fits one pooled ridge model

\[
J_{\mathrm{future}}\approx\alpha+\beta^\top X_{\mathrm{model}}
\]

on the full supplied sample and returns `v_i(t) = beta_i` at every time. The
result is a time-constant in-sample coefficient association under the chosen
scaling and penalty. It is not a causal marginal contribution, structural value
gradient, or optimizer. A predictive claim requires train-only fitting and
genuinely held-out scoring, which this helper does not provide.

## Aggregate and relative channels

Write a player-level payoff as

\[
v(t)=\bar v(t)\mathbf 1+v^Z(t).
\]

The aggregate component and mean-zero relative component are distinct. The
legacy estimator discards the aggregate component and fits only `v^Z`. It cannot
therefore measure total value creation or destruction, and mean centering does
not establish zero-sum redistribution.

## Legacy descriptive fit

`estimate_A_from_series` fits the contemporaneous relation

\[
G_c\approx A X_k.
\]

It does not use mixture changes and does not show that replicator dynamics
generated the observations. Symmetric/skew decompositions, rolling estimates,
and equilibrium records are algebraic or descriptive summaries only. They do
not establish management quality, allocation regimes, or strategic cycles.

`find_ESS` skips singular KKT systems and may omit continua of equilibria. Its
ESS and local-stability fields must not be used as hypothesis tests.

## Minimal descriptive example

```python
from ts2eg import (
    estimate_A_from_series,
    find_ESS,
    nmf_on_X,
    value_gradient_payoffs,
)

# X_alloc: nonnegative N x T allocation shares
v = value_gradient_payoffs(X_alloc, J_future, ridge=1e-3)

K = 4
S, _ = nmf_on_X(X_alloc, k=K, iters=300, seed=1)
est = estimate_A_from_series(S, X_alloc, v, k=K, ridge=1e-2)

A = est["A"]
in_sample_R2 = est["R2"]
records = find_ESS(A, max_support=K)
certified = [r for r in records if r["ess_status"] == "certified"]
```

This code executes the legacy descriptive payoff-field path only. Time shuffles,
rolling windows, event windows, or lagged targets do not by themselves validate
a null, a motion model, or a causal interpretation. Causal attribution requires
a separately defended intervention, natural experiment, randomized design, or
structural identification argument.

See `notebooks/FINANCE_DEMO.ipynb` for a stipulated synthetic demonstration. It
does not validate an empirical financial analysis.
