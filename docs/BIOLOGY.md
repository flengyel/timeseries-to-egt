# BIOLOGY: Exploratory Payoff-Field Applications

This note records biological input and payoff conventions that can exercise the
current legacy `ts2eg` compatibility path. It does not recommend that path as a
detector or establish that inferred latent coordinates follow replicator
dynamics.

The legacy pipeline is scientifically flawed as a detection procedure and has no
validated ability to detect biological dynamics, interactions, regimes, cycles,
ESS in data, or treatment effects.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator: supplied mixtures are transformed to centered log-ratio increments and
used to fit a canonical motion operator. Nothing in the current compatibility
path is a substitute for that estimator.

> **Current status.** NMF is available as a reconstruction tool, not as a
> validated selector of replicator coordinates. `estimate_A_from_series` fits a
> contemporaneous payoff field, not mixture motion, and its `R2` is in-sample.
> Nash, ESS-certificate, and local-stability outputs are descriptive properties
> of that fitted operator. No biological null test is validated at this
> milestone.

## Observations and latent components

- Rows may represent microbial taxa or guilds, phytoplankton groups, immune-cell
  modules, tumor subclones, or pathway modules.
- Let `X` have shape `(N, T)`. Counts and relative abundances answer different
  questions and should not be interchanged silently.
- `nmf_on_X` produces nonnegative reconstruction factors. The legacy estimator
  does not use the returned coefficient matrix `H`; it normalizes the loading
  matrix and independently re-infers simplex coefficients for each observation.
  These coefficients are exploratory latent components, not identified
  ecological strategies.
- Fit the basis and every preprocessing transformation on the same data scale.
  An inferential analysis would additionally fit them on training data only and
  freeze them before validation or test evaluation.

## Candidate payoff definitions

### Per-capita growth

For strictly positive counts or biomass,

\[
v_i(t)=\frac{\log n_i(t+1)-\log n_i(t)}{\Delta t}.
\]

`growth_payoffs(counts, dt, pad)` computes this finite-difference series. Its
small numerical offset for zero counts is a computational convention that must
be reported and given a biological interpretation.

Relative abundances alone do not determine absolute per-capita growth. Writing
`n_i(t) = p_i(t) N_total(t)` is valid only when total abundance is observed or
otherwise justified. Without it, use an explicitly relative log-ratio analysis
and do not claim recovery of aggregate growth.

### Predictive ablation

`info_gain_payoffs(X, J_future, ridge, window)` computes leave-one-feature-out
ridge MSE differences for a chosen future endpoint. The current helper fits and
scores on the same global or rolling sample; its global output is constant over
time. The result is an in-sample predictive association, not
information-theoretic information, causal contribution, or biological fitness.

## Aggregate and relative channels

For a player-level payoff vector, write

\[
v(t)=\bar v(t)\mathbf 1+v^Z(t),\qquad
v^Z(t)=v(t)-\bar v(t)\mathbf 1.
\]

The aggregate channel and mean-zero relative channel carry different
information. The legacy estimator passes only `v^Z` forward and discards the
aggregate channel. It therefore cannot measure aggregate creation, destruction,
or population growth. Mean centering is a channel definition; it is not evidence
of competition, coordination, or a zero-sum game.

## Legacy descriptive fit

The current fit estimates the contemporaneous relation

\[
G_c\approx A X_k.
\]

It does not use changes in `X_k` and does not estimate an operator that generated
mixture motion. The symmetric and antisymmetric parts of `A` are algebraic
summaries of the fitted payoff field. A large antisymmetric norm does not by
itself establish ecological cycles.

`find_ESS(A)` reports descriptive Nash, ESS-certificate, and local linearization
records. The support enumerator skips singular KKT systems and may omit continua
of equilibria. These records do not establish observed stability, basin size,
treatment response, or regime change.

## Minimal descriptive example

```python
from ts2eg import estimate_A_from_series, find_ESS, growth_payoffs, nmf_on_X

# counts: strictly positive N x T abundance data
v = growth_payoffs(counts, dt=1.0)
X = counts / counts.sum(axis=0, keepdims=True)

# Exploratory reconstruction; the estimator re-infers its own simplex coefficients.
S, _ = nmf_on_X(X, k=4, iters=300, seed=1)
est = estimate_A_from_series(S, X, v, k=4, ridge=1e-2)

A = est["A"]
in_sample_R2 = est["R2"]
records = find_ESS(A)
certified = [r for r in records if r["ess_status"] == "certified"]
```

This example executes the legacy descriptive path only. Circular shifts, phase
scrambles, label permutations, rolling fits, or changes in ESS records have no
calibrated inferential interpretation here. Diet changes, therapies, or other
interventions require a separately specified causal design.

See `notebooks/BIO_DEMO.ipynb` for a stipulated synthetic demonstration. It does
not validate an empirical biological analysis.
