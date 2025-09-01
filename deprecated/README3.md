# Time-Series → Evolutionary Games Toolkit

Turn multivariate time series into normal-form games, fit an evolutionary payoff operator \(A\), and analyze stability via replicator dynamics (Nash/ESS).

## Modules

- `gameify_timeseries_finance_biology_socio_cyber.py` — core transforms and games (Options A/B), NMF helper, Helmert and unweighted projectors, `find_ESS`.
- `egt_extensions.py` — add-ons:
  - Weighted projectors `weighted_projectors(weights)`.
  - Seasonal VAR Option B `var_information_sharing_game_seasonal(...)`.
  - Weighted A-estimation `estimate_A_from_series_weighted(...)`.
  - IAAFT surrogates `iaaft`, `iaaft_matrix`, and harness `surrogate_ess_frequency(...)`.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
import gameify_timeseries_finance_biology_socio_cyber as gm
import egt_extensions as ext

# X: N x T multivariate series
# Build strategy basis S via NMF on rectified X
X0 = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-9)
S, H = gm.nmf_on_X(X0, k=3, iters=200, seed=2, normalize="l2")

# Weighted A estimation
X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
w = np.ones(X.shape[0])  # or area weights, etc.
est = ext.estimate_A_from_series_weighted(S, X_std, X_std, k=S.shape[1], lambda_=1e-3, weights=w)
A = est["A"]

# ESS / Nash
ess_list = gm.find_ESS(A, tol=1e-8, max_support=S.shape[1])

# Seasonal information-sharing game
res = ext.var_information_sharing_game_seasonal(
    X, p=2, ridge=1e-3, seasonal_period=12, seasonal_multiples=[1,2],
    include_self_always=True, lambda_common=0.0, weights=w
)
```

## Surrogate-based significance

```python
summ = ext.surrogate_ess_frequency(S, X_std, X_std, k=S.shape[1], lambda_=1e-3,
                                   num_surrogates=50, iaaft_iters=100, weights=w, seed=0)
print(summ)  # {'ess_rate': ..., 'nash_rate': ..., 'R2_mean': ..., 'R2_std': ...}
```

## Tests

Run all tests:

```bash
pytest -q
```

Continuous integration runs via GitHub Actions on Python 3.9–3.12.

## Design notes

- **Geometry.** Common-interest vs zero-sum split via projectors; weighting supported through `weighted_projectors(weights)`.
- **Replicator invariance.** Estimation enforces column-sum centering of \(A\) in strategy space.
- **Seasonality.** Seasonal VAR adds lag blocks at \(s, 2s, ...\) in addition to 1..p.
- **Nulls.** IAAFT surrogates preserve each series’ marginal and spectrum while breaking cross-player coupling.

## Climate use-case (example)

1. Work with anomalies (remove trends/seasonality).
2. Choose weights (e.g., area: cos(latitude)).
3. Build `S` via NMF or EOF-derived components.
4. Fit `A` in rolling windows; test ESS frequency against IAAFT surrogates.
5. Attribute by removing candidate drivers (e.g., ENSO) and refitting.

## Caution

An ESS in the fitted model indicates stable strategic structure in the representation, not causality. Validate with surrogates, rolling windows, and predictive checks.
