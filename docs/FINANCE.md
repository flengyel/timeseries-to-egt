# Financial Analysis with `gamify-timeseries`

This note explains how to apply the pipeline to public-company financials.

## What you model

Treat line items as **players** and quarterly history as a multivariate series:
- COGS, SG&A, R&D, CapEx, Net Working Capital change, etc.
- Normalize to **shares of revenue** (common-size) or **shares of operating cash inflow**.
- Keep sign conventions (costs negative if not using shares).

**Avoid derived ratios** (e.g., EBITDA margin) that are algebraic combinations of the above.

## Remove accounting identities

Some rows obey deterministic identities (bookkeeping). Build a constraint matrix `C` (rows are identities) and project onto the **nullspace**:

```python
from gameify_timeseries import project_out_identities
X_perp = project_out_identities(X, C)  # shape N x T
```

This prevents the game from “discovering” trade-offs that are just accounting.

## Pick a value target and build payoffs

Choose a scalar **value target** `J_future` (e.g., 1–4Q-ahead TSR, EPS growth, ROIC change). Construct per-item **payoff vectors** as a **value gradient**:

```python
from gameify_timeseries import value_gradient_payoffs
v = value_gradient_payoffs(X_perp, J_future, model='ridge', ridge=1e-3, standardize=True)  # N x T
```

Interpretation: `v_i(t)` is the marginal contribution of item *i* to `J_future` (linear ridge: time-invariant Beta, repeated over T).

## Learn strategies and fit an evolutionary game

1) Learn `S` (items × K) via nonnegative matrix factorization (NMF) or archetypal analysis (outside this module). Columns are **capital allocation archetypes**.

2) Estimate a **strategy-level payoff operator** `A` and search for ESS:

```python
from gameify_timeseries import estimate_A_from_series, find_ESS
est = estimate_A_from_series(S, X_perp, v, k=K, ridge=1e-2)
A = est["A"]
ess = [r for r in find_ESS(A, tol=1e-8, max_support=K) if r["is_ess"]]
```

An ESS is a **stable mixture of archetypes** under the learned payoff field tied to your value target.

## Practical guidance

- **Seasonality**: work with YoY differences or include seasonal fixed effects.
- **Outliers**: winsorize large one-offs (impairments, tax items).
- **Sample size**: with quarterly data, keep `K` small (3–6) and use ridge.
- **Diagnostics**: report `R^2` for the strategy-signal fit, symmetric vs. skew decomposition of `A`, rolling-window stability, and null checks (time shuffles).
- **Interpretation**: “ESS exists” ≠ “good management.” Validate with event windows and out-of-sample `J_future` performance.

## Minimal notebook

See `finance_demo.ipynb` for a runnable synthetic example that you can adapt to real 10-Q/10-K data.

## License

MIT.
