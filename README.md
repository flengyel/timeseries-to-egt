# timeseries-to-egt (ts2eg): From Multivariate Time Series to Evolutionary Games

This project **interprets a multivariate time series as an evolutionary (normal‑form) game** and then analyzes it with **replicator dynamics** to test for **Evolutionarily Stable Strategies (ESS)**. Concretely:

- From signals \(X \in \mathbb{R}^{N\times T}\), we construct per‑time **payoff vectors** and decompose them into **common‑interest** vs **zero‑sum** directions.
- We learn a small set of **strategies** \(S\in\mathbb{R}^{N\times k}\) and infer **mixtures** \(x(t)\in\Delta_k\).
- We fit a **strategy‑level payoff operator** \(A\) so that **replicator dynamics** apply.
- We search for **ESS** and use **IAAFT surrogates** to assess **significance**. Finding an ESS is treated as **evidence of strategic interaction** in the data.

**Package:** `ts2eg` (src‑layout). Canonical code: `src/ts2eg/core.py` and `src/ts2eg/extensions.py`.

---

## Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

Python \(\ge\) 3.9 recommended. CI runs `pytest` on 3.9–3.12.

---

## Repository layout (actual)

```text
.
├── pyproject.toml
├── README.md
├── requirements.txt
├── .github/workflows/ci.yml
├── src/
│   └── ts2eg/
│       ├── __init__.py
│       ├── core.py          # canonical pipeline
│       └── extensions.py    # weighted projectors, seasonal VAR, IAAFT harness
├── tests/
│   ├── test_projectors.py
│   ├── test_seasonal_var.py
│   ├── test_extensions_egt.py
│   └── test_package_import.py
├── notebooks/               # curated demos (each has a bootstrap cell)
├── docs/                    # domain notes (FINANCE.md, BIOLOGY.md, etc.)
└── examples/
    └── demo_var_game.py     # runnable end‑to‑end example
```

---

## Quick start (package‑only)

```python
import numpy as np
import ts2eg as gm
from ts2eg import extensions as ext

# X: N x T time series (players x time)
N, T = 5, 360
rng = np.random.default_rng(1)
X = rng.standard_normal((N, T))

# Seasonal information-sharing game with optional weights
w = np.ones(N)
res = ext.var_information_sharing_game_seasonal(
    X, p=2, ridge=1e-3, seasonal_period=12, seasonal_multiples=[1, 2],
    include_self_always=True, lambda_common=0.0, weights=w
)

# Strategy basis (NMF) and replicator operator A
X0 = (X - X.min(axis=1, keepdims=True)) / (np.ptp(X,axis=1, keepdims=True) + 1e-9)
S, H = gm.nmf_on_X(X0, k=3, iters=200, seed=2, normalize="l2")

X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
A = ext.estimate_A_from_series_weighted(S, X_std, X_std, k=3, ridge=1e-3, weights=w)["A"]

ess = [e for e in gm.find_ESS(A, tol=1e-8, max_support=3) if e["is_ess"]]

# Optional: surrogate significance via IAAFT
sig = ext.surrogate_ess_frequency(S, X_std, X_std, k=3, ridge=1e-3,
                                  num_surrogates=50, iaaft_iters=100, weights=w, seed=0)
print("ESS supports:", [e["support"] for e in ess])
print("Surrogate ESS rate:", sig["ess_rate"])
```

---

See derivations in [docs/ts2eg_math_background.tex](docs/ts2eg_math_background.tex).

## Why this exists

Many “inverse EGT” papers start from observed **strategy frequencies**. Real datasets are often **player‑level time series** instead. This repo provides a **coordinate pipeline** from player signals to strategies and payoffs:

1. Decompose per‑time payoffs into **common‑interest** vs **zero‑sum** directions (ANOVA/Helmert).
2. Induce payoffs from time series via two choices: static discretized profiles or **information‑sharing** gains (VAR‑like OLS).
3. Learn a compact **strategy basis** and infer time‑varying **mixtures** \(x(t)\in\Delta_k\).
4. Fit a **strategy‑level payoff operator** \(A\) so replicator dynamics apply.
5. Search for **ESS** and quantify significance with **IAAFT surrogates**.
(Original phrasing restored and expanded.)

---

## Pipeline (precise)

**0) Mean/contrast geometry (ANOVA/Helmert).**  
Common‑interest projector \(M_I=\tfrac{1}{N}\mathbf1\mathbf1^\top\); centering \(M_Z=I-M_I\). Optional Helmert basis (first column \(\mathbf1/\sqrt N\)).

**1) Payoff induction from time series.**
- **Static profiles:** discretize \(M_Z X\) (e.g., terciles) and map joint profiles to \(\mathbb E[X_{t+1}\mid a_t]\); keep \(M_I/M_Z\) components.
- **Information sharing (VAR = Vector Autoregression):** per player \(i\), baseline MSE (self lags) vs MSE (shared lags); **gain = baseline − observed**; center across players.

**2) Strategies and mixtures.**  
Learn \(S\in\mathbb R^{N\times k}\) (NMF/archetypes recommended). Infer \(x(t)\in\Delta_k\) by projecting \(X_{\cdot,t}\) onto cone\((S)\) with a simplex constraint.

**3) Strategy‑level signals.**  
\(g(t)=S^\top v_Z(t)\) where \(v_Z(t)=M_Z v(t)\).

**4) Replicator‑compatible operator.**  
Center in strategy space \(M_Z^{(k)}=I-\tfrac{1}{k}\mathbf1\mathbf1^\top\). Fit \(A\) via ridge so \(M_Z^{(k)} g \approx A x\); row/column centering for invariance.

**5) Equilibria and ESS.**  
Enumerate supports; solve \(A_{JJ}x_J=\alpha \mathbf1\) with \(\mathbf1^\top x_J=1\); check off‑support; test stability via the replicator Jacobian on the tangent space.

---

## Minimal API

```python
# Geometry
helmert_Q(N) -> Q
projectors(N) -> (M_I, M_Z)
weighted_projectors(weights) -> (M_Iw, M_Zw)

# Payoffs
static_game_from_series(...)
var_information_sharing_game(...)
var_information_sharing_game_seasonal(...)

# Strategy learning
nmf_on_X(X, k, iters=200, seed=None, normalize="l2") -> (S, H)

# Replicator operator
estimate_A_from_series(S, X, v, k, ridge=0.0) -> {..., "A": A}
estimate_A_from_series_weighted(..., weights=None) -> {..., "A": A}

# Equilibria
find_ESS(A, tol=1e-8, max_support=None) -> list[dict]

# Surrogates
iaaft(x, n_iter=100, rng=None, seed=None) -> x_sur
iaaft_matrix(X, n_iter=100, seed=None) -> X_sur
surrogate_ess_frequency(...) -> {"ess_rate": float, ...}
```

---

## Diagnostics & significance

- Held‑out \(R^2\) of \(M_Z^{(k)}g \approx A x\)  
- Energy split \(A_s=(A+A^\top)/2\) vs \(A_a=(A-A^\top)/2\)  
- IAAFT surrogate ESS rate vs observed  
- Rolling windows of \(A\) and ESS (nonstationarity)  
- Edge case: if \(X_t=c_t\mathbf1\), then \(v_Z\equiv 0\) → no EGT signal (by design)

---

## Applications & demos (selection)

- **Finance:** capital–labor, factor rotation & governance.  
- **Biology:** ecology/microbiology/cancer per‑capita growth, interaction fields, ESS.  
- **Sociology:** coalitions, protests, diffusion.  
- **Cybersecurity:** detector portfolio governance, deception, orchestration.  
- **SETI (toy):** spectral band allocation nulls.

See `examples/demo_var_game.py` and `notebooks/` for runnable artifacts.

---

## Notebooks & docs

- **notebooks/**: curated demos include a bootstrap cell so `import ts2eg` works before install  
- **docs/**: FINANCE.md, BIOLOGY.md, SOCIOLOGY.md, CYBERSECURITY.md, CAPITAL_LABOR.md summarize domain use

---

## Tests

```bash
pytest -q
```
All tests pass locally on Python ≥ 3.9.

---

## Reproducibility

Seed RNGs; keep OLS/ridge deterministic; save \(S\), \(A\), and window params; report \(k\), \(\lambda\), and commit hash.

---

## License

MIT.

---

## Citation

```bibtex
@misc{ts2eg_egt,
  title   = {Timeseries to EGT: From Multivariate Signals to Evolutionary Games},
  author  = {Florian Lengyel},
  year    = {2025},
  note    = {Version <tag/commit>}
}
```
