# timeseries-to-egt (ts2eg): From Multivariate Time Series to Evolutionary Games

[![CI](https://github.com/flengyel/timeseries-to-egt/actions/workflows/ci.yml/badge.svg)](https://github.com/flengyel/timeseries-to-egt/actions/workflows/ci.yml)

<!-- ts2eg-intro-exact-start -->
## Introduction

`ts2eg` is an **inverse evolutionary game theory** pipeline that tests a multivariate time series $X\in\mathbb{R}^{N\times T}$ for **competitive (zero-sum) strategic interaction**.

* The test asks: after removing individual dynamics and **common-interest signal**, **is there evidence that agents interact through relative payoffs** consistent with replicator dynamics?
* The pipeline **projects away the common-interest component** ($M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top$) and **retains only the zero-sum signal** ($M_Z=I_N-M_I$). A purely cooperative process (signal in the $M_I$ direction) is filtered out and yields a null result.

**Hypotheses and test statistic.**

* $H_0$: any apparent equilibrium arises from per-series autocorrelation only (no cross-player coupling).
* $H_1$: there exists a centered payoff operator $A$ in strategy space that produces an ESS under replicator dynamics.

Test statistic: the **ESS indicator** on the observed data (1 if an ESS is found, else 0). The null ESS rate is estimated via IAAFT surrogates; report the empirical $p$-value
$$
p=\frac{1+\#\{\text{surrogates with ESS}\}}{1+S},
$$
where $S$ is the number of surrogates. (Low $p$ with an observed ESS rejects $H_0$.)

See mathematical background: `docs/ts2eg_math_background.tex`.

## Pipeline (end-to-end)

| Stage                   | Input           | Core method                     | Output                                                   |
| ----------------------- | --------------- | ------------------------------- | -------------------------------------------------------- |
| 1. Signal isolation     | $X$             | Project with $M_Z$              | Zero-sum payoff signal $v_Z(t)$                          |
| 2. Payoff induction     | $X$             | VAR information-sharing (ridge) | Per-time payoffs $v(t)$                                  |
| 3. Strategy learning    | $X$             | Dictionary learning (NMF)       | $S\in\mathbb{R}^{N\times k}$, mixtures $x(t)\in\Delta_k$ |
| 4. Game estimation      | $v_Z(t),\,x(t)$ | Centered ridge regression       | $A\in\mathbb{R}^{k\times k}$                          |
| 5. Equilibrium analysis | $A$             | Replicator Jacobian test        | ESS presence/absence                                     |
| 6. Significance         | $X$             | IAAFT surrogates                | Empirical $p$-value                                      |


**Stage 1 (definition).** Define the mean and centering projectors in player space

$$
M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top,\qquad M_Z=I_N-M_I.
$$

For any player-level vector $y\in\mathbb{R}^N$, decompose $y=y_I+y_Z$ with $y_Z=M_Z y$ and **retain only $y_Z$**. In practice we apply this to payoffs: once Stage 2 produces $v(t)$, set

$$
v_Z(t)=M_Z\,v(t),
$$

and carry $v_Z(t)$ forward.

**Weighted variant:** for $w\ge 0$ with $\sum_i w_i>0$, let $\pi=w/\sum_i w_i$ and use $M_I^{(w)}=\mathbf{1}\pi^\top,\; M_Z^{(w)}=I_N-\mathbf{1}\pi^\top$.

**Note:** if $X_{\cdot,t}=c_t\,\mathbf{1}$ for all $t$, then $v_Z\equiv 0$ and the test (correctly) returns a null result.

**Stage 2 (definition).** For each player $i$, fit (i) a self-only AR($p$) model and (ii) a full VAR($p$) using all players (ridge-regularized). The information-sharing payoff is the **MSE reduction**  
$\;\;v_i=\mathrm{MSE}_{\text{self}}-\mathrm{MSE}_{\text{full}}.$  
Form $v(t)=(v_1,\dots,v_N)^\top$ and carry forward $v_Z(t)=M_Zv(t)$.

## Mathematical setup

* **Projectors.** $M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top,\; M_Z=I_N-M_I.$ Helmert $Q$ gives an orthonormal basis with $q_1\propto\mathbf{1}$. 
* **Weighted projectors.** For weights $w\in\mathbb{R}_{\ge0}^N$ with $\sum_i w_i>0$, define $\pi=w/\sum_i w_i$. Then
  $$
  M_I^{(w)}=\mathbf{1}\,\pi^\top,\qquad M_Z^{(w)}=I_N-\mathbf{1}\,\pi^\top.
  $$
* **Strategies.** Factor $X\approx S H$ (NMF) with $S\in\mathbb{R}^{N\times k}_{\ge0}$, $H\in\mathbb{R}^{k\times T}_{\ge0}$; define $x(t)$ as the $t$-th column of $H$ re-normalized to $\Delta_k$. Stack $X_k=[x(1)\,\cdots\,x(T)]\in\mathbb{R}^{k\times T}$.
* **Signals.** Let $g(t)=S^\top v_Z(t)$ and stack $G=[g(1)\,\cdots\,g(T)]$. Center **only** the signals as $G_c=M_Z^{(k)}G$ with $M_Z^{(k)}=I_k-\tfrac{1}{k}\mathbf{1}\mathbf{1}^\top$. (Mixtures $x(t)$ are not centered; they live on $\Delta_k$.)
* **Estimator.**
  $$
  C_{xx}=X_k X_k^\top,\quad C_{gx}=G_c X_k^\top,\quad
  A = C_{gx}\,(C_{xx}+\rho I_k)^{-1},\; \rho=\texttt{ridge},\quad
  A\leftarrow M_Z^{(k)}\,A\,M_Z^{(k)}.
  $$
  The estimator uses `np.linalg.solve` with `pinv` fallback and **enforces $A\mathbf{1}=\mathbf{0}$ and $\mathbf{1}^\top A=\mathbf{0}$** via the centering step.
* **Dynamics/ESS.** Replicator $\dot{x}_i=x_i\big((Ax)_i-x^\top A x\big)$. The function `find_ESS` enumerates supports, checks Nash feasibility, and tests **local stability via the Jacobian on the tangent space**.

## Interpreting the results

* **Positive (reject $H_0$):** an ESS is found on the data and the empirical $p$-value from surrogates is small $\Rightarrow$ evidence for **competitive, zero-sum interaction** in $X$ beyond per-series autocorrelation.
* **Null:** either no strategic coupling, or only cooperative/common-interest signal (deliberately excluded by $M_Z$), or competitive signal too weak to detect.

## Numerical/robustness notes

* Invariance to level shifts via $M_Z$.
* Centered operator $A$ ($A\mathbf{1}=0$ and $\mathbf{1}^\top A=0$) by construction.
* Ridge regularization with pseudo-inverse fallback for near-singular $C_{xx}$.
* Weighted variants guard against $\sum w=0$.
* Reproducible examples (`--seed`) seed NumPy (and Python RNG where used); NumPy-2-compatible.

<!-- ts2eg-intro-exact-end -->

**Package:** `ts2eg` (src‑layout). Code: `src/ts2eg/core.py` and `src/ts2eg/extensions.py`.

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

Python $\ge$ 3.9 recommended. CI runs `pytest` on 3.9–3.12.

---

## Repository layout

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

## Quick start

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
3. Learn a compact **strategy basis** and infer time‑varying **mixtures** $x(t)\in\Delta_k$.
4. Fit a **strategy‑level payoff operator** $A$ so replicator dynamics apply.
5. Search for **ESS** and quantify significance with **IAAFT surrogates**.
(Original phrasing restored and expanded.)

---

## Pipeline

**0) Mean/contrast geometry (ANOVA/Helmert).**  
Common‑interest projector $M_I=\tfrac{1}{N}\mathbf1\mathbf1^\top$; centering $M_Z=I-M_I$. Optional Helmert basis (first column $\mathbf1/\sqrt N$).

**1) Payoff induction from time series.**

* **Static profiles:** discretize $M_Z X$ (e.g., terciles) and map joint profiles to $\mathbb E[X_{t+1}\mid a_t]$; keep $M_I/M_Z$ components.
* **Information sharing (VAR = Vector Autoregression):** per player $i$, baseline MSE (self lags) vs MSE (shared lags); **gain = baseline − observed**; center across players.

**2) Strategies and mixtures.**  
Learn $S\in\mathbb R^{N\times k}$ (NMF/archetypes recommended). Infer $x(t)\in\Delta_k$ by projecting $X_{\cdot,t}$ onto cone$(S)$ with a simplex constraint.

**3) Strategy‑level signals.**  
$g(t)=S^\top v_Z(t)$ where $v_Z(t)=M_Z v(t)$.

**4) Replicator‑compatible operator.**  
Center in strategy space $M_Z^{(k)}=I-\tfrac{1}{k}\mathbf1\mathbf1^\top$. Fit $A$ via ridge so $M_Z^{(k)} g \approx A x$; row/column centering for invariance.

**5) Equilibria and ESS.**  
Enumerate supports; solve $A_{JJ}x_J=\alpha \mathbf1$ with $\mathbf1^\top x_J=1$; check off‑support; test stability via the replicator Jacobian on the tangent space.

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

* Held‑out $R^2$ of $M_Z^{(k)}g \approx A x$  
* Energy split $A_s=(A+A^\top)/2$ vs $A_a=(A-A^\top)/2$  
* IAAFT surrogate ESS rate vs observed  
* Rolling windows of $A$ and ESS (nonstationarity)  
* Edge case: if $X_t=c_t\mathbf1$, then $v_Z\equiv 0$ → no EGT signal (by design)

---

## Applications & demos

* **Finance:** capital–labor, factor rotation & governance.  
* **Biology:** ecology/microbiology/cancer per‑capita growth, interaction fields, ESS.  
* **Sociology:** coalitions, protests, diffusion.  
* **Cybersecurity:** detector portfolio governance, deception, orchestration.  
* **SETI (toy):** spectral band allocation nulls.

See `examples/demo_var_game.py` and `notebooks/` for runnable artifacts.

---

## Notebooks & docs

* **notebooks/**: curated demos include a bootstrap cell so `import ts2eg` works before install  
* **docs/**: FINANCE.md, BIOLOGY.md, SOCIOLOGY.md, CYBERSECURITY.md, CAPITAL_LABOR.md summarize domain use

---

## Tests

```bash
pytest -q
```

All tests pass locally on Python ≥ 3.9.

---

## Reproducibility

Seed RNGs; keep OLS/ridge deterministic; save $S$, $A$, and window params; report $k$, $\lambda$, and commit hash.

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

## Math glossary

**Vectors & matrices**  

* $\mathbf{1}_n \in \mathbb{R}^n$: all-ones column vector  
* $I_n$: $n\times n$ identity  
* $X \in \mathbb{R}^{N\times T}$: time series (rows = series, cols = time)  
* $S \in \mathbb{R}^{N\times k}$: nonnegative strategy loadings (from NMF)  
* $X_k \in \mathbb{R}^{k\times T}$: reduced membership/latent time series  
* $A \in \mathbb{R}^{k\times k}$: normal-form payoff operator in strategy space  

**Projectors (unweighted)**  

* Mean projector onto the constant subspace:  
  $$M_I^{(n)}=\frac{1}{n}\,\mathbf{1}_n \mathbf{1}_n^\top \quad\text{(symmetric, idempotent, rank 1)}$$
* Centering projector onto the zero-mean subspace:  
  $$M_Z^{(n)}=I_n-M_I^{(n)}=I_n-\frac{1}{n}\,\mathbf{1}_n \mathbf{1}_n^\top \quad\text{(symmetric, idempotent)}$$
* Identities: $M_I^{(n)}\mathbf{1}_n=\mathbf{1}_n$, $M_Z^{(n)}\mathbf{1}_n=\mathbf{0}$, $M_I^{(n)}M_Z^{(n)}=0$, $M_I^{(n)}+M_Z^{(n)}=I_n$.

**Projectors (weighted)**  
For weights $w\in\mathbb{R}_{\ge 0}^n$ with $\sum_i w_i>0$, define $\pi = w/\sum_i w_i$. Then  
$$M_I^{(n)}(w)=\mathbf{1}_n \pi^\top,\qquad M_Z^{(n)}(w)=I_n-\mathbf{1}_n \pi^\top.$$
(Orthogonal under the $w$-weighted inner product.)

**Helmert basis $Q$**  

* $Q\in\mathbb{R}^{N\times N}$ orthonormal with $Q^\top Q=I_N$.  
* First column $q_1=\mathbf{1}_N/\sqrt{N}$ spans the mean subspace; columns $2..N$ span the centered subspace.  
* Code: `Q = ts2eg.helmert_Q(N)`; invariants: `Q.T @ Q == I`, `Q[:,0] == 1/sqrt(N)`.

**Centering a payoff operator**  
When payoffs are defined up to additive constants in strategies, enforce  
$$A \leftarrow M_Z^{(k)}\,A\,M_Z^{(k)}$$
to remove row/column offsets in strategy space. (Code guarantees `A = MZ_k @ A @ MZ_k`.)

**Ridge-regularized fit for $A$**  
Given $C_{xx}=X_k X_k^\top$ and $C_{gx}=G_c X_k^\top$ (centered signals $G_c$),  
$$A = C_{gx}\,(C_{xx}+\rho I_k)^{-1},\qquad \rho=\texttt{ridge}\ge 0,$$
implemented with `np.linalg.solve`, fallback to `np.linalg.pinv` on singularities.

**ESS / Nash (sketch)**  
`ts2eg.find_ESS(A)` enumerates mixed-support candidates, checks Nash and evolutionary stability (replicator Jacobian on supports); returns `is_nash`, `is_ess`.

**API touchpoints**  

* `helmert_Q(N)`, `projectors(N)` → $Q$, $M_I^{(N)}$, $M_Z^{(N)}$  
* `estimate_A_from_series(S, X, v, k, ridge=...)` → $A$ (+ diagnostics)  
* `extensions.estimate_A_from_series_weighted(..., ridge=..., weights=...)` uses $M_I^{(n)}(w)$, $M_Z^{(n)}(w)$ for weighted centering  
* `find_ESS(A)` → ESS/Nash summary  

**Further derivations**  
See **docs/ts2eg_math_background.tex** for proofs linking $Q$, projectors, and centering of $A$.
