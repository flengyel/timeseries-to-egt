# gameify-timeseries: From Multivariate Time Series to Evolutionary Games

**TL;DR.** This repo turns an $N\times T$ multivariate time series into a **normal-form game**, fits an **evolutionary (replicator) model** on a small set of learned strategies, and tests for **Evolutionarily Stable Strategies (ESS)**. Evidence of an ESS is presented as evidence of **strategic interaction** in the data.

* Core file: `gameify_timeseries.py`
* Demo notebook: `egt_timeseries_demo.ipynb`

---

## Why this exists

Most “inverse EGT” work starts from observed **strategy frequencies**. Many real datasets are **player-level time series** instead. This repo provides a **coordinate pipeline** from player signals to strategies and payoffs:

1. center and rotate (Helmert/ANOVA) → common-interest vs. zero-sum directions;
2. induce payoffs from time series (two options);
3. learn a small **strategy basis**; infer time-varying **mixtures** $x(t)\in\Delta_k$;
4. fit a **strategy-level payoff operator** $A$ so replicator dynamics apply;
5. search for **ESS**.

---

## Installation

This project is intentionally light on dependencies.

```bash
python -m venv .venv
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib  # notebook demo uses pandas/matplotlib
```

> No `statsmodels` required: VAR-like regressions are solved by OLS inside the scaffold.

---

## Repository layout

```
.
├── gameify_timeseries.py        # core transforms, game builders, EGT utilities
├── egt_timeseries_demo.ipynb    # end-to-end demo (optional)
└── README.md
```

---

## Quick start

```python
import numpy as np
from gameify_timeseries import (
    static_game_from_series,           # Option A
    var_information_sharing_game,      # Option B
    estimate_A_from_series,
    find_ESS,
)

# X: N x T multivariate time series (players x time)
N, T = 4, 400
rng = np.random.default_rng(0)
X = rng.standard_normal((N, T))  # replace with your data

# 1) Build per-player payoff series v (choose one option)
# Option A: discretized state -> conditional next-step payoff
resA = static_game_from_series(X, lag=1, bins="tercile", zero_sum_discretization=True)
# Turn dict-of-profiles into a time series if you like; or use Option B:

# Option B: information-sharing game via OLS-VAR regression
resB = var_information_sharing_game(X, p=2, ridge=1e-3)
# For EGT fitting, supply a payoff series v (N x T). In quick tests, v = X is acceptable:
v = X

# 2) Provide a strategy basis S (N x k). Use NMF/archetypes in practice.
# As a placeholder, rectified PCA:
U, svals, Vt = np.linalg.svd(X - X.mean(axis=1, keepdims=True), full_matrices=False)
k = 3
S = np.abs(U[:, :k])   # for real analyses, prefer NMF/archetypes

# 3) Estimate A (k x k) and search for ESS
est = estimate_A_from_series(S, X, v, k=k, lambda_=1e-3)
A = est["A"]
cands = find_ESS(A, tol=1e-8, max_support=k)
ess = [r for r in cands if r["is_ess"]]
print("ESS supports:", [r["support"] for r in ess])
```

Or open the demo:

* `egt_timeseries_demo.ipynb` (sets up NMF for $S$, fits $A$, and runs `find_ESS`).

---

## Pipeline overview (precise)

### 0) Mean/contrast geometry (Helmert/ANOVA)

* Common-interest projector: $M_I=\frac{1}{N}\mathbf 1\mathbf 1^\top$.
* Zero-sum projector: $M_Z=I-M_I$ (centering).
* Optional orthogonal basis $Q$ (Helmert): first column $\mathbf 1/\sqrt N$, others span $\mathbf 1^\perp$.

**Why:** Replicator dynamics are invariant to adding a constant payoff to all strategies; i.e., they depend only on **centered** fitness. $M_Z$ enforces that invariance at the player level.

### 1) Induce per-player payoffs $v(t)$ from time series

Two interchangeable choices:

**A. Static, discretized profiles → conditional next-step mean**

* Discretize per-time **relative positions** (apply $M_Z$ first, then terciles).
* Each joint profile $a_t$ maps to a payoff vector $\hat u(a)=\mathbb E[X_{t+1}\mid a_t=a]$.
* Store components $M_I\hat u(a)$ and $M_Z\hat u(a)$.

**B. Information-sharing game (OLS-VAR)**

* Players choose whether to **share** their lags.
* For player $i$, compare MSE using only self lags vs. MSE with shared lags; **gain** = baseline − observed.
* Center across players: $u_Z=M_Z u$, optionally add team component $\lambda M_I u$.

### 2) Learn strategies and infer mixtures $x(t)$

* Choose $k$. Learn a **strategy basis** $S=[s_1,\dots,s_k]\in\mathbb R^{N\times k}$ (prefer **NMF** or archetypal analysis).
* Infer memberships $x(t)\in\Delta_k$ by **projecting** $X_{\cdot,t}$ onto cone$(S)$ with a simplex constraint:

  $$
  x(t)=\arg\min_{x\in\Delta_k}\|Sx - X_{\cdot,t}\|_2^2
  $$

  (implemented via projected-gradient; see `estimate_A_from_series`).

### 3) Map player payoffs to **strategy-level signals**

* Centered player payoffs: $v_Z(t)=M_Z v(t)\in\mathbb R^N$.
* Strategy signal: $g(t)=S^\top v_Z(t)\in\mathbb R^k$ (columns of $S$ are unit-norm by default).
  Intuition: $g_i$ is the alignment of the payoff field with strategy $i$’s player pattern.

### 4) Fit a replicator-compatible payoff operator

* Model: symmetric population game with $f(x)=A x$, $A\in\mathbb R^{k\times k}$.
* Replicator only “sees” centered fitness: $M_Z^{(k)}=I-\frac{1}{k}\mathbf 1\mathbf 1^\top$.
* Estimate $A$ by ridge:

  $$
  \min_A\sum_t\|\,M_Z^{(k)}g(t)-A x(t)\,\|_2^2+\lambda\|A\|_F^2,
  $$

  then **row-center** $A\leftarrow M_Z^{(k)} A$ (replicator invariance).

### 5) Equilibria and ESS

* **Nash candidates:** For each support $J$, solve $A_{JJ}x_J=\alpha \mathbf 1,\; \mathbf 1^\top x_J=1$; check off-support inequalities.
* **ESS test:** (i) off-support strategies strictly worse; (ii) Jacobian of replicator on the support’s **tangent** has eigenvalues with negative real part.
* Implemented in `find_ESS(A, tol, max_support)` with explicit Jacobian.

---

## API (concise)

```python
helmert_Q(N) -> Q
projectors(N) -> (M_I, M_Z)

static_game_from_series(X, lag=1, bins="tercile", zero_sum_discretization=True)
  -> StaticGameResult  # dicts of profile -> payoff, plus I/Z components

var_information_sharing_game(X, p=1, ridge=0.0, restrict_profiles=None,
                             include_self_always=True, lambda_common=0.0)
  -> VARSharingGameResult  # profile -> payoff gains, centered + components

estimate_A_from_series(S, X, v, k, lambda_=0.0)
  -> dict(A, Xk, Gk, Gc, R2, Cxx, Cgx)

find_ESS(A, tol=1e-8, max_support=None)
  -> list of {support, x, alpha, is_nash, is_ess, eigvals_tangent}
```

---

## Diagnostics you SHOULD report

* **Fit quality:** $R^2$ of $M_Z^{(k)}g \approx A x$ on held-out time.
* **Decomposition:** energy in $A_s=(A+A^\top)/2$ vs. $A_a=(A-A^\top)/2$ to gauge potential vs. cyclic structure.
* **Nulls:** time-shuffle within players (or circularly shift rows), re-fit. A large drop in fit/ESS frequency supports non-accidental structure.
* **Rolling windows:** stability of $A$ and any ESS over time (nonstationarity matters).
* **Edge cases:** if $X_t=c_t\mathbf 1$, then $v_Z\equiv 0$ and no EGT signal should be found.

---

## Design choices & caveats

* **Strategies vs. players.** The replicator state is $x(t)\in\Delta_k$ over **strategies**, not players. This pipeline enforces that separation.
* **PCA caution.** PCs are signed and live in $\mathbb R^N$, not on the simplex. Prefer **NMF**/**archetypes** (nonnegativity, parts-based). If you must use PCA, split positive/negative parts and renormalize.
* **Scaling.** Normalize columns of $S$; project $x(t)$ onto the **simplex** (not just softmax).
* **Computation.** `find_ESS` enumerates supports; for $k\gtrsim 15$, set `max_support` or use heuristic support search.
* **Causality.** Option B encodes predictive gains; it is not causal identification. Treat policy claims cautiously.

---

## Relation to existing work (tight)

* **Orthogonal game decompositions.** Potential/identical-interest, zero-sum/harmonic, and nonstrategic components with projection formulas; this repo uses **the same geometry** but stated in **ANOVA/Helmert** coordinates.

  * Candogan, Menache, Ozdaglar, Parrilo (2011). *Math. of OR*.
  * Monderer & Shapley (1996). *Games and Economic Behavior*.
  * Hwang & Rey-Bellet (2011, 2020). *arXiv*; *GEB*.

* **Empirical game-theoretic analysis (EGTA).** Induces games from simulations/observations; this repo’s Option A/B is related but anchored in centering/Helmert transformations.

* **Inverse EGT / learning utilities.** Prior work estimates payoffs from observed strategy shares; here we **construct** those shares from player-level time series via a strategy basis and simplex projection.

* **VAR/Granger networks.** Option B’s information-sharing gains reuse standard forecast-error reductions as a payoff surrogate.

**Novelty claim (what’s different):** the **explicit ANOVA/Helmert mean/contrast transform** applied **columnwise** to construct player payoffs, followed by projection to **strategy-level signals** and a ridge fit of $A$ for replicator, appears to be under-documented as a single, ready-to-run pipeline. It’s a methods contribution; test it on synthetic + real datasets.

---

## Reproducibility

* Seeded RNG in demos.
* Deterministic OLS/ridge fits.
* Provide data and basis $S$ to reproduce figures.
* Report exact code/commit, $k$, ridge $\lambda$, and windowing.

---

## Roadmap

* Option C: **spectral** payoffs (band powers via $Q^\top S(\omega) Q$).
* Dynamic fit: directly regress $\Delta x(t)$ on the replicator field.
* Symmetry constraints: enforce $A=A^\top$ (potential) or $A+A^\top=0$ (zero-sum) during estimation.
* Heuristic support search / MILP for ESS when $k$ is large.
* Proper NMF/archetypal model selection and uncertainty (bootstrap).

---

## Citing

If this repo is used in a publication, please cite:

```
@misc{gameify_timeseries_egt,
  title = {Gameify Time Series: From Multivariate Signals to Evolutionary Games},
  author = {Florian Lenhue;},
  howpublished = {\url{https://github.com/flengyel/gameify-timeseries}},
  year = {2025},
  note = {Version <tag/commit>}
}
```

Related foundational references (minimal):

* D. Monderer and L. S. Shapley (1996), *Potential Games*, **GEB**.
* O. Candogan, I. Menache, A. Ozdaglar, P. A. Parrilo (2011), *Flows and Decompositions of Games*, **Math. OR**.
* S.-H. Hwang and L. Rey-Bellet (2011, 2020), *Decompositions/Strategic Decompositions of Games*.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgments

Thanks to the ANOVA/Helmert tradition in multivariate statistics and the orthogonal decompositions literature in game theory; the pipeline here is a bridge between them.
