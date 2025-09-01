# BIOLOGY: Ecological & Biomedical Applications of Time-Series → EGT

This document summarizes **how to use the pipeline in biology**, with payoff definitions that are standard to ecologists, microbiologists, and cancer biologists.

---

## Players, signals, strategies

- **Players** = measured entities that compete for resources or space: microbial taxa/guilds, phytoplankton groups, immune cell modules, tumor subclones, or pathway modules.\n
- **Signal matrix** `X ∈ R^{N×T}` = time series per player. For microbes/subclones, `X` is typically **relative abundances** (compositional). For counts, you MAY use raw counts for payoff construction and normalized shares for strategy inference.\n
- **Strategies** `S = [s₁…s_k]` learned by **NMF/archetypal analysis** on `X` (nonnegative, column-normalized). Each column is an **archetypal guild/state**. The pipeline infers the mixture `x(t) ∈ Δ_k` (simplex) over time.

> Why strategies? You typically have many taxa/genes but a small number of **ecological roles**; strategies compress the system to interpretable components.

---

## Payoff definitions (domain-standard)

### 1) Per-capita growth rate (Malthusian fitness) — *most interpretable*
Let `n_i(t)` be counts (or biomass) for player `i`. The **per-capita growth rate** is
\[
v_i(t) \;=\; \frac{\log(n_i(t+1)) - \log(n_i(t))}{Δt}.
\]
This is the discrete-time Malthusian fitness used across ecology, microbiology, and cancer dynamics.

- In code: `growth_payoffs(counts, dt, pad)` returns `v ∈ R^{N×T}`.\n
- If only relative abundances `p_i(t)` are available, treat `n_i(t)=p_i(t)·N_total(t)` with an offset; or work with **log-ratio** growth using a reference taxon.\n
- For single-cell lineage or tumor subclones, use **clone counts** (ctDNA-inferred or deconvolved) with purity/coverage adjustments.

**Interpretation:** `v_i(t)>0` means taxon/subclone `i` is increasing per capita at time `t`. The pipeline centers payoffs across players (replicator invariance) to focus on trade-offs.

### 2) Information-gain payoffs for a scalar endpoint
If a scalar endpoint exists (e.g., total biomass at horizon `H`, metabolite concentration, symptom score), define payoffs as **contribution to forecasting** that endpoint:

- Fit a simple predictor of `J_{t+H}` from `X_t` with ridge and compute **Leave-One-Feature-Out** (LOFO) **MSE improvements**.\n
- In code: `info_gain_payoffs(X, J_future, ridge, window)` returns `v ∈ R^{N×T}` with `v_i(t) = MSE_{−i} - MSE_full` (nonnegative means feature `i` helps).

**Interpretation:** Channels that consistently improve target prediction carry positive payoffs; the centered component highlights relative contributions.

---

## Model mapping and intuition

- **Centering across players** (`M_Z`): replicator dynamics only depend on **relative** fitness. Subtracting the player-average payoff isolates competition/coordination signals.\n
- **Replicator vs LV:** When per-capita growth is approximately linear in the strategy mixture (`f(x) = A x`), replicator dynamics on `x` provide a compressed **interaction field** akin to LV but in **strategy space**.\n
- **A decomposition:** Split `A` into symmetric `A_s` (potential-like, aligning to a scalar objective) and skew `A_a` (cyclic trade-offs). Large `||A_a||` flags rock–paper–scissors dynamics, predator–prey–like loops, or resource switching cycles.

---

## Practical workflow

1. **Preprocess**\n
   - Denoise (winsorize outliers, filter rare taxa).\n
   - If compositional: use relative abundances for `X` (strategy inference) and counts/log-counts for growth payoffs.\n
   - Handle seasonality (e.g., 7-day/annual) with fixed effects or differences.\n

2. **Strategies**\n
   - Learn `S` with NMF (`k = 3–6` typical); interpret columns as **guilds** (e.g., oligotroph vs copiotroph) or **clonal states**.\n

3. **Payoffs**\n
   - Prefer `growth_payoffs` when counts exist; otherwise `info_gain_payoffs` to a meaningful endpoint `J`.\n

4. **Fit & analyze**\n
   - `est = estimate_A_from_series(S, X, v, k, ridge)`.\n
   - Examine `R²` (held-out if possible), `||A_s||` vs `||A_a||`, and **ESS** via `find_ESS(A)`.\n

5. **Validate**\n
   - **Nulls:** circular shifts by row; AAFT/phase-scramble; label permutations.\n
   - **Perturbations:** diet switches, blooms, therapy on/off windows; check that the vector field rotates as expected.\n
   - **Rolling windows:** regime shifts (seasonal turnover, treatment phases).\n

---

## Research directions (examples)

- **Microbiome (human/environmental):** Do guild mixes exhibit ESS across seasons or under diet/antibiotic shocks? Are cyclic trade-offs stronger during blooms?\n
- **Phytoplankton/zooplankton series:** Seasonal ESS vs turnover; coupling to nutrient/temperature covariates.\n
- **Immunodynamics:** Module-level reallocation after vaccination/challenge; ESS differences between responders/non-responders.\n
- **Cancer subclones:** Therapy-induced shifts in `A`; ESS loss at progression; adaptive therapy as guided movement along the replicator vector field.\n

---

## Reporting checklist

- Data source, preprocessing, and whether counts or relative abundances were used.\n
- Strategy basis `S` (heatmap with labels), number of strategies `k`.\n
- Payoff construction (`growth` vs `info_gain`) and horizon.\n
- Fit quality (`R²`), symmetry ratio `||A_a||/||A||`, ESS supports and basin evidence.\n
- Null controls and robustness (rolling windows, seed sensitivity).\n

---

## Minimal code snippet

```python
from gameify_timeseries import growth_payoffs, estimate_A_from_series, find_ESS
# counts: N x T
v = growth_payoffs(counts, dt=1.0)
X = counts / (counts.sum(axis=0, keepdims=True) + 1e-12)   # shares for strategy inference
S = nmf_on_X(X, k=4)  # any NMF; normalize columns
est = estimate_A_from_series(S, X, v, k=4, ridge=1e-2)
A = est['A']
ess = [r for r in find_ESS(A) if r['is_ess']]
```

See `BIO_DEMO.ipynb` for a runnable synthetic example you can adapt to real longitudinal datasets (microbiome, plankton, immune challenges, or tumor subclones).
