# SOCIOLOGY: Applying Time‑Series → EGT to Mathematical Sociology

This note explains how to apply the `gamify-timeseries` pipeline to **coalitions, collective action, online governance, diffusion, and organizational role allocation**. It emphasizes payoff definitions that are recognizable to sociologists and political scientists.

---

## Core mapping

- **Players**: groups/tactics/roles that *compete or coordinate* over finite resources (attention, members, seats, budget). Examples: political parties, movement tactics (march/strike/blockade), moderator levers (warn/revert/ban), adopter cohorts, internal org roles.
- **Signals (`X ∈ ℝ^{N×T}`)**: per-time *shares* or *normalized intensities* of each player (columns sum to 1 is often a good convention).
- **Strategies (`S ∈ ℝ^{N×k}`)**: **archetypes** discovered from the data by NMF/archetypal analysis (nonnegative, column‑normalized). At each time \(t\), the system is a mixture \(x(t)\in\Delta_k\).
- **Payoffs (`v ∈ ℝ^{N×T}`)**: domain‑legible measures of *advantage* at time \(t\) (see below). Replicator analysis uses the centered component \(M_Z v\) so it focuses on *relative* advantage.

---

## Payoff definitions (domain‑legible)

1. **Per‑capita growth (Malthusian fitness)** — Canonical for parties, membership groups, tactics counted over time.
   \[
   v_i(t) \;=\; \frac{\log n_i(t{+}1) - \log n_i(t)}{\Delta t}.
   \]
   Use when you have counts (members, votes, events). In code: `growth_payoffs(counts, dt)`.

2. **Information‑gain toward a social outcome** — When an external outcome exists (future vote share, mobilization, bill passage, retention rate).
   - Define a prediction horizon \(H\) and target \(J_{t+H}\).
   - Compute **leave‑one‑feature‑out** (LOFO) improvement in prediction of \(J\) from \(X_t\).
   - In code: `info_gain_payoffs(X, J_future, ridge, window)` returns \(v_i(t)=\mathrm{MSE}_{-i}-\mathrm{MSE}_{\text{full}}\).

3. **Cost‑aware variants (optional)** — Subtract a cost term (e.g., repression risk, moderation load) from each \(v_i\) before centering.

> Use **one** payoff definition per analysis; mixing them blurs interpretation.

---

## Workflow

1. **Build `X`** (shares or normalized intensities). Handle seasonality with fixed effects or differencing if needed.  
2. **Choose `v`**: `growth_payoffs` for counts or `info_gain_payoffs` for outcome‑driven analysis.  
3. **Learn `S`** by NMF (3–6 archetypes typical). Normalize columns.  
4. **Estimate** `A` with `estimate_A_from_series(S, X, v, k, lambda_)`.  
5. **Analyze**: fit quality \(R^2\); symmetric vs. skew parts of \(A\); **ESS** with `find_ESS(A)`; rolling‑window stability.  
6. **Validate**:  
   - **Nulls**: circular row shifts; block bootstrap; permute target \(J\).  
   - **Interventions**: elections, policy changes, major protests; check whether the vector field rotates as expected.

---

## Research directions

- **Party competition / coalitions**: stable coalition mixes (ESS) vs rotational cycles; compare pre/post electoral reform.  
- **Protest–policing cycles**: tactic portfolios and repression responses; identify cyclic interaction structure (\(\|A_a\|\)).  
- **Online governance (Wikipedia/communities)**: find stable moderation portfolios that maximize retention/quality.  
- **Diffusion of norms**: adopter cohort mixes; regimes with ESS vs boom‑bust cycles.  
- **Organizations**: allocation of staff/effort across roles; robust role blends under leadership changes.

---

## Interpretation aids

- \(A = A_s + A_a\):  
  - \(A_s\) (symmetric) acts like a potential — aligns with a scalar objective.  
  - \(A_a\) (skew) encodes **cycles** — arms‑race or rotation among archetypes.  
- **ESS** = evolutionarily stable strategy (stable archetype mix). Its presence/absence and basin size are substantive findings.

---

## Ethics & cautions

- Use aggregated, de‑identified data when possible. Avoid sensitive attributes as “players” unless approved and necessary.  
- Beware endogeneity: outcomes and memberships co‑determine each other; prefer lagged targets, natural experiments, or instrumented designs.  
- Report negative results (no ESS; weak fit) — they are informative about volatility/nonstationarity.

---

## Reporting checklist

- Data provenance & preprocessing; whether counts or shares used.  
- Payoff definition and horizon.  
- Strategy basis heatmap; \(k\).  
- Fit (\(R^2\)), \(\|A_a\|/\|A\|\), ESS supports & stability (rolling).  
- Null results and intervention checks.

---

## Minimal code

```python
from gameify_timeseries import growth_payoffs, info_gain_payoffs
from gameify_timeseries import estimate_A_from_series, find_ESS

# counts: N x T (e.g., parties or tactics)
v = growth_payoffs(counts, dt=1.0)        # or: v = info_gain_payoffs(X, J, ridge=1e-2, window=52)

# X: shares for strategy inference
X = counts / (counts.sum(axis=0, keepdims=True) + 1e-12)

# Learn S (NMF with column normalization), then:
est = estimate_A_from_series(S, X, v, k=S.shape[1], lambda_=1e-2)
A = est["A"]
ess = [r for r in find_ESS(A) if r["is_ess"]]
```
