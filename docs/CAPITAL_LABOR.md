# CAPITAL–LABOR: Mapping the Theory of the Firm to Evolutionary Games

This note shows how to analyze **capital–labor dynamics** with the `ts2eg` pipeline in a value‑neutral way. The method separates **common‑interest** gains from **zero‑sum** redistribution and offers falsifiable tests for **systematic surplus capture** by either side.

---

## Mapping

- **Players** (rows): budget/effort competitors whose shares sum to 1 each period, e.g., `LaborComp`, `CapEx`, `R&D`, `SG&A`, `Payouts`.\n
- **Signals** `X ∈ ℝ^{N×T}`: per‑period **allocation shares** (normalize each column to 1). If you include accounting identities, remove them with `project_out_identities`.\n
- **Strategies** `S`: **allocation archetypes** learned via nonnegative factorization (`nmf_on_X`). Each time $t$, the firm is a mixture $x(t)\in\Delta_k$ of these archetypes.\n
- **Payoffs** `v ∈ ℝ^{N×T}`: domain‑legible measures of advantage. The replicator uses the **centered** component $M_Z v$, focusing on **relative** payoffs.\n

---

## Payoff definitions

1. **Distributional (surplus‑capture) payoff** — direct capital vs labor test.\n
   - Let `VA` be value added, `W` labor compensation, `P` operating surplus.\n",
   "   - Define $v_{\text{Labor}}(t) = \Delta\log(W/VA)$ and $v_{\text{Capital}}(t) = \Delta\log(P/VA)$.\n",
   "   - Assign these to `LaborComp` and `Payouts` rows (others = 0), then center across players.\n",
   "   - Interpretation: positive centered value for capital concurrent with negative for labor indicates **redistribution toward capital**, given this payoff.\n",
   "\n",
2. **Value‑gradient payoff** — managerial lens.\n",
   "   - Choose horizon $H$ and target $J_{t+H}$ (e.g., **ROIC**, **FCF yield**).\n",
   "   - Use `value_gradient_payoffs(X, J_future)` to estimate per‑channel gradients; center them.\n",
   "   - Interpretation: if centered gradients reward payout shares and penalize labor shares, the **shareholder‑value optimum** tilts against labor under the chosen model.\n",
   "\n",
3. **Information‑gain payoff** — observational.\n",
   "   - `info_gain_payoffs(X, J_future, window)` gives leave‑one‑feature‑out forecast improvements; centered values show **relative** leverage.\n",
   "\n",
Use one payoff family per analysis to keep interpretation clean; report sensitivity across families.\n

---

## Falsifiable test for “systematic surplus capture”

Define
$$
D \;=\; \overline{v^{Z}_{\text{Capital}}} - \overline{v^{Z}_{\text{Labor}}}\,,
$$
the difference of **time‑averaged centered payoffs** (capital minus labor).

1. Compute `v` and $v^Z=M_Z v$.\n
2. Estimate $D$.\n
3. Generate a **row‑shift null** by circularly shifting each row of `v` independently (preserves autocorrelation; breaks cross‑row coordination). Recompute $D$ over `B` surrogates.\n
4. Report a z‑score or empirical p‑value.\n

**Interpretation:** Large positive $D$ with small null probability is **evidence** of systematic surplus capture by capital over labor (conditional on the payoff definition). Negative $D$ indicates the converse. If $D$ ≈ null, the data do not support systematic capture in the window.\n

---

## Dynamic analysis

- Learn strategies `S` with `nmf_on_X`; fit `A` with `estimate_A_from_series(S, X, v, k, λ)`.\n
- Decompose $A=A_s+A_a$: symmetric $A_s$ (potential‑like) vs skew $A_a$ (cycles). Large $\|A_a\|$ flags **bargaining/rotation cycles** (buybacks → wage catch‑up → automation → …).\n
- Search for **ESS** with `find_ESS(A)`. ESS persistence across rolling windows indicates a **stable allocation regime**; loss of ESS suggests strategic instability/regime change.\n
- Always include **nulls** (row shifts; phase‑scramble if applicable).\n

---

## Minimal recipe

```python
from ts2eg import nmf_on_X, value_gradient_payoffs, estimate_A_from_series, find_ESS

# X: N x T allocation shares (columns sum to 1)
v = value_gradient_payoffs(X, J_future=roic_lead, ridge=1e-2)  # or distributional Δlog shares

S, H = nmf_on_X(X, k=3, iters=300, seed=1)
est = estimate_A_from_series(S, X, v, k=3, ridge=1e-2)
A, R2 = est['A'], est['R2']
ess = [r for r in find_ESS(A) if r['is_ess']]
```

---

## Novelty & literature (brief)

- Maps the **theory of the firm** to **evolutionary games** by representing internal allocation as **strategy mixtures** and decomposing observed payoffs into **common‑interest** vs **zero‑sum** components via an explicit projector $M_Z$.\n
- Unifies strands from political economy (labor vs capital shares), empirical corporate finance (capital allocation & payouts), and evolutionary dynamics (replicator, ESS), yielding a compact, testable interaction field `A`.\n
- Related ideas exist (labor‑share time‑series; meta‑games; inverse population games), but this **end‑to‑end time‑series construction** with domain‑legible payoffs and **stability tests** is a **methods contribution** rather than a purely theoretical advance.\n

---

## Caveats & ethics

- **Endogeneity**: allocations and outcomes co‑determine; prefer lagged targets, event windows (policy/union wins), or instrumented designs.\n
- **Measurement**: reports may lack consistent headcount/compensation; triangulate with industry aggregates; document assumptions.\n
- **Attribution**: findings are conditional on payoff definition and horizon $H$; report robustness across payoff families.\n
- **Reporting**: include negative/ambiguous results; they inform about volatility/nonstationarity rather than “winning.”\n

---

## Files\n

- `FIRM_DEMO.ipynb`: synthetic example (distributional and value‑gradient payoffs; ESS; nulls).\n
- `core.py`: scaffold (`nmf_on_X`, payoff builders, `estimate_A_from_series`, `find_ESS`).\n
- `README.md`: project overview and references.\n
