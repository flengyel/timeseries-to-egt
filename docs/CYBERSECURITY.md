# CYBERSECURITY: Detector Portfolios as Evolutionary Games

This note explains how to use the `gamify-timeseries` pipeline for **IDS/SIEM/EDR/NDR** governance, deception portfolios, and response orchestration.

---

## Mapping

- **Players**: detectors/rule families (signature, anomaly, DNS, sandbox, EDR, ML), deception assets, or response automation steps.\n
- **Signals (`X ∈ ℝ^{N×T}`)**: per-interval shares of alerts/compute/triage time per player (columns sum to 1).\n
- **Strategies (`S`)**: archetypes of detector usage learned by NMF/archetypal analysis (column-normalized, nonnegative). Each time \(t\), the system is a mix \(x(t) \in \Delta_k\).\n
- **Payoffs (`v`)**: **information-gain** toward an external target \(J\): negative future incident cost (dwell time, lateral spread, escalations), SLA success, or risk score. Use rolling LOFO ablation: `info_gain_payoffs(X, J, window)`.\n
- **Operator (`A`)**: estimated strategy-level interaction field; decompose into symmetric \(A_s\) (potential-like) and skew \(A_a\) (cycles). ESS = stable detector mix.

---

## Research directions

1. **Adaptive detector routing (portfolio governance)**\n
   - Question: do mixes stabilize (ESS) under normal load and rotate (large \(\\|A_a\\|\)) during campaigns?\n
   - Design: hourly `X` from SIEM (alert share per detector) and weekly `J` (−dwell time). Rolling `A` and ESS; intervention checks at rule/policy updates.\n

2. **Deception portfolio optimization**\n
   - Players: honeypot classes/placements. Target `J`: intel value or derailment score.\n
   - Hypothesis: environments with diverse attacker TTPs require skew-dominant \(A\) (cycles) and mix rebalancing.\n

3. **Response workflow orchestration**\n
   - Players: SOAR playbook steps/tools. Target `J`: −time-to-containment, with cost of false positives.\n
   - Use replicator field as a controller to nudge time allocation toward stable, effective mixes.\n

4. **Segmented lateral-movement monitoring**\n
   - Players: segment-specific sensors. Target `J`: −(cross-segment infection) at horizon \(H\).\n
   - Readout: \(A_a\) reveals attack–defense oscillations; ESS breakdown flags regime shifts.\n

5. **Red-team exercises as controlled interventions**\n
   - Use toggled rules/assets to validate causal direction: does the vector field predict the observed reallocation?\n

---

## Evaluation protocol

- **Targets & alignment**: strictly align \(J_{t+H}\) to info known at \(t\); choose \(H\) appropriate to operations (hours–days).\n
- **Costs**: include false-positive rate, analyst time, and compute in \(J\).\n
- **Nulls**: circular row shifts; day-of-week fixed effects; randomly permuted rule-update times; attack-free periods.\n
- **Rolling windows**: report \(R^2\), ESS persistence, and \(\\|A_a\\|/\\|A\\|\) stability.\n
- **Benchmarks**: compare against naive equal-weight routing and simple ridge-weighted ensembles.\n

---

## Minimal code

```python
from gameify_timeseries import info_gain_payoffs, estimate_A_from_series, find_ESS

# X: detectors x time usage shares (normalize columns to 1)
# J: negative incident cost (higher is better)
v = info_gain_payoffs(X, J, ridge=1e-2, window=168)  # weekly window (hourly data)

# Learn S (NMF), estimate A, analyze
S = nmf_on_X(X, k=4)  # any nonnegative NMF; normalize columns
est = estimate_A_from_series(S, X, v, k=4, ridge=1e-2)
A = est['A']
ess = [r for r in find_ESS(A) if r['is_ess']]
```

---

## Cautions

- **Feedback loops**: detectors influence `J`; validate with interventional toggles/red-team runs.\n
- **Concept drift**: retrain in rolling windows; require out-of-sample checks.\n
- **Privacy & security**: aggregate, anonymize, and avoid exposing signatures/TTPs.\n

See `CYBER_DEMO.ipynb` for a runnable synthetic example.
