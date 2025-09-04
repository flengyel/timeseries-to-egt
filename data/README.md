# Real data integration (offline-friendly)

This repo’s notebooks are designed to run **headless** with synthetic/stub data.
You can optionally drop small, local CSVs in `data/` to run the same pipelines on real data.

## Current expectations

- **FIRM_DEMO.ipynb**
  - Looks for `data/firm_quarterly.csv`. If missing, it writes a *template* CSV to `/mnt/data/firm_quarterly_template.csv` and continues with stubs.
  - Expected columns (example schema; units are illustrative):
    - `date` (e.g., `2010Q1`, `2010Q2`, …)
    - `LaborComp`, `CapEx`, `RnD`, `SGA`, `Payouts`  → nonnegative flows that will be column-normalized to shares
    - `VA` (value added), `W` (wages), `P` (profits), optional `ROIC`
  - Pipeline stages exercised: Strategy learning → Game estimation → ESS → Surrogates (value-gradient payoffs).

- **BIO_DEMO / CYBER_DEMO / PROTEST_DEMO / SOCIO_DEMO / egt_timeseries_demo / FINANCE_DEMO**
  - Currently self-contained (synthetic). If you add real CSVs later, follow the pattern above:
    1. Load from `data/your_file.csv` (if present), else synthesize.
    2. Keep shapes `(N, T)` and nonnegativity/normalization consistent with the README.
    3. Leave a brief comment with the expected schema.

## CI constraints & tips

- CI runs on Ubuntu with **no network**, and a writable `/mnt/data`. Do **not** rely on downloads/APIs.
- The env var `TS2EG_CI=1` is set in the runner; notebook cells should downshift iterations when present.
- Keep local CSVs small; this folder is ignored by git (see `.gitignore`).

## Minimal recipe to use your own CSV locally

1. Save your CSV(s) under `data/`.
2. Ensure columns match what the notebook expects (see comments at the load sites).
3. Run:
   python scripts/run_notebooks.py --only FIRM_DEMO.ipynb --timeout 180
4. Inspect `artifacts/notebooks/*.executed.ipynb` for outputs.
