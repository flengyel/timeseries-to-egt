# gamify-timeseries


## Applications & Demos

This repository provides an end-to-end pipeline for converting multivariate time series into **evolutionary games** and analyzing stability (ESS), cyclic dynamics, and zero‑sum vs common‑interest structure.

### Finance
- **Capital–Labor analysis** — mapping firm allocation to EGT; falsifiable surplus‑capture tests.  
  - Notebook: `FIRM_DEMO.ipynb` (includes rolling‑window metrics & a real‑data stub)  
  - Doc: `CAPITAL_LABOR.md`
- **Portfolio/Factor rotation & ensemble governance** — schedule competing sleeves/models; detect regime shifts.  
  - Notebook: `FINANCE_DEMO.ipynb` (if present)
  - Doc: `FINANCE.md` (if present)

### Biology
- **Ecology/Microbiology/Cancer** — per‑capita growth and info‑gain payoffs; interaction fields and ESS.  
  - Notebook: `BIO_DEMO.ipynb`  
  - Doc: `BIOLOGY.md`

### Mathematical Sociology
- **Coalitions/Protests/Governance/Diffusion** — growth payoffs and outcome‑driven info‑gain payoffs.  
  - Notebooks: `SOCIO_DEMO.ipynb`, `PROTEST_DEMO.ipynb`  
  - Doc: `SOCIOLOGY.md`

### Cybersecurity
- **Detector portfolio governance / deception / workflow orchestration** — information‑gain to incident cost.  
  - Notebook: `CYBER_DEMO.ipynb`  
  - Doc: `CYBERSECURITY.md`

### SETI (Toy)
- **Spectral band allocation demo** — *toy only* to exercise the method with strong nulls.  
  - Notebook: `SETI_DEMO.ipynb`

> All demos use the common scaffold in `gameify_timeseries.py` (projectors, payoff builders, `nmf_on_X`, `estimate_A_from_series`, `find_ESS`).



**Data template:** a CSV skeleton is generated on first run at `/mnt/data/firm_quarterly_template.csv`.
