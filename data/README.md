# Local Data for Legacy Demonstrations

The notebooks are designed to execute headlessly with stipulated synthetic or
stub data. They exercise a scientifically unvalidated legacy payoff-field
pipeline with no demonstrated ability to detect dynamics, strategic interaction,
regimes, cycles, ESS in observed data, or domain effects. Their fitted-matrix,
equilibrium, and local-linearization outputs are compatibility diagnostics only.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator based on centered log-ratio increments. Supplying a CSV to the current
compatibility notebooks does not implement or validate that repair.

Adding observed data does not turn a demonstration into a validated analysis.
The current estimator fits contemporaneous payoffs, returns an in-sample `R2`,
uses an unvalidated NMF-based latent representation, and discards the aggregate
payoff channel. No notebook null procedure supplies calibrated inference at the
current milestone.

## Optional local files

The committed notebooks do not read or write local datasets; all seven are
self-contained stipulated demonstrations. The ignored `data/` directory is a
conventional location for local exploratory inputs only. Connecting one of those
inputs requires an explicit, reviewable notebook or script change.

For example, a possible firm-data schema would contain:

- `date`, such as `2010Q1`;
- nonnegative flow columns such as `LaborComp`, `CapEx`, `RnD`, `SGA`, and
  `Payouts`;
- `VA`, `W`, and `P` for value added, labor compensation, and operating surplus;
- optional predictive targets such as `ROIC`.

The meanings, units, signs, accounting relationships, and positivity conditions
must be checked before use. Operating surplus must not be identified with payouts
without a separate justification.

The committed demonstrations are:

- `notebooks/BIO_DEMO.ipynb`
- `notebooks/CYBER_DEMO.ipynb`
- `notebooks/FINANCE_DEMO.ipynb`
- `notebooks/PROTEST_DEMO.ipynb`
- `notebooks/SOCIO_DEMO.ipynb`
- `notebooks/egt_timeseries_demo.ipynb`

## Reproducible execution

Keep local CSV files small and untracked. Do not add downloads or API calls to the
offline notebook path. Run a selected notebook with:

```bash
MPLBACKEND=Agg TS2EG_CI=1 \
  python scripts/run_notebooks.py --only FIRM_DEMO.ipynb --timeout 180
```

The runner reports each executed notebook and stops at the first failing cell.
It does not modify committed notebooks or synthesize executed notebook copies.

Any future inferential analysis must specify an estimand, null hypothesis,
statistic, and validation gate; fit preprocessing and latent bases on training
data only; keep selection and final test data separate; and repeat the complete
selection rule inside every resampling replicate. Those requirements are not
satisfied merely by supplying a CSV to these notebooks.
