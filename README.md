# timeseries-to-egt (`ts2eg`)

[![CI](https://github.com/flengyel/timeseries-to-egt/actions/workflows/ci.yml/badge.svg)](https://github.com/flengyel/timeseries-to-egt/actions/workflows/ci.yml)

`ts2eg` currently provides geometry utilities, payoff constructors, a legacy
payoff-field fit, and matrix-level equilibrium diagnostics for multivariate time
series. The legacy pipeline is scientifically flawed as an empirical detector:
it has no validated capability to detect dynamics, interaction, regimes, cycles,
ESS in observations, or domain effects. Its repair is specified in the v2 design
documents but the motion estimator is not implemented. The current package also
supplies no calibrated hypothesis test for strategic interaction or any domain
application.

## Scientific status

The present public path is retained for compatibility and exploration:

| Stage | Operation | Current interpretation |
| --- | --- | --- |
| Player-space decomposition | Split a payoff vector into aggregate and mean-zero relative components | Exact linear decomposition |
| Payoff construction | Static profiles, predictive MSE reductions, growth rates, or application-supplied values | Definition-dependent; predictive gains are associations |
| Basis construction | Nonnegative matrix factorization | Reconstruction tool, not a validated selector of replicator coordinates |
| Mixture re-inference | Project each observation onto the simplex against a normalized basis | Legacy convex-hull fit with no amplitude term |
| Operator fit | Regress contemporaneous centered payoff signals on inferred mixtures | Descriptive payoff-field coefficient matrix, not a motion estimator |
| Equilibria | Enumerate Nash candidates and report ESS and local-stability diagnostics | Properties of the fitted matrix only |

The deprecated `surrogate_ess_frequency` routine is uncalibrated. Its output is
a legacy descriptive frequency, not a p-value, significance measure, or causal
result. No replacement inferential procedure is available at this milestone.

## Aggregate and relative payoff channels

For a player-level payoff vector $v(t)\in\mathbb R^N$, define

$$
\bar v(t)=\frac1N\mathbf 1^\top v(t),\qquad
v^Z(t)=v(t)-\bar v(t)\mathbf 1.
$$

The aggregate channel $\bar v(t)\mathbf 1$ records common creation or
destruction under the chosen payoff definition. The relative channel $v^Z(t)$
records advantage relative to the player mean. A mean-zero vector alone does not
define a zero-sum or constant-sum strategic game.

The legacy operator estimators currently pass only $v^Z$ forward and discard
the aggregate channel. They therefore cannot measure aggregate creation or
destruction. Payoff builders that return `payoffs_I` and `payoffs_Z` expose both
components before this legacy loss.

For strictly positive weights $w$, the weighted projection uses

$$
\pi=\frac{w}{\sum_iw_i},\qquad
M_I^{(w)}=\mathbf1\pi^\top,\qquad
M_Z^{(w)}=I-M_I^{(w)}.
$$

## Payoff constructors

- `static_game_from_series` returns profile-indexed conditional next-step
  payoff vectors and their aggregate/relative decomposition.
- `var_information_sharing_game` and its seasonal variant return
  profile-indexed, in-sample predictive MSE reductions. These are predictive
  associations. A separate, explicit profile-to-time mapping is required before
  they can be supplied to an operator estimator.
- `growth_payoffs` returns a per-time log-growth field.
- `value_gradient_payoffs` fits one full-sample ridge coefficient vector and
  repeats it over time. It is not a time-varying gradient.
- `info_gain_payoffs` reports global or rolling in-sample leave-one-feature-out
  MSE differences. It does not identify causal influence.

## Reconstruction coordinates and legacy mixtures

`nmf_on_X(X, k)` returns a reconstruction $X\approx SH$. Reconstruction error
does not establish that the columns of $S$ define coordinates in which the
replicator equation holds.

The current operator estimators do not use the returned $H$. They normalize
the columns of $S$ and independently compute

$$
x(t)\in\arg\min_{x\in\Delta_k}\|Sx-X_{\cdot,t}\|_2^2
$$

by projected gradient. This is a simplex-constrained convex-hull approximation,
not a cone fit, and it contains no separate amplitude variable. Use the same
preprocessing scale for basis fitting and mixture re-inference. A fitted,
train-only preprocessing/measurement object is deferred to a later milestone.

## Legacy payoff-field operator

Let $G=S^\top v^Z$, $M_k=I_k-\mathbf1\mathbf1^\top/k$, and
$G_c=M_kG$. The current estimator fits

$$
A_{\rm raw}=G_cX_k^\top(X_kX_k^\top+\rho I_k)^{-1},\qquad
B=M_kA_{\rm raw}.
$$

The public compatibility key `A` contains $B$; `A_raw` contains the raw ridge
coefficient. Left canonicalization is required because replicator dynamics
identify payoff matrices under

$$
A\sim A+\mathbf1c^\top.
$$

The canonical representative satisfies $\mathbf1^\top B=0$. Right-centering
is not an identification normalization and generally changes the replicator
field.

This fit uses contemporaneous payoff signals. It never estimates
$\dot x$, finite differences, or centered log-ratio motion, so it does not show
that replicator dynamics generated the observations.

`R2` is retained as a compatibility key for the in-sample zero-field score

$$
R^2_{\rm in}=1-\frac{\|G_c-BX_k\|_F^2}{\|G_c\|_F^2}.
$$

The score is undefined when $G_c=0$, and the current API returns `nan` in that
case. No held-out score is returned by this estimator.

## Descriptive equilibrium diagnostics

For a supplied payoff matrix, `find_ESS`:

- enumerates supports and checks symmetric Nash inequalities;
- reports `ess_status` as `certified`, `rejected`, or `undetermined` using the
  symmetric payoff part on an orthonormal support tangent basis;
- reports Jacobian tangent and transverse-invasion eigenvalues separately from
  ESS certification.

`is_ess` is a compatibility alias for `ess_status == "certified"`. A false value
is inconclusive when the status is `undetermined`. Singular KKT systems are
skipped, so continua of equilibria and some degenerate candidates may be absent
from the returned list.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install -e .
```

Python 3.9 or later is required. CI runs the unit-test matrix on Python
3.9–3.13 and runs the committed fast notebooks on Python 3.12.

## Descriptive quick start

This stipulated example uses the same share matrix for NMF and legacy mixture
re-inference. It demonstrates software behavior; it does not validate latent
replicator coordinates or operator recovery.

```python
import numpy as np
import ts2eg

rng = np.random.default_rng(7)
counts = np.maximum(rng.lognormal(size=(5, 120)), 1e-8)
X = counts / counts.sum(axis=0, keepdims=True)
v = ts2eg.growth_payoffs(counts, dt=1.0)

S, _ = ts2eg.nmf_on_X(X, k=3, iters=150, seed=7)
fit = ts2eg.estimate_A_from_series(S, X, v, k=3, ridge=1e-2)

records = ts2eg.find_ESS(fit["A"], max_support=3)
print("In-sample payoff-field R2:", fit["R2"])
print("Fitted-matrix ESS-certified supports:", [
    r["support"] for r in records if r["ess_status"] == "certified"
])
```

## Public API

```text
# geometry
centering_projector(k)
canonicalize_payoff_operator(A)
replicator_field(A, x)
helmert_Q(N)
projectors(N)
weighted_projectors(weights)

# payoff construction
static_game_from_series(...)
var_information_sharing_game(...)
var_information_sharing_game_seasonal(...)
project_out_identities(...)
value_gradient_payoffs(...)
growth_payoffs(...)
info_gain_payoffs(...)

# legacy reconstruction/payoff-field path
nmf_on_X(...)
estimate_A_from_series(...)
estimate_A_from_series_weighted(...)

# descriptive equilibria and deprecated compatibility
find_ESS(...)
surrogate_ess_frequency(...)
```

The regularization keyword is `ridge`. The pre-v0.1.1 `lambda_` spelling is not
accepted.

## Repository layout

```text
src/ts2eg/
  geometry.py       payoff equivalence and replicator-field geometry
  core.py           payoff builders and the legacy descriptive path
  extensions.py     weighted variants, seasonal VAR, deprecated IAAFT harness
tests/
  test_geometry.py
  test_equilibria.py
  test_documentation_contract.py
  test_repository_hygiene.py
notebooks/           synthetic, output-free descriptive demonstrations
docs/               mathematical background, design contracts, domain notes
scripts/             notebook audit, direct execution, and repository checks
examples/            command-line descriptive smoke example
```

All domain notes and notebooks are proposed or stipulated demonstrations. No
capital–labor, financial, biological, sociological, protest, or cybersecurity
test has been statistically validated.

## Development checks

```bash
python -m pip install -e .
pytest -q
MPLBACKEND=Agg TS2EG_CI=1 \
  python scripts/run_notebooks.py --tag fast --timeout 120
```

See [the mathematical background](docs/ts2eg_math_background.tex),
[the statistical contract](docs/design/STATISTICAL_CONTRACT.md), and
[the redesign plan](docs/exec-plans/TS2EG_V2_REDESIGN.md).

## Reproducibility

Record the input/preprocessing definition, payoff construction, basis seed,
$k$, `ridge`, and commit SHA. Treat basis-seed and rolling-window comparisons
as descriptive unless a separately specified selection and calibration design
has been executed.

## License and citation

MIT.

```bibtex
@misc{ts2eg_egt,
  title  = {timeseries-to-egt (ts2eg)},
  author = {Florian Lengyel},
  year   = {2025},
  note   = {Version <tag-or-commit>}
}
```
