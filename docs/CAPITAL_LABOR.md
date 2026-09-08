# CAPITAL–LABOR: Exploratory Payoff-Field Contrasts

This note records capital–labor input and payoff conventions that can exercise
the current legacy `ts2eg` compatibility path. It does not recommend that path
as a detector, estimate capital–labor mixture motion, identify causal
redistribution, or provide a calibrated test of systematic surplus capture.

The legacy pipeline is scientifically flawed as a detection procedure and has no
validated ability to detect capital–labor dynamics, interaction, regimes, cycles,
ESS in data, redistribution, or surplus capture.

The specified repair is the unimplemented Milestone-3 observed-mixture motion
estimator: supplied mixtures are transformed to centered log-ratio increments and
used to fit a canonical motion operator. Nothing in the current compatibility
path is a substitute for that estimator.

> **Current status.** NMF is a reconstruction tool, not a validated selector of
> replicator coordinates. `estimate_A_from_series` fits a contemporaneous payoff
> field and returns an in-sample `R2`. Nash, ESS-certificate, and local-stability
> records describe only that fitted operator. No observed surplus-capture test is
> validated at this milestone.

## Observation design

Rows may represent allocation or accounting channels such as labor compensation,
capital expenditure, R&D, SG&A, and payouts. These channels are not automatically
game-theoretic players. Column normalization to allocation shares is a modeling
choice and must be documented.

`nmf_on_X` provides nonnegative reconstruction factors. The legacy estimator
does not use its returned `H`; it normalizes the loading matrix and independently
re-infers simplex coefficients for each observation. These coefficients are
exploratory allocation components, not identified replicator strategies.

`project_out_identities` may help diagnose homogeneous accounting constraints,
but its output can be negative and need not retain column sums. Do not pass that
output silently to an NMF or allocation-share model. Fit every preprocessing
transformation and basis on a common scale.

## Candidate descriptive payoffs

### Distributional contrast

For strictly positive value added `VA`, labor compensation `W`, and operating
surplus `P`, one may define

\[
v_L(t)=\Delta\log\!\left(\frac{W(t)}{VA(t)}\right),\qquad
v_C(t)=\Delta\log\!\left(\frac{P(t)}{VA(t)}\right).
\]

These are changes in selected accounting shares. Operating surplus must not be
assigned to a `Payouts` channel unless that identification is separately
justified. A larger capital than labor contrast does not by itself establish a
conserved transfer, appropriation, or causal capture.

### Predictive ridge coefficient

`value_gradient_payoffs(X, J_future)` fits one pooled full-sample ridge model

\[
J_{\mathrm{future}}\approx\alpha+\beta^\top X
\]

and returns

\[
v_i(t)=\beta_i
\]

for every time `t`. The output is a time-constant in-sample coefficient
association under the chosen scaling and penalty. It is not a causal value
gradient, a marginal contribution, or evidence of a shareholder-value optimum.

### Predictive ablation

`info_gain_payoffs(X, J_future, ridge, window)` computes leave-one-channel-out
ridge MSE differences on the fitting sample, globally or within each rolling
window. It is an in-sample predictive association, not held-out information gain,
causal leverage, rent, or appropriation.

Use one payoff definition per descriptive fit and report sensitivity without
selecting a definition because it favors a desired interpretation.

## Aggregate and relative channels

For a player-level payoff vector,

\[
v(t)=\bar v(t)\mathbf 1+v^Z(t),\qquad
v^Z(t)=v(t)-\bar v(t)\mathbf 1.
\]

The aggregate and mean-zero relative channels carry different information. The
legacy estimator discards the aggregate channel and fits only `v^Z`. It
therefore cannot measure aggregate value creation or destruction. A mean-zero
relative vector does not establish a zero-sum strategic interaction.

## Why the proposed row-shift test is degenerate

Consider the descriptive contrast

\[
D=\left\langle v_C^Z\right\rangle_t-
  \left\langle v_L^Z\right\rangle_t ,
\]

where \(\langle\cdot\rangle_t\) denotes the time average. At every time,

\[
v_C^Z(t)-v_L^Z(t)
=\bigl(v_C(t)-\bar v(t)\bigr)
 -\bigl(v_L(t)-\bar v(t)\bigr)
=v_C(t)-v_L(t).
\]

Hence

\[
D=\left\langle v_C\right\rangle_t-
  \left\langle v_L\right\rangle_t.
\]

Let \(\tau_C\) and \(\tau_L\) be arbitrary circular shifts applied independently
to the two rows, followed by recomputation of the centered payoff. Circular
shifts preserve row means, so every surrogate satisfies

\[
D^*
=\left\langle\tau_C v_C\right\rangle_t-
  \left\langle\tau_L v_L\right\rangle_t
=D.
\]

The surrogate distribution is therefore a point mass at the observed statistic.
It supplies no null distribution, z-score, or empirical p-value. For
`value_gradient_payoffs`, the failure is stronger: each row is constant in time,
so a circular shift leaves the entire payoff array unchanged.

`D` may be reported as a descriptive difference of time-averaged relative
payoffs. The existing row-shift procedure cannot test it, and no observed
surplus-capture test is validated at the current milestone.

## Legacy descriptive operator

The current estimator fits

\[
G_c\approx A X_k
\]

at the same times. It does not use changes in `X_k`, so it does not estimate an
operator that generated allocation motion. The symmetric and antisymmetric parts
of `A` are algebraic summaries of this payoff-field fit. A large skew norm does
not identify bargaining cycles, automation cycles, or allocation regimes.

`find_ESS(A)` reports descriptive equilibrium and local-linearization records.
It skips singular KKT systems and may omit continua of equilibria. Rolling
changes in `A`, its skew ratio, or its equilibrium records do not establish
regime change.

## Minimal descriptive example

```python
from ts2eg import (
    estimate_A_from_series,
    find_ESS,
    nmf_on_X,
    value_gradient_payoffs,
)

# X: nonnegative N x T allocation shares
v = value_gradient_payoffs(X, J_future=roic_lead, ridge=1e-2)
S, _ = nmf_on_X(X, k=3, iters=300, seed=1)
est = estimate_A_from_series(S, X, v, k=3, ridge=1e-2)

A = est["A"]
in_sample_R2 = est["R2"]
records = find_ESS(A)
certified = [r for r in records if r["ess_status"] == "certified"]
```

This code executes the legacy descriptive payoff-field path only. Its
time-constant predictive coefficients and in-sample fit do not validate a
capital–labor hypothesis.

Lagging a target or selecting an event window does not remove confounding.
Causal attribution requires a separately specified intervention, natural
experiment, randomized variation, or structural identification argument.

See `notebooks/FIRM_DEMO.ipynb` for a stipulated synthetic demonstration. It
must not be treated as null calibration or evidence about observed firms.
