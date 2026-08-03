#!/usr/bin/env bash
set -euo pipefail
F="README.md"
START="<!-- ts2eg-intro-exact-start -->"
END="<!-- ts2eg-intro-exact-end -->"

[[ -f "$F" ]] || { echo "ERROR: README.md not found"; exit 2; }

# idempotent: if already present, do nothing
if grep -q "$START" "$F"; then
  echo "README already contains the exact intro; nothing to do."
  exit 0
fi

blk="$(mktemp)"
cat > "$blk" <<'MD'
<!-- ts2eg-intro-exact-start -->

## Introduction

`ts2eg` maps a multivariate time series $X\in\mathbb{R}^{N\times T}$ to specified payoff fields, fitted evolutionary-game operators, and descriptive equilibrium diagnostics.

* The pipeline separates common-interest effects ($M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top$) from relative-payoff effects ($M_Z=I_N-M_I$).
* Nash, ESS-certificate, and local-stability results describe the fitted operator only.

> **Inferential status:** the former surrogate ESS-frequency procedure is uncalibrated, deprecated, and must not be used as a p-value or hypothesis test. This milestone introduces no replacement inferential test.

See mathematical background: `docs/ts2eg_math_background.tex`.

## Pipeline (end-to-end)

| Stage                   | Input           | Core method                     | Output                                                   |
| ----------------------- | --------------- | ------------------------------- | -------------------------------------------------------- |
| 1. Signal isolation     | $X$             | Project with $M_Z$              | Zero-sum payoff signal $v_Z(t)$                          |
| 2. Payoff induction     | $X$             | VAR information-sharing (ridge) | Per-time payoffs $v(t)$                                  |
| 3. Strategy learning    | $X$             | Dictionary learning (NMF)       | $S\in\mathbb{R}^{N\times k}$, mixtures $x(t)\in\Delta_k$ |
| 4. Game estimation      | $v_Z(t),\,x(t)$ | Centered ridge regression       | $A\in\mathbb{R}^{k\times k}$                             |
| 5. Equilibrium analysis | $A$             | Nash checks and ESS certificate | Descriptive statuses and stability eigenvalues           |

## Mathematical setup (minimal)

* **Projectors.** $M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top,\; M_Z=I_N-M_I.$ Helmert $Q$ gives an orthonormal basis with $q_1\propto \mathbf{1}$. Weighted versions $M_I^{(w)},M_Z^{(w)}$ are supported conceptually.
* **Strategies.** Factor $X\approx S H$ (NMF) with $S\in\mathbb{R}^{N\times k}_{\ge0}$, $H\in\mathbb{R}^{k\times T}_{\ge0}$; define $x(t)$ as the $t$-th column of $H$ re-normalized to $\Delta_k$.
* **Signals.** $g(t)=S^\top v_Z(t)$; center with $M_Z^{(k)}=I_k-\tfrac{1}{k}\mathbf{1}\mathbf{1}^\top$.
* **Estimator.**

  $$
    C_{xx}=X_k X_k^\top,\quad C_{gx}=G_c X_k^\top,\quad
    A_{\mathrm{raw}} = C_{gx}\,(C_{xx}+\rho I_k)^{-1},\ \rho=\texttt{ridge},\quad
    A=M_Z^{(k)}A_{\mathrm{raw}}.
  $$

  Code path uses `np.linalg.solve` with `pinv` fallback and enforces $\mathbf{1}^\top A=0$ without right-centering.
* **Descriptive equilibria.** Replicator $\dot{x}_i=x_i((Ax)_i-x^\top A x)$. `find_ESS` reports Nash checks, an ESS certificate, and separate local-stability eigenvalues.

## Interpretation

Equilibrium statuses are properties of the fitted operator, not inferential conclusions about the data-generating process.

## Numerical/robustness notes

* Invariance to level shifts via $M_Z$.
* Canonical operator $A$ satisfies $\mathbf{1}^\top A=0$; $A\mathbf{1}=0$ is not imposed.
* Ridge regularization with pseudo-inverse fallback for near-singular $C_{xx}$.
* Weighted variants guard against $\sum w=0$.
* Reproducible examples (`--seed`), NumPy 2-compatible.

---

<!-- ts2eg-intro-exact-end -->
MD

# Insert under CI badge if present; else after first H1
tmp="$(mktemp)"
badge_line=$(grep -n 'actions/workflows/ci\.yml/badge\.svg' "$F" | head -n1 | cut -d: -f1 || true)

if [[ -n "${badge_line:-}" ]]; then
  awk -v n="$badge_line" -v blk="$blk" '
    NR==n { print; print ""; while ((getline L < blk) > 0) print L; close(blk); next }
    { print }
  ' "$F" > "$tmp"
else
  awk -v blk="$blk" '
    BEGIN{done=0}
    NR==1 && $0 ~ /^# / { print; print ""; while ((getline L < blk) > 0) print L; close(blk); done=1; next }
    { print }
    END{ if(!done) exit 1 }
  ' "$F" > "$tmp"
fi

mv "$tmp" "$F"
rm -f "$blk"

# Best-effort newline normalization on Windows
unix2dos -q "$F" >/dev/null 2>&1 || true

echo "README updated with exact detailed intro."
