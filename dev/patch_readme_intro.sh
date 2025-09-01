#!/usr/bin/env bash
set -euo pipefail

README="README.md"
SECTION_H1='## Introduction — what `ts2eg` tests'

if [[ ! -f "$README" ]]; then
  echo "ERROR: $README not found. Run this from the repo root."
  exit 1
fi

# Abort if already present (idempotent)
if grep -Fq "$SECTION_H1" "$README"; then
  echo "README already contains: $SECTION_H1"
  exit 0
fi

# Build the insertion block in a temp file
INS="$(mktemp)"
cat > "$INS" <<'MD'
## Introduction — what `ts2eg` tests

`ts2eg` is an **inverse evolutionary game theory** pipeline that tests whether a multivariate time series \(X\in\mathbb{R}^{N\times T}\) contains **competitive (zero-sum) strategic interaction**.

- The test asks: after removing individual dynamics and shared environment, **is there evidence that agents interact through relative payoffs** consistent with replicator dynamics?
- The pipeline **projects away common-interest effects** (\(M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top\)) and **retains only the zero-sum signal** (\(M_Z=I_N-M_I\)). A purely cooperative process (signal in the \(M_I\) direction) is **filtered out** and yields a null result.

**Hypotheses.**  
\(H_0\): any apparent equilibrium structure arises from per-series autocorrelation only (no cross-player coupling).  
\(H_1\): there exists a centered payoff operator \(A\) in strategy space that produces an ESS under replicator dynamics; the effect exceeds the IAAFT surrogate null.

See mathematical background: `docs/ts2eg_math_background.tex`.

## Pipeline (end-to-end)

| Stage | Input | Core method | Output |
|---|---|---|---|
| 1. Signal isolation | \(X\) | Project with \(M_Z\) | Zero-sum payoff signal \(v_Z(t)\) |
| 2. Payoff induction | \(X\) | VAR information-sharing (ridge) | Per-time payoffs \(v(t)\) |
| 3. Strategy learning | \(X\) | Dictionary learning (NMF) | \(S\in\mathbb{R}^{N\times k}\), mixtures \(x(t)\in\Delta_k\) |
| 4. Game estimation | \(v_Z(t),\,x(t)\) | Centered ridge regression | \(A\in\mathbb{R}^{k\times k}\) |
| 5. Equilibrium analysis | \(A\) | Replicator Jacobian test | ESS presence/absence |
| 6. Significance | \(X\) | IAAFT surrogates | Empirical \(p\)-value |

## Mathematical setup (minimal)

- **Projectors.** \(M_I=\tfrac{1}{N}\mathbf{1}\mathbf{1}^\top,\; M_Z=I_N-M_I.\) Helmert \(Q\) gives an orthonormal basis with \(q_1\propto \mathbf{1}\). Weighted versions \(M_I^{(w)},M_Z^{(w)}\) are supported conceptually.
- **Strategies.** Factor \(X\approx S H\) (NMF) with \(S\in\mathbb{R}^{N\times k}_{\ge0}\), \(H\in\mathbb{R}^{k\times T}_{\ge0}\); define \(x(t)\) as the \(t\)-th column of \(H\) re-normalized to \(\Delta_k\).
- **Signals.** \(g(t)=S^\top v_Z(t)\); center with \(M_Z^{(k)}=I_k-\tfrac{1}{k}\mathbf{1}\mathbf{1}^\top\).
- **Estimator.**
  \[
    C_{xx}=X_k X_k^\top,\quad C_{gx}=G_c X_k^\top,\quad
    A = C_{gx}\,(C_{xx}+\rho I_k)^{-1},\ \rho=\texttt{ridge},\quad
    A\leftarrow M_Z^{(k)}A M_Z^{(k)}.
  \]
  Code path uses `np.linalg.solve` with `pinv` fallback and enforces row/column centering.
- **Dynamics/ESS.** Replicator \(\dot{x}_i=x_i((Ax)_i-x^\top A x)\). `find_ESS` enumerates supports and tests local stability via the Jacobian on the tangent space.

## What a positive result means

- **Positive (reject \(H_0\)):** evidence for **competitive, zero-sum interaction** in \(X\) that cannot be explained by per-series autocorrelation alone.
- **Null:** either no strategic coupling, or only cooperative/common-interest structure (deliberately excluded by \(M_Z\)).

## Numerical/robustness notes

- Invariance to level shifts via \(M_Z\).
- Centered operator \(A\) (\(A\mathbf{1}=0\) and \(\mathbf{1}^\top A=0\)) by construction.
- Ridge regularization with pseudo-inverse fallback for near-singular \(C_{xx}\).
- Weighted variants guard against \(\sum w=0\).
- Reproducible examples (`--seed`), NumPy 2-compatible.
MD

# Choose insertion point: after badge if present, else after first H1
badge_line=$(grep -n 'actions/workflows/ci\.yml' "$README" | head -n1 | cut -d: -f1 || true)
if [[ -n "${badge_line:-}" ]]; then
  insert_after="$badge_line"
else
  insert_after=$(awk '/^# /{print NR; exit}' "$README")
fi

if [[ -z "${insert_after:-}" ]]; then
  echo "ERROR: could not locate insertion anchor (badge or first H1)."
  rm -f "$INS"
  exit 2
fi

# Backup then apply
ts=$(date +%Y%m%d-%H%M%S)
cp -v "$README" "${README}.bak.${ts}"

awk -v n="$insert_after" -v f="$INS" 'NR==n{print; system("cat \""f"\""); next} {print}' "$README" > "${README}.new"
mv "${README}.new" "$README"
rm -f "$INS"

echo "Inserted section after line $insert_after."
echo
echo "Preview diff (top 200 lines):"
git --no-pager diff -- "$README" | sed -n '1,200p' || true
echo
echo "Next:"
echo "  git add README.md && git commit -m \"docs: add intro/pipeline from report\" && git push"
