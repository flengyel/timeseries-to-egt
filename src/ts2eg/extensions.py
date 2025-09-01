
"""
extensions.py
=================
Extensions for the time-series → game toolkit.

Adds:
  1) Weighted projectors (W-inner-product mean & centering)
  2) Seasonal-VAR support for Option B (information-sharing game)
  3) IAAFT surrogate generator + harness that re-estimates A and tallies ESS frequency

This file does NOT modify the base module. It imports and extends it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Optional
import numpy as np

# Import public pieces from the base toolkit
from . import core as gm

# -------- (1) Weighted projectors --------

def weighted_projectors(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted projection onto span{1} and its complement under the W-inner product,
    where W = diag(weights). For v in R^N,
      M_I^(w) v = ((1^T W v) / (1^T W 1)) * 1
    Matrix form: M_I^(w) = 1 * (1^T W) / (1^T W 1) = 1 * (w^T) / sum(w).
    Returns (M_Iw, M_Zw).
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be positive and finite")
    N = w.size
    denom = float(np.sum(w))
    M_Iw = np.ones((N, 1)) @ (w.reshape(1, N) / denom)
    M_Zw = np.eye(N) - M_Iw
    return M_Iw, M_Zw

# -------- helpers: seasonal lag builder & ridge MSE --------

def _build_lagged_seasonal(X: np.ndarray, p: int, seasonal_period: int = 0,
                           seasonal_multiples: Iterable[int] | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build (Y, Z, lag_list) with base lags 1..p plus seasonal lags s*m for m in seasonal_multiples.
    Shapes: Y: (T-max_lag, N), Z: (T-max_lag, N * L), where L = number of lags used.
    """
    N, T = X.shape
    lag_set = set(range(1, p + 1))
    if seasonal_period and seasonal_period > 0 and seasonal_multiples:
        for m in seasonal_multiples:
            lag_set.add(seasonal_period * int(m))
    lag_list = sorted([l for l in lag_set if l < T])
    max_lag = max(lag_list) if lag_list else 0
    Y = X[:, max_lag:].T
    Z_blocks = [X[:, max_lag - lag : T - lag].T for lag in lag_list]
    Z = np.concatenate(Z_blocks, axis=1) if Z_blocks else np.zeros((T - max_lag, 0))
    return Y, Z, lag_list

def _ols_mse(y: np.ndarray, X: np.ndarray, ridge: float = 0.0) -> float:
    """In-sample MSE of OLS / ridge of y on X with intercept."""
    T = y.shape[0]
    X1 = np.concatenate([np.ones((T, 1)), X], axis=1)
    if ridge > 0:
        d = X1.shape[1]
        beta = np.linalg.solve(X1.T @ X1 + ridge * np.eye(d), X1.T @ y)
    else:
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(np.mean(resid ** 2))

# -------- (2) Seasonal VAR Option B (weights optional) --------

def var_information_sharing_game_seasonal(
    X: np.ndarray,
    p: int = 1,
    ridge: float = 0.0,
    restrict_profiles: Optional[Iterable[Tuple[int, ...]]] = None,
    include_self_always: bool = True,
    lambda_common: float = 0.0,
    seasonal_period: int = 0,
    seasonal_multiples: Iterable[int] | None = None,
    weights: np.ndarray | None = None,
) -> gm.VARSharingGameResult:
    """
    Seasonal extension of the base Option B. Binary "share" profile a in {0,1}^N determines which
    players' lagged signals are available to all. For player i, predict x_i(t) from chosen lags of the
    shared set (and i's own if include_self_always). Payoff_i(a) = MSE_i(self-only) − MSE_i(a).
    Center across players using weighted or unweighted projectors.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N x T)")
    N, T = X.shape
    if T <= p + 5:
        raise ValueError("Time series too short for chosen lag p")

    # Projectors
    if weights is None:
        M_I, M_Z = gm.projectors(N)
    else:
        M_I, M_Z = weighted_projectors(weights)

    Y, Z, lag_list = _build_lagged_seasonal(X, p, seasonal_period=seasonal_period, seasonal_multiples=seasonal_multiples)

    def cols_for_players(players: Iterable[int]) -> list[int]:
        idx = []
        P = set(players)
        L = len(lag_list)
        for l in range(L):
            block = l * N
            for j in range(N):
                if j in P:
                    idx.append(block + j)
        return idx

    # Baseline: self-only predictors
    baseline_mse = np.zeros(N)
    for i in range(N):
        cols = cols_for_players([i])
        baseline_mse[i] = _ols_mse(Y[:, i], Z[:, cols], ridge=ridge)

    if restrict_profiles is None:
        if N > 12:
            raise ValueError("N too large for exhaustive 2^N profiles; pass restrict_profiles")
        profiles = [tuple((m >> b) & 1 for b in range(N)) for m in range(1 << N)]
    else:
        profiles = list(restrict_profiles)

    pay, pay_I, pay_Z = {}, {}, {}
    for a in profiles:
        share = {j for j, bit in enumerate(a) if bit == 1}
        u = np.zeros(N)
        for i in range(N):
            predictors = set(share)
            if include_self_always:
                predictors.add(i)
            cols = cols_for_players(predictors)
            mse = _ols_mse(Y[:, i], Z[:, cols], ridge=ridge) if cols else np.var(Y[:, i])
            u[i] = baseline_mse[i] - mse
        u_Z = (M_Z @ u.reshape(-1, 1)).ravel()
        u_I = (M_I @ u.reshape(-1, 1)).ravel()
        pay[a] = u_Z + lambda_common * u_I
        pay_I[a], pay_Z[a] = u_I, u_Z

    meta = {
        "N": N, "T": T, "p": p, "ridge": ridge,
        "profiles": len(profiles),
        "include_self_always": include_self_always,
        "lambda_common": lambda_common,
        "seasonal_period": seasonal_period,
        "seasonal_multiples": list(seasonal_multiples) if seasonal_multiples else None,
    }
    return gm.VARSharingGameResult(pay, pay_I, pay_Z, baseline_mse, meta)

# -------- (3) Weighted A-estimation + IAAFT surrogates & harness --------

def estimate_A_from_series_weighted(S: np.ndarray, X: np.ndarray, v: np.ndarray, k: int,
                                    lambda_: float = 0.0, weights: np.ndarray | None = None):
    """
    Weighted variant of gm.estimate_A_from_series:
      1) vZ = M_Z v (weighted if weights provided).
      2) g(t) = S^T vZ(t).
      3) infer memberships x(t) by projecting X(:,t) onto cone(S) with simplex constraint.
      4) ridge for A in g ≈ A x with row-centering.
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D (N x k)")
    N, kS = S.shape
    if kS != k:
        raise ValueError("k must equal S.shape[1]")
    if X.ndim != 2 or v.ndim != 2:
        raise ValueError("X and v must be 2D (N x T)")
    if X.shape[0] != N or v.shape[0] != N or X.shape[1] != v.shape[1]:
        raise ValueError("Shapes must agree: S=(N,k); X,v=(N,T) with same N,T.")
    T = X.shape[1]

    # Normalize S columns
    S = S.copy().astype(float)
    col_norms = np.linalg.norm(S, axis=0)
    col_norms[col_norms == 0.0] = 1.0
    S /= col_norms

    # Weighted centering across players
    if weights is None:
        M_I, M_Z = gm.projectors(N)
    else:
        M_I, M_Z = weighted_projectors(weights)

    vZ = M_Z @ v
    Gk = S.T @ vZ

    # memberships via base projected-gradient helper
    Xk = np.zeros((k, T))
    for t in range(T):
        Xk[:, t] = gm._infer_membership_via_pg(S, X[:, t], iters=200)

    onek = np.ones((k, 1))
    MZ_k = np.eye(k) - (onek @ onek.T) / k
    Gc = MZ_k @ Gk
    Cxx = Xk @ Xk.T
    Cgx = Gc @ Xk.T
    A = Cgx @ np.linalg.solve(Cxx + lambda_ * np.eye(k), np.eye(k))
    A = MZ_k @ A

    Ghat = A @ Xk
    resid = Gc - Ghat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum(Gc**2)) + 1e-12
    R2 = 1.0 - ss_res / ss_tot
    return {"A": A, "Xk": Xk, "Gk": Gk, "Gc": Gc, "R2": R2, "Cxx": Cxx, "Cgx": Cgx}

def iaaft(x: np.ndarray, n_iter: int = 100, rng: np.random.Generator | None = None) -> np.ndarray:
    """Iterative Amplitude Adjusted Fourier Transform surrogate for a 1D series."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    T = x.size
    x_sorted = np.sort(x)
    Xf = np.fft.rfft(x)
    amp = np.abs(Xf)
    y = np.copy(x)
    rng.shuffle(y)
    for _ in range(n_iter):
        Yf = np.fft.rfft(y)
        Yf = amp * np.exp(1j * np.angle(Yf))
        y = np.fft.irfft(Yf, n=T)
        ranks = np.argsort(np.argsort(y))
        y = x_sorted[ranks]
    return y

def iaaft_matrix(X: np.ndarray, n_iter: int = 100, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, T = X.shape
    Y = np.zeros_like(X, dtype=float)
    for i in range(N):
        Y[i] = iaaft(X[i], n_iter=n_iter, rng=rng)
    return Y

def surrogate_ess_frequency(S: np.ndarray, X: np.ndarray, v: np.ndarray | None, k: int,
                            lambda_: float = 0.0, num_surrogates: int = 50, iaaft_iters: int = 100,
                            tol: float = 1e-8, max_support: int | None = None,
                            weights: np.ndarray | None = None, seed: int = 0):
    """
    Generate IAAFT surrogates (row-wise, breaking cross-player dependence), re-estimate A (weighted),
    and tally ESS frequency using gm.find_ESS.
    If v is None, use v = X.
    Returns dict with 'ess_rate', 'nash_rate', 'R2_mean', 'R2_std'.
    """
    if v is None:
        v = X
    rng = np.random.default_rng(seed)
    ess_count, nash_count = 0, 0
    R2_list = []
    for s in range(num_surrogates):
        seed_s = int(rng.integers(0, 2**32 - 1))
        Xs = iaaft_matrix(X, n_iter=iaaft_iters, seed=seed_s)
        vs = iaaft_matrix(v, n_iter=iaaft_iters, seed=(seed_s ^ 0x9e3779b1))
        est = estimate_A_from_series_weighted(S, Xs, vs, k=k, lambda_=lambda_, weights=weights)
        A = est["A"]
        R2_list.append(est["R2"])
        res = gm.find_ESS(A, tol=tol, max_support=max_support)
        nash_any = any(r["is_nash"] for r in res)
        ess_any = any(r["is_ess"] for r in res)
        nash_count += int(nash_any)
        ess_count += int(ess_any)
    return {
        "ess_rate": ess_count / num_surrogates,
        "nash_rate": nash_count / num_surrogates,
        "R2_mean": float(np.mean(R2_list)),
        "R2_std": float(np.std(R2_list)),
        "num_surrogates": num_surrogates,
    }



