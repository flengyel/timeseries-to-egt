"""
core.py
---------------------
Transform multivariate time series into normal-form games and fit an evolutionary game
(Replicator dynamics) directly from data.

Contents
========
Core linear algebra:
  - helmert_Q(N): orthonormal Helmert matrix (q1 = 1/sqrt(N))
  - projectors(N): M_I (mean projector), M_Z (centering projector)

Game constructions:
  - static_game_from_series(X, ...): Option A (discretized state -> conditional next-step payoff)
  - var_information_sharing_game(X, p, ...): Option B (VAR-based "share" profiles -> gains)

EGT add-ons:
  - estimate_A_from_series(S, X, v, k, ridge): fit k×k payoff operator A from series
  - find_ESS(A, ...): enumerate mixed Nash candidates and test ESS via replicator Jacobian

Finance & Bio helpers:
  - project_out_identities(X, C): remove linear accounting identities
  - value_gradient_payoffs(X, J_future, ...): construct payoffs from a scalar value target
  - growth_payoffs(counts, dt, pad): per-capita growth payoffs from counts
  - info_gain_payoffs(X, y_next, ...): ablation-based information-gain payoffs (rolling or global)

All arrays: players × time (N × T).
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List, Optional
import numpy as np

# ---------- Core transforms ----------
def helmert_Q(N: int) -> np.ndarray:
    """Return the N×N Helmert (orthonormal) matrix Q; first column = 1/sqrt(N)."""
    if N < 1:
        raise ValueError("N must be >= 1")
    Q = np.zeros((N, N), dtype=float)
    Q[:, 0] = 1.0 / np.sqrt(N)
    for r in range(2, N + 1):
        Q[0:r-1, r-1] = 1.0 / np.sqrt((r - 1) * r)
        Q[r-1, r-1] = -(r - 1) / np.sqrt((r - 1) * r)
    return Q

def projectors(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (M_I, M_Z) for dimension N: mean projector and centering projector."""
    one = np.ones((N, 1))
    M_I = (one @ one.T) / N
    M_Z = np.eye(N) - M_I
    return M_I, M_Z

# ---------- Option A: Static game from time series ----------
def _bin_terciles(x: np.ndarray) -> np.ndarray:
    """Bin each player's series into {-1,0,+1} using terciles; x is N×T."""
    N, T = x.shape
    labels = np.zeros_like(x, dtype=int)
    for i in range(N):
        q1, q2 = np.quantile(x[i], [1/3, 2/3])
        labels[i] = np.where(x[i] < q1, -1, np.where(x[i] > q2, 1, 0))
    return labels

def _profiles_from_labels(labels: np.ndarray, t_idx: Iterable[int]) -> List[Tuple[int, ...]]:
    """Turn N×T integer labels into tuples per time index."""
    return [tuple(int(labels[:, t][k]) for k in range(labels.shape[0])) for t in t_idx]

@dataclass
class StaticGameResult:
    payoffs: Dict[Tuple[int, ...], np.ndarray]    # profile -> N-vector payoff (original basis)
    payoffs_I: Dict[Tuple[int, ...], np.ndarray]  # common-interest component
    payoffs_Z: Dict[Tuple[int, ...], np.ndarray]  # zero-sum component
    profiles_seen: List[Tuple[int, ...]]
    meta: Dict[str, object]

def static_game_from_series(
    X: np.ndarray,
    lag: int = 1,
    bins: str = "tercile",
    zero_sum_discretization: bool = True,
) -> StaticGameResult:
    """
    Option A. Discretize per-time relative positions and map each joint profile to
    the conditional mean next-step payoff vector X[:, t+lag].

    X : N×T (players × time)
    bins : 'tercile' (can be extended)
    zero_sum_discretization : if True, bin M_Z @ X (deviations); else bin X.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N × T)")
    N, T = X.shape
    if T <= lag + 2:
        raise ValueError("Time series too short for given lag")
    M_I, M_Z = projectors(N)
    Xd = (M_Z @ X) if zero_sum_discretization else X.copy()
    if bins != "tercile":
        raise NotImplementedError("Only 'tercile' binning is implemented")
    labels = _bin_terciles(Xd)
    t_idx = list(range(0, T - lag))
    profiles = _profiles_from_labels(labels, t_idx)
    bucket: Dict[Tuple[int, ...], List[np.ndarray]] = {}
    for s, t in zip(profiles, t_idx):
        bucket.setdefault(s, []).append(X[:, t + lag])
    pay, pay_I, pay_Z = {}, {}, {}
    for s, vecs in bucket.items():
        U = np.mean(np.stack(vecs, axis=1), axis=1)   # N-vector
        pay[s] = U
        pay_I[s] = M_I @ U
        pay_Z[s] = M_Z @ U
    meta = {"N": N, "T": T, "lag": lag, "bins": bins,
            "zero_sum_discretization": zero_sum_discretization,
            "profiles": len(bucket)}
    return StaticGameResult(pay, pay_I, pay_Z, list(bucket.keys()), meta)

# ---------- Option B: VAR-based information-sharing game ----------
@dataclass
class VARSharingGameResult:
    payoffs: Dict[Tuple[int, ...], np.ndarray]    # profile -> centered N-vector of gains
    payoffs_I: Dict[Tuple[int, ...], np.ndarray]
    payoffs_Z: Dict[Tuple[int, ...], np.ndarray]
    baseline_mse: np.ndarray                      # self-only MSE per player
    meta: Dict[str, object]

def _build_lagged(X: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Y, Z) with Y_t = X_t (t=p..), Z_t = [X_{t-1},...,X_{t-p}] stacked; shapes (T-p,N), (T-p,Np)."""
    N, T = X.shape
    Y = X[:, p:].T
    Z = np.concatenate([X[:, p-k:T-k].T for k in range(1, p+1)], axis=1)
    return Y, Z

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

def var_information_sharing_game(
    X: np.ndarray,
    p: int = 1,
    ridge: float = 0.0,
    restrict_profiles: Optional[Iterable[Tuple[int, ...]]] = None,
    include_self_always: bool = True,
    lambda_common: float = 0.0,
) -> VARSharingGameResult:
    """
    Binary “share” profile a∈{0,1}^N determines which players’ lagged signals are
    available to all. For player i, predict x_i(t) from p lags of the shared set (and i’s own if
    include_self_always). Payoff_i(a) = MSE_i(self-only) − MSE_i(a). Center across players.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N × T)")
    N, T = X.shape
    if T <= p + 5:
        raise ValueError("Time series too short for chosen lag p")
    M_I, M_Z = projectors(N)
    Y, Z = _build_lagged(X, p)

    def cols_for_players(players: Iterable[int]) -> List[int]:
        idx = []
        P = set(players)
        for k_lag in range(1, p + 1):
            block = (k_lag - 1) * N
            for j in range(N):
                if j in P:
                    idx.append(block + j)
        return idx

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
        "profile_definition": "bit j=1 means player j shares its signal; predictors are union of shared signals (and self if include_self_always).",
    }
    return VARSharingGameResult(pay, pay_I, pay_Z, baseline_mse, meta)

# ---------- EGT add-ons ----------
def _project_to_simplex(y: np.ndarray) -> np.ndarray:
    """Euclidean projection of y onto the probability simplex {x>=0, sum x = 1} (Duchi et al., 2008)."""
    k = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, k + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    x = np.maximum(y - theta, 0.0)
    return x

def _infer_membership_via_pg(S: np.ndarray, y: np.ndarray, iters: int = 200) -> np.ndarray:
    """
    Project y (N,) onto cone(S) with sum-to-one and nonnegativity using projected gradient on ||Sx - y||^2.
    Returns x in the simplex.
    """
    N, k = S.shape
    L = np.linalg.norm(S.T @ S, 2)
    if not np.isfinite(L) or L <= 0:
        L = 1.0
    eta = 1.0 / L
    x = np.full(k, 1.0 / k)
    for _ in range(iters):
        grad = S.T @ (S @ x - y)
        x = _project_to_simplex(x - eta * grad)
    return x

def estimate_A_from_series(S: np.ndarray, X: np.ndarray, v: np.ndarray, k: int, ridge: float = 0.0):
    """
    Estimate a k×k payoff operator A for a symmetric population game from:
      - S: strategy basis (N×k), columns s_i (prefer nonnegative / normalized)
      - X: data series (N×T), per-player signals
      - v: payoff series (N×T), per-player instantaneous payoffs (e.g., Option A/B output)
      - k: number of strategies (= S.shape[1])
      - ridge: ridge regularization

    Pipeline:
      1) vZ = M_Z v (zero-sum across players).
      2) g(t) = S^T vZ(t)  (strategy-level signals).
      3) Infer memberships x(t) by projected gradient: min ||Sx - X(:,t)|| s.t. x in simplex.
      4) Ridge regression for A from g ≈ A x, with row-centering enforced (replicator-invariant).

    Returns dict with A, memberships Xk, signals Gk/Gc, and R^2 fit.
    """
    if S.ndim != 2:
        raise ValueError("S must be 2D (N × k)")
    N, kS = S.shape
    if kS != k:
        raise ValueError("k must equal S.shape[1]")
    if X.ndim != 2 or v.ndim != 2:
        raise ValueError("X and v must be 2D (N × T)")
    if X.shape[0] != N or v.shape[0] != N or X.shape[1] != v.shape[1]:
        raise ValueError("Shapes must agree: S=(N,k); X,v=(N,T) with same N,T.")
    T = X.shape[1]

    # Normalize S columns for scale stability
    S = S.copy().astype(float)
    col_norms = np.linalg.norm(S, axis=0)
    col_norms[col_norms == 0.0] = 1.0
    S /= col_norms

    # Project payoffs to zero-sum across players
    oneN = np.ones((N, 1))
    MZ_N = np.eye(N) - (oneN @ oneN.T) / N
    vZ = MZ_N @ v                      # (N × T)

    # Strategy-level signals g(t)
    Gk = S.T @ vZ                      # (k × T)

    # Infer memberships x(t) on the simplex
    Xk = np.zeros((k, T))
    for t in range(T):
        Xk[:, t] = _infer_membership_via_pg(S, X[:, t], iters=200)

    # Center strategies (replicator invariance) and ridge regression for A
    onek = np.ones((k, 1))
    MZ_k = np.eye(k) - (onek @ onek.T) / k
    Gc = MZ_k @ Gk
    Cxx = Xk @ Xk.T
    Cgx = Gc @ Xk.T
    try:
        A = Cgx @ np.linalg.solve(Cxx + ridge * np.eye(k), np.eye(k))
    except np.linalg.LinAlgError:
        A = Cgx @ np.linalg.pinv(Cxx + ridge * np.eye(k))
    A = MZ_k @ A @ MZ_k  # enforce row/column centering

    # Fit diagnostics
    Ghat = A @ Xk
    resid = Gc - Ghat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum(Gc**2)) + 1e-12
    R2 = 1.0 - ss_res / ss_tot
    return {"A": A, "Xk": Xk, "Gk": Gk, "Gc": Gc, "R2": R2, "Cxx": Cxx, "Cgx": Cgx}

def find_ESS(A: np.ndarray, tol: float = 1e-8, max_support: int | None = None):
    """
    Enumerate mixed Nash candidates and test ESS for the replicator dynamic in a symmetric k-strategy game.

    For each support J:
      - Solve for (x_J, alpha): A_JJ x_J = alpha * 1_J, 1^T x_J = 1.
      - Nash check: (A x)_i = alpha on J, and <= alpha off J (within tolerances).
      - ESS: (i) strict disadvantage off J; (ii) Jacobian of replicator restricted to the
        tangent of J has eigenvalues with negative real parts.

    Returns a list of dicts {support, x, alpha, is_nash, is_ess, eigvals_tangent}.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square (k × k)")
    k = A.shape[0]
    import itertools

    if max_support is None:
        max_support = k

    results = []
    for s in range(1, min(max_support, k) + 1):
        for J in itertools.combinations(range(k), s):
            J = list(J)
            AJJ = A[np.ix_(J, J)]
            oneJ = np.ones((s, 1))
            # Solve KKT: [AJJ | -1][xJ; alpha] = 0, [1^T | 0][xJ; alpha] = 1
            KKT = np.block([[AJJ, -oneJ], [oneJ.T, np.zeros((1, 1))]])
            rhs = np.concatenate([np.zeros((s, 1)), np.array([[1.0]])], axis=0)
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                continue
            xJ = sol[:s, 0]
            alpha = float(sol[s, 0])
            if np.any(xJ < -1e-6):
                continue
            # Embed in simplex
            x = np.zeros(k)
            x[J] = np.maximum(xJ, 0.0)
            x = x / x.sum()

            # Nash inequalities
            u = A @ x
            on_support_ok = np.all(np.abs(u[J] - alpha) <= 1e-5 + 1e-5 * np.abs(alpha))
            off_support_ok = np.all(u[[i for i in range(k) if i not in J]] <= alpha + 1e-7)
            is_nash = on_support_ok and off_support_ok
            if not is_nash:
                results.append({"support": tuple(J), "x": x, "alpha": alpha,
                                "is_nash": False, "is_ess": False, "eigvals_tangent": None})
                continue

            # Strict disadvantage off support
            off_strict = True
            for i in range(k):
                if i not in J and u[i] >= alpha - 1e-8:
                    off_strict = False
                    break

            # Replicator Jacobian on tangent
            # Replicator: dot x_i = x_i( (A x)_i - x^T A x )
            bar_u = float(x @ u)
            c = (A + A.T) @ x
            Jmat = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    Jmat[i, j] = (u[i] - bar_u) * (1.0 if i == j else 0.0) + x[i] * (A[i, j] - c[j])

            if s >= 2:
                # Tangent basis within support: e_1 - e_s, ..., e_{s-1} - e_s
                B = np.zeros((k, s - 1))
                for r in range(s - 1):
                    e = np.zeros(k); e[J[r]] = 1.0; e[J[-1]] = -1.0
                    B[:, r] = e
                JT = B.T @ Jmat @ B
                eigvals = np.linalg.eigvals(JT)
            else:
                eigvals = np.array([Jmat[J[0], J[0]]])

            stable_tangent = np.all(np.real(eigvals) < -np.sqrt(tol))
            is_ess = is_nash and off_strict and stable_tangent

            results.append({"support": tuple(J), "x": x, "alpha": alpha,
                            "is_nash": True, "is_ess": bool(is_ess),
                            "eigvals_tangent": eigvals})
    return results

# ---------- Finance helpers ----------
def project_out_identities(X: np.ndarray, C: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project columns of X (N x T) onto the nullspace of constraint rows C (m x N).
    Each row of C represents a linear identity c^T x = 0 (e.g., revenue - COGS - gross_profit = 0).
    Returns X_perp = P_{ker C} X with P = I - C^T (C C^T)^+ C (stable pseudoinverse).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N x T)")
    if C.ndim != 2:
        raise ValueError("C must be 2D (m x N)")
    N = X.shape[0]
    if C.shape[1] != N:
        raise ValueError("C must have N columns matching rows of X")
    CCt = C @ C.T
    m = CCt.shape[0]
    P = np.eye(N) - C.T @ np.linalg.solve(CCt + eps * np.eye(m), C)
    return P @ X

def value_gradient_payoffs(
    X: np.ndarray,
    J_future: np.ndarray,
    model: str = "ridge",
    ridge: float = 1e-3,
    standardize: bool = True,
) -> np.ndarray:
    """
    Construct a per-player payoff series v (N x T) from a scalar future value target J_future (T,).
    - model='ridge': fits J ≈ alpha + beta^T X_t (pooled over t). Returns v(t) ≡ beta for all t.
    - standardize: if True, z-score features across time before fitting; beta is mapped back to original scale.
    Returns v of shape (N x T).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N x T)")
    N, T = X.shape
    if J_future.shape[0] != T:
        raise ValueError("J_future must have length T")
    if standardize:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        Xs = (X - mu) / sd
    else:
        mu = np.zeros((N,1))
        sd = np.ones((N,1))
        Xs = X.copy()
    X1 = np.concatenate([np.ones((T,1)), Xs.T], axis=1)  # T x (N+1)
    beta_full = np.linalg.solve(X1.T @ X1 + ridge * np.eye(N+1), X1.T @ J_future)
    alpha = beta_full[0]
    beta_s = beta_full[1:]
    beta = (beta_s / sd.ravel())
    v = np.tile(beta.reshape(N,1), (1,T))
    return v

# ---------- Bio / general helpers ----------
def growth_payoffs(counts: np.ndarray, dt: float = 1.0, pad: str = "edge") -> np.ndarray:
    """
    Per-capita growth payoffs from count/abundance data.
    counts: N x T nonnegative (prefer >0; we add 1e-12 for safety)
    returns v: N x T with v_i(t) ≈ (log n_i(t+1) - log n_i(t)) / dt, padded to length T.
    pad: 'edge' repeats last valid column; 'zero' pads zeros.
    """
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (N x T)")
    N, T = counts.shape
    eps = 1e-12
    logc = np.log(counts + eps)
    if T == 1:
        return np.zeros_like(counts, dtype=float)
    g = (logc[:, 1:] - logc[:, :-1]) / float(dt)  # N x (T-1)
    if pad == "edge":
        last = g[:, -1:]
    elif pad == "zero":
        last = np.zeros((N, 1))
    else:
        raise ValueError("pad must be 'edge' or 'zero'")
    return np.concatenate([g, last], axis=1)

def info_gain_payoffs(
    X: np.ndarray,
    y_next: np.ndarray,
    model: str = "ridge",
    ridge: float = 1e-3,
    window: int | None = None,
) -> np.ndarray:
    """
    Per-channel information-gain payoffs via ablation for a next-step prediction target.
    X: N x T feature/share matrix at time t
    y_next: length T target aligned to X_t (e.g., y at horizon H), numeric
    model='ridge': fits y ≈ alpha + beta^T X_t over a window (or globally if window=None)
    ridge: L2 regularization for stability
    window: if None, compute a single global LOFO (leave-one-feature-out) payoff per channel (constant over time).
            if int >= max(8, N+2), compute rolling LOFO on each time t using last `window` samples.

    Returns v: N x T where v_i(t) = MSE_without_i - MSE_full on the fit domain (nonnegative means feature i is helpful).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (N x T)")
    N, T = X.shape
    if y_next.shape[0] != T:
        raise ValueError("y_next must have length T")

    def ridge_mse(Xt, yt):
        Tn, d = Xt.shape
        X1 = np.concatenate([np.ones((Tn,1)), Xt], axis=1)
        A = X1.T @ X1 + ridge * np.eye(d+1)
        b = X1.T @ yt
        w = np.linalg.solve(A, b)
        resid = yt - X1 @ w
        return float(np.mean(resid**2))

    Xt = X.T
    if window is None:
        mse_full = ridge_mse(Xt, y_next)
        v = np.zeros((N, T))
        for i in range(N):
            mse_wo = ridge_mse(np.delete(Xt, i, axis=1), y_next)
            v[i, :] = (mse_wo - mse_full)
        return v

    W = int(window)
    if W < max(8, N+2):
        raise ValueError(f"window must be >= max(8, N+2) (got {W})")
    v = np.zeros((N, T))
    for t in range(T):
        a = max(0, t - W + 1); b = t + 1
        Xw = Xt[a:b, :]; yw = y_next[a:b]
        mse_full = ridge_mse(Xw, yw)
        for i in range(N):
            mse_wo = ridge_mse(np.delete(Xw, i, axis=1), yw)
            v[i, t] = (mse_wo - mse_full)
    return v


# ---------- Matrix factorization helpers ----------
def nmf_on_X(
    X: np.ndarray,
    k: int,
    iters: int = 400,
    seed: int | None = 0,
    normalize: str = "l2",
    eps: float = 1e-12,
    W0: np.ndarray | None = None,
    H0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Nonnegative Matrix Factorization (multiplicative updates) on nonnegative X (N x T).
    # Returns (W, H) with shapes (N x k), (k x T). Columns of W normalized if normalize != 'none'.
    if X.ndim != 2:
        raise ValueError("X must be 2D (N x T)")
    V = np.maximum(X.astype(float), 0.0)
    N, T = V.shape
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    W = np.maximum(np.array(W0) if W0 is not None else rng.random((N, k)), eps)
    H = np.maximum(np.array(H0) if H0 is not None else rng.random((k, T)), eps)
    ones_V = np.ones_like(V)
    for _ in range(max(1, iters)):
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.T @ ones_V + eps)
        H = np.maximum(H, eps)
        WH = W @ H + eps
        W *= ((V / WH) @ H.T) / (ones_V @ H.T + eps)
        W = np.maximum(W, eps)
        if normalize != "none":
            if normalize == "l2":
                col = np.sqrt((W**2).sum(axis=0, keepdims=True)) + eps
            elif normalize == "l1":
                col = np.sum(np.abs(W), axis=0, keepdims=True) + eps
            else:
                raise ValueError("normalize must be 'l2', 'l1', or 'none'")
            W /= col; H *= col.T
    return W, H

# ---------- Optional demo when run as a script ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, T = 4, 400
    t = np.arange(T)
    common = 0.5 * np.sin(2*np.pi*t/50)
    contrasts = np.vstack([
        0.8*np.sin(2*np.pi*(t+10)/30),
        0.5*np.cos(2*np.pi*(t+5)/40),
        -0.3*np.sin(2*np.pi*(t+3)/60),
        0.2*np.cos(2*np.pi*(t+7)/25),
    ])
    X = (common + contrasts + 0.2*rng.standard_normal((N, T)))

    # Option A quick run
    resA = static_game_from_series(X, lag=1, bins="tercile", zero_sum_discretization=True)
    print("Option A: profiles:", len(resA.payoffs))

    # Placeholder S from rectified PCA
    U, svals, Vt = np.linalg.svd(X - X.mean(axis=1, keepdims=True), full_matrices=False)
    k = 3
    S = np.abs(U[:, :k])
    v = X  # placeholder payoff

    est = estimate_A_from_series(S, X, v, k=k, ridge=1e-3)
    A = est["A"]
    print("A shape:", A.shape, "R2:", round(est["R2"], 3))

    ess_list = find_ESS(A, tol=1e-8, max_support=3)
    n_ess = sum(1 for r in ess_list if r["is_ess"])
    n_nash = sum(1 for r in ess_list if r["is_nash"])
    print("Nash candidates:", n_nash, "ESS found:", n_ess)
