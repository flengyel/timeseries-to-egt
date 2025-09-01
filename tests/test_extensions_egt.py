import os, sys
# Make src/ importable when not installed yet
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src  = os.path.join(_repo, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

import ts2eg as gm
from ts2eg import extensions as ext
import numpy as np

def synthetic_series(N=5, T=360, seed=321):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    season = 0.6*np.sin(2*np.pi*t/12)
    common = 0.3*np.sin(2*np.pi*t/60)
    contrasts = np.vstack([
        0.8*np.sin(2*np.pi*(t+5)/30),
        0.4*np.cos(2*np.pi*(t+3)/40),
        -0.5*np.sin(2*np.pi*(t+11)/48),
        0.2*np.cos(2*np.pi*(t+7)/36),
        -0.3*np.sin(2*np.pi*(t+2)/24),
    ])[:N]
    return (season + common + contrasts + 0.1*rng.standard_normal((N, T)) )

def test_weighted_A_row_centering_and_R2():
    X = synthetic_series()
    N, T = X.shape
    # Build S via base NMF on [0,1]-scaled X
    X0 = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-9)
    S, H = gm.nmf_on_X(X0, k=3, iters=150, seed=2, normalize="l2")
    K = S.shape[1]
    # Standardize X for estimation; use it as payoff field for smoke test
    X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    w = np.array([0.5, 0.8, 1.0, 0.9, 0.7])[:N]
    est = ext.estimate_A_from_series_weighted(S, X_std, X_std, k=K, lambda_=1e-3, weights=w)
    A = est["A"]
    # Row sums approximately zero (replicator invariance)
    assert np.allclose(A.sum(axis=0), 0.0, atol=1e-8)
    # Reasonable fit metric
    assert 0.0 <= est["R2"] <= 1.0

def test_iaaft_preserves_sorted_values_and_spectrum():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    y = ext.iaaft(x, n_iter=50, rng=rng)
    # Sorted values identical (rank-ordered mapping enforces marginal)
    assert np.allclose(np.sort(x), np.sort(y), atol=1e-12)
    # Fourier amplitude close
    Xf = np.abs(np.fft.rfft(x))
    Yf = np.abs(np.fft.rfft(y))
    rel_err = np.linalg.norm(Xf - Yf) / (np.linalg.norm(Xf) + 1e-12)
    assert rel_err < 1e-2

def test_surrogate_harness_bounds():
    X = synthetic_series()
    X_std = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    S, H = gm.nmf_on_X((X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-9),
                        k=3, iters=120, seed=3, normalize="l2")
    w = np.array([0.6, 0.9, 1.0, 0.8, 0.7])[:X.shape[0]]
    summ = ext.surrogate_ess_frequency(S, X_std, X_std, k=S.shape[1], lambda_=1e-3,
                                       num_surrogates=3, iaaft_iters=30, weights=w, seed=42)
    assert 0.0 <= summ["ess_rate"] <= 1.0
    assert 0.0 <= summ["nash_rate"] <= 1.0
    assert 0.0 <= summ["R2_mean"] <= 1.0
    assert summ["num_surrogates"] == 3

def test_find_ESS_known_games():
    # Rock-Paper-Scissors: mixed Nash, not ESS (neutrally stable)
    A_rps = np.array([[0, 1, -1],
                      [-1, 0, 1],
                      [1, -1, 0]], dtype=float)
    res = gm.find_ESS(A_rps, tol=1e-9, max_support=3)
    # There should be a Nash with full support but is_ess False
    full = [r for r in res if set(r['support']) == {0,1,2} and r['is_nash']]
    assert len(full) >= 1
    assert not any(r['is_ess'] for r in full)

    # Hawk-Dove (anti-coordination): mixed ESS
    A_hd = np.array([[0, 3],
                     [2, 0]], dtype=float)
    res2 = gm.find_ESS(A_hd, tol=1e-9, max_support=2)
    mixed = [r for r in res2 if set(r['support']) == {0,1} and r['is_nash']]
    assert len(mixed) >= 1
    assert any(r['is_ess'] for r in mixed)
