import os, sys, numpy as np
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); _src = os.path.join(_repo, "src")
if os.path.isdir(_src) and _src not in sys.path: sys.path.insert(0, _src)
import ts2eg as gm
from ts2eg import extensions as ext

def test_helmert_orthonormal_and_first_column():
    N=8; Q=gm.helmert_Q(N)
    assert np.allclose(Q.T @ Q, np.eye(N), atol=1e-12)
    assert np.allclose(Q[:,0], np.ones(N)/np.sqrt(N), atol=1e-12)

def test_A_left_canonicalization_enforced():
    rng=np.random.default_rng(0); N,T,k=6,120,3
    X=rng.standard_normal((N,T)); S=np.abs(rng.standard_normal((N,k)))
    Xs=(X-X.mean(axis=1,keepdims=True))/(X.std(axis=1,keepdims=True)+1e-12)
    est=gm.estimate_A_from_series(S, Xs, Xs, k=k, ridge=1e-3)
    A=est["A"]
    assert np.allclose(A, gm.canonicalize_payoff_operator(est["A_raw"]), atol=1e-12)
    assert np.allclose(np.ones(k)@A, 0.0, atol=1e-8)

def test_static_game_outputs_have_MI_MZ():
    rng=np.random.default_rng(1); N,T=4,200
    X=rng.standard_normal((N,T))
    res=gm.static_game_from_series(X, lag=1, bins="tercile", zero_sum_discretization=True)
    a=next(iter(res.payoffs.keys()))
    assert a in res.payoffs_I and a in res.payoffs_Z
