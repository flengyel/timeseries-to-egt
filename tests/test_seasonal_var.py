import os, sys
# Make src/ importable when not installed yet
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src  = os.path.join(_repo, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

import ts2eg as gm
from ts2eg import extensions as ext
import numpy as np

def synthetic_series(N=5, T=360, seed=123):
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
    X = (season + common + contrasts + 0.15*rng.standard_normal((N, T)) )
    return X

def test_seasonal_var_shapes_and_profiles():
    X = synthetic_series()
    res = ext.var_information_sharing_game_seasonal(
        X, p=2, ridge=1e-3, seasonal_period=12, seasonal_multiples=[1,2],
        include_self_always=True, lambda_common=0.0, weights=None
    )
    # 2 base lags + 2 seasonal lags -> same profile space size as base
    assert isinstance(res.payoffs, dict)
    assert res.meta['p'] == 2
    assert res.meta['seasonal_period'] == 12
    assert res.meta['seasonal_multiples'] == [1,2]
    # Should cover all 2^N profiles for N<=12 (here N=5 -> 32)
    assert res.meta['profiles'] == 32

def test_seasonal_equals_base_when_no_seasonal_and_no_weights():
    X = synthetic_series()
    base = gm.var_information_sharing_game(X, p=2, ridge=1e-3)
    extn = ext.var_information_sharing_game_seasonal(
        X, p=2, ridge=1e-3, seasonal_period=0, seasonal_multiples=None, weights=None
    )
    # With same RNG seeds omitted, payoffs may differ slightly only via numerical noise; here should match exactly.
    assert base.meta['profiles'] == extn.meta['profiles']
    # Compare a few random profiles for equality
    keys = list(base.payoffs.keys())[:5]
    for k in keys:
        assert np.allclose(base.payoffs[k], extn.payoffs[k], atol=1e-10)
