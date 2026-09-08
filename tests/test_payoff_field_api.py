import os
import sys

import numpy as np
import pytest


REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_ROOT = os.path.join(REPOSITORY_ROOT, "src")
if os.path.isdir(SOURCE_ROOT) and SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

import ts2eg as gm
from ts2eg import extensions as ext


def _payoff_field_inputs():
    rng = np.random.default_rng(20260908)
    S = rng.random((4, 3))
    X = rng.random((4, 36))
    v = rng.standard_normal((4, 36))
    return S, X, v


@pytest.mark.parametrize(
    ("estimator", "extra"),
    [
        (gm.estimate_A_from_series, {}),
        (ext.estimate_A_from_series_weighted, {"weights": np.array([1.0, 2.0, 3.0, 4.0])}),
    ],
)
def test_payoff_field_score_is_in_sample_zero_field_improvement(estimator, extra):
    S, X, v = _payoff_field_inputs()
    est = estimator(S, X, v, k=3, ridge=1e-3, **extra)

    expected_residual = est["Gc"] - est["A"] @ est["Xk"]
    expected_ss_res = float(np.sum(expected_residual**2))
    expected_ss_tot = float(np.sum(est["Gc"] ** 2))

    assert np.isclose(est["ss_res"], expected_ss_res)
    assert np.isclose(est["ss_tot"], expected_ss_tot)
    assert np.isclose(est["R2"], 1.0 - expected_ss_res / expected_ss_tot)

    common_payoff = np.ones((4, 1)) @ np.linspace(-2.0, 3.0, X.shape[1]).reshape(1, -1)
    shifted = estimator(S, X, v + common_payoff, k=3, ridge=1e-3, **extra)
    assert np.allclose(shifted["Gc"], est["Gc"])
    assert np.allclose(shifted["A"], est["A"])


@pytest.mark.parametrize(
    ("estimator", "extra"),
    [
        (gm.estimate_A_from_series, {}),
        (ext.estimate_A_from_series_weighted, {"weights": np.array([1.0, 2.0, 3.0])}),
    ],
)
def test_zero_payoff_field_has_undefined_R2(estimator, extra):
    S = np.eye(3)
    X = np.array(
        [
            [0.8, 0.2, 0.1, 0.4],
            [0.1, 0.7, 0.2, 0.3],
            [0.1, 0.1, 0.7, 0.3],
        ]
    )
    est = estimator(S, X, np.zeros_like(X), k=3, ridge=1e-3, **extra)

    assert est["ss_res"] == 0.0
    assert est["ss_tot"] == 0.0
    assert np.isnan(est["R2"])


def test_value_gradient_payoffs_are_full_sample_and_time_constant():
    rng = np.random.default_rng(17)
    X = rng.standard_normal((4, 40))
    target = 0.6 * X[0] - 0.25 * X[2] + rng.normal(scale=0.05, size=40)

    payoffs = gm.value_gradient_payoffs(X, target, model="ridge", ridge=1e-3)

    assert payoffs.shape == X.shape
    assert np.allclose(payoffs, payoffs[:, [0]])
    for row in payoffs:
        assert np.array_equal(np.roll(row, 7), row)


def test_value_gradient_payoffs_reject_unsupported_models_and_nonvector_targets():
    X = np.arange(24.0).reshape(3, 8)
    target = np.arange(8.0)

    with pytest.raises(ValueError, match="model must be 'ridge'"):
        gm.value_gradient_payoffs(X, target, model="ols")
    with pytest.raises(ValueError, match="one-dimensional"):
        gm.value_gradient_payoffs(X, target[:, None])


@pytest.mark.parametrize(
    "estimator",
    [
        gm.estimate_A_from_series,
        ext.estimate_A_from_series_weighted,
        ext.surrogate_ess_frequency,
    ],
)
def test_estimators_do_not_accept_retired_lambda_keyword(estimator):
    S = np.eye(2)
    X = np.array([[0.8, 0.4], [0.2, 0.6]])

    with pytest.raises(TypeError, match="lambda_"):
        estimator(S, X, X, k=2, lambda_=1e-3)
