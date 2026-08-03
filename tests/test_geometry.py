import numpy as np
import pytest

from ts2eg.geometry import (
    canonicalize_payoff_operator,
    centering_projector,
    replicator_field,
)


def _seeded_game_and_mixture():
    rng = np.random.default_rng(20250803)
    A = rng.standard_normal((4, 4))
    x = rng.dirichlet(np.ones(4))
    return A, x


def test_replicator_field_is_invariant_under_payoff_equivalence():
    A, x = _seeded_game_and_mixture()
    c = np.array([0.7, -1.2, 0.3, 2.1])
    equivalent = A + np.ones((4, 1)) @ c.reshape(1, -1)

    assert np.allclose(replicator_field(A, x), replicator_field(equivalent, x))


def test_left_canonicalization_preserves_replicator_field():
    A, x = _seeded_game_and_mixture()
    canonical = canonicalize_payoff_operator(A)

    assert np.allclose(replicator_field(A, x), replicator_field(canonical, x))
    assert np.allclose(np.ones(4) @ canonical, 0.0, atol=1e-12)


def test_right_centering_changes_replicator_field_for_seeded_game():
    A, x = _seeded_game_and_mixture()
    right_centered = A @ centering_projector(4)

    difference = replicator_field(A, x) - replicator_field(right_centered, x)
    assert np.linalg.norm(difference) > 1e-6


def test_replicator_field_is_equivariant_under_strategy_permutation():
    A, x = _seeded_game_and_mixture()
    permutation = np.array([2, 0, 3, 1])
    P = np.eye(4)[permutation]

    permuted_field = replicator_field(P @ A @ P.T, P @ x)
    assert np.allclose(permuted_field, P @ replicator_field(A, x))


def test_sign_reversal_is_not_a_payoff_equivalence():
    A, x = _seeded_game_and_mixture()
    field = replicator_field(A, x)
    reversed_field = replicator_field(-A, x)

    assert np.linalg.norm(field) > 1e-6
    assert np.allclose(reversed_field, -field)
    assert not np.allclose(reversed_field, field)


@pytest.mark.parametrize("k", [0, -1, 2.5, True])
def test_centering_projector_rejects_invalid_dimensions(k):
    with pytest.raises(ValueError, match="positive integer"):
        centering_projector(k)


def test_geometry_rejects_invalid_operator_and_mixture_shapes():
    with pytest.raises(ValueError, match="square"):
        canonicalize_payoff_operator(np.ones((2, 3)))
    with pytest.raises(ValueError, match="probability vector"):
        replicator_field(np.eye(2), np.array([0.2, 0.2]))
