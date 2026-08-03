import numpy as np

import ts2eg as gm


def _record_for_support(records, support):
    matches = [record for record in records if record["support"] == tuple(support)]
    assert len(matches) == 1
    return matches[0]


def test_strict_pure_ess_has_vacuous_tangent_and_transverse_stability():
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    record = _record_for_support(gm.find_ESS(A, tol=1e-9), (0,))

    assert record["is_nash"]
    assert record["ess_status"] == "certified"
    assert record["is_ess"]
    assert record["tangent_basis"].shape == (1, 0)
    assert record["ess_tangent_eigenvalues"].size == 0
    assert record["jacobian_tangent_eigenvalues"].size == 0
    assert np.all(record["transverse_invasion_eigenvalues"] < 0.0)
    assert record["stability_status"] == "certified"
    assert record["is_locally_asymptotically_stable"] is True


def test_equilibrium_diagnostics_are_payoff_equivalence_invariant():
    A = np.array([[0.0, 3.0], [2.0, 0.0]])
    c = np.array([1.0e6, -7.0e5])
    equivalent = A + np.ones((2, 1)) @ c.reshape(1, -1)

    original = {record["support"]: record for record in gm.find_ESS(A, tol=1e-9)}
    shifted = {record["support"]: record for record in gm.find_ESS(equivalent, tol=1e-9)}

    assert original.keys() == shifted.keys()
    for support in original:
        left, right = original[support], shifted[support]
        assert np.allclose(left["x"], right["x"])
        assert left["is_nash"] == right["is_nash"]
        assert left["ess_status"] == right["ess_status"]
        assert left["is_ess"] == right["is_ess"]
        assert left["stability_status"] == right["stability_status"]
        assert left["is_locally_asymptotically_stable"] == right[
            "is_locally_asymptotically_stable"
        ]
        assert np.allclose(
            left["ess_tangent_eigenvalues"], right["ess_tangent_eigenvalues"]
        )
        assert np.allclose(
            left["jacobian_tangent_eigenvalues"],
            right["jacobian_tangent_eigenvalues"],
        )
        assert np.allclose(
            left["transverse_invasion_eigenvalues"],
            right["transverse_invasion_eigenvalues"],
        )


def test_strict_pure_ess_certificate_is_payoff_equivalence_invariant():
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    c = np.array([-3.0e5, 8.0e5])
    equivalent = A + np.ones((2, 1)) @ c.reshape(1, -1)

    original = _record_for_support(gm.find_ESS(A, tol=1e-9), (0,))
    shifted = _record_for_support(gm.find_ESS(equivalent, tol=1e-9), (0,))

    assert original["is_nash"] and shifted["is_nash"]
    assert original["ess_status"] == shifted["ess_status"] == "certified"
    assert original["stability_status"] == shifted["stability_status"] == "certified"
    assert np.allclose(
        original["transverse_invasion_eigenvalues"],
        shifted["transverse_invasion_eigenvalues"],
    )


def test_pure_off_support_tie_is_undetermined():
    A = np.array([[1.0, 0.0], [1.0, 0.0]])
    record = _record_for_support(gm.find_ESS(A, tol=1e-9), (0,))

    assert record["is_nash"]
    assert record["ess_status"] == "undetermined"
    assert not record["is_ess"]
    assert np.allclose(record["transverse_invasion_eigenvalues"], 0.0)
    assert record["stability_status"] == "undetermined"
    assert record["is_locally_asymptotically_stable"] is None


def test_neutral_rock_paper_scissors_is_nash_but_not_ess_certified():
    A = np.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    record = _record_for_support(gm.find_ESS(A, tol=1e-9), (0, 1, 2))

    assert record["is_nash"]
    assert record["ess_status"] == "rejected"
    assert not record["is_ess"]
    assert np.allclose(record["ess_tangent_eigenvalues"], 0.0, atol=1e-12)
    assert record["stability_status"] == "undetermined"
    assert record["is_locally_asymptotically_stable"] is None

    Q = record["tangent_basis"]
    assert Q.shape == (3, 2)
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-12)
    assert np.allclose(np.ones(3) @ Q, 0.0, atol=1e-12)


def test_hawk_dove_mixed_equilibrium_is_ess_certified():
    A = np.array([[0.0, 3.0], [2.0, 0.0]])
    record = _record_for_support(gm.find_ESS(A, tol=1e-9), (0, 1))

    assert record["is_nash"]
    assert record["ess_status"] == "certified"
    assert record["is_ess"]
    assert np.all(record["ess_tangent_eigenvalues"] < 0.0)
    assert record["stability_status"] == "certified"
    assert record["is_locally_asymptotically_stable"] is True
