import os, sys
# Make src/ importable when not installed yet
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src  = os.path.join(_repo, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

import ts2eg as gm
from ts2eg import extensions as ext
import numpy as np

def test_unweighted_projectors_idempotent_and_sum_to_I():
    N = 7
    M_I, M_Z = gm.projectors(N)
    I = np.eye(N)
    assert np.allclose(M_I @ M_I, M_I, atol=1e-12)
    assert np.allclose(M_Z @ M_Z, M_Z, atol=1e-12)
    assert np.allclose(M_I + M_Z, I, atol=1e-12)
    # Mean vector is fixed by M_I, killed by M_Z
    one = np.ones((N,1))
    assert np.allclose(M_I @ one, one, atol=1e-12)
    assert np.allclose(M_Z @ one, np.zeros((N,1)), atol=1e-12)
    # Symmetry (orthogonal projector under standard inner product)
    assert np.allclose(M_I.T, M_I, atol=1e-12)
    assert np.allclose(M_Z.T, M_Z, atol=1e-12)

def test_weighted_projectors_properties():
    w = np.array([0.5, 0.7, 1.0, 1.2, 0.9, 0.8, 0.6])
    M_Iw, M_Zw = ext.weighted_projectors(w)
    N = w.size
    I = np.eye(N)
    # Idempotent and complementary
    assert np.allclose(M_Iw @ M_Iw, M_Iw, atol=1e-12)
    assert np.allclose(M_Zw @ M_Zw, M_Zw, atol=1e-12)
    assert np.allclose(M_Iw + M_Zw, I, atol=1e-12)
    # Span{1} fixed; W-orthogonality: 1^T W M_Zw = 0
    one = np.ones((N,1))
    assert np.allclose(M_Iw @ one, one, atol=1e-12)
    assert np.allclose((w.reshape(1,-1) @ (M_Zw)), np.zeros((1,N)), atol=1e-12)
