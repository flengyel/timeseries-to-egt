#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/FINANCE_DEMO.ipynb")

# Match the legacy "canonical imports with fallbacks" block
RE_CANON_BLOCK = re.compile(
    r"(?s)#\s*---\s*Canonical imports with fallbacks\s*---.*?from ts2eg import value_gradient_payoffs", re.M
)
# Also handle the case where only the last line is present
RE_VALGRAD_BAD = re.compile(r"^\s*from\s+ts2eg\s+import\s+value_gradient_payoffs\s*$", re.M)

CANON_FIXED = """# --- Canonical imports (ts2eg only) ---
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ts2eg as gm
from ts2eg.core import nmf_on_X, estimate_A_from_series, find_ESS
# value-gradient payoffs: import from core, else define minimal ridge version
try:
    from ts2eg.core import value_gradient_payoffs
except Exception:
    import numpy as _np
    def value_gradient_payoffs(X, J, model='ridge', ridge=1e-2, standardize=True):
        X = _np.asarray(X, dtype=float); J = _np.asarray(J, dtype=float)
        if X.ndim != 2: raise ValueError("X must be 2D (N x T)")
        N, T = X.shape
        Xc = X.copy()
        if standardize:
            Xc = (Xc - Xc.mean(axis=1, keepdims=True)) / (_np.std(Xc, axis=1, keepdims=True) + 1e-12)
            Jc = (J - J.mean()) / (_np.std(J) + 1e-12)
        else:
            Jc = J
        G = Xc @ Xc.T  # N x N
        beta = _np.linalg.solve(G + ridge * _np.eye(N), Xc @ Jc)  # N
        return beta[:, None] * _np.ones((1, T))  # N x T
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None  # optional
"""

RE_LAMBDA = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")

# Replace legacy nmf_multiplicative(...) -> nmf_on_X(...), drop manual normalization
RE_NMF_CALL = re.compile(r"nmf_multiplicative\s*\(", re.M)
RE_NORM_LINE = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "") or ""

        # 1) Replace the whole legacy canonical block if present
        if RE_CANON_BLOCK.search(src):
            src = RE_CANON_BLOCK.sub(CANON_FIXED, src)
            changed = True
        # 2) Or just fix a lone "from ts2eg import value_gradient_payoffs"
        src2 = RE_VALGRAD_BAD.sub(CANON_FIXED, src)
        if src2 != src:
            src = src2
            changed = True

        # 3) lambda_ -> ridge
        src3 = RE_LAMBDA.sub("ridge=", src)
        if src3 != src:
            src = src3
            changed = True

        # 4) nmf_multiplicative -> nmf_on_X (and ensure normalize='l2')
        if RE_NMF_CALL.search(src):
            src = RE_NMF_CALL.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")
            if "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)
            changed = True

        # 5) drop redundant manual normalization of S_null
        if RE_NORM_LINE.search(src):
            src = RE_NORM_LINE.sub("# (removed) nmf_on_X already normalizes S columns", src)
            changed = True

        if changed:
            c["source"] = src

    # Set CI tag if missing
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    if changed:
        nbf.write(nb, NB)
        print("[patched] FINANCE_DEMO imports/kw/NMF updated")
    else:
        print("[ok] FINANCE_DEMO: no change needed")

if __name__ == "__main__":
    main()
