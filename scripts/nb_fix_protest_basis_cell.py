#!/usr/bin/env python
from pathlib import Path
import re, nbformat as nbf

NB = Path("notebooks/PROTEST_DEMO.ipynb")
HEAD = re.compile(r"^\s*#\s*Ensure strategy basis S exists", re.M)

NEW = """# Ensure strategy basis S exists before estimation (and ensure X first)
import numpy as _np
if 'X' not in globals():
    if 'X_share' in globals():
        X = _np.asarray(X_share, dtype=float)
    elif 'v_growth' in globals():
        X = _np.asarray(v_growth, dtype=float)
    elif 'counts' in globals():
        X = _np.asarray(counts, dtype=float)
    else:
        raise NameError("X is undefined; expected X_share, v_growth, or counts earlier in the notebook.")

try:
    _ = S  # noqa: F821
except NameError:
    K = int(globals().get('K', 3)) if 'K' in globals() else 3
    S, H = nmf_on_X(X, k=K, iters=50, seed=1, normalize='l2')
"""

nb = nbf.read(NB, as_version=4)
patched = False
for c in nb.cells:
    if c.get("cell_type") != "code":
        continue
    src = c.get("source","") or ""
    if HEAD.search(src):
        c["source"] = NEW
        patched = True
        break

if not patched:
    raise SystemExit("[abort] Target cell not found")

with NB.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("[patched] PROTEST_DEMO: fixed basis/ensure-X cell")
