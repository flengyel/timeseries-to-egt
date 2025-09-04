#!/usr/bin/env python
from pathlib import Path
import nbformat as nbf, re

NB = Path("notebooks/SOCIO_DEMO.ipynb")
HEAD = re.compile(r"^\s*#\s*Ensure strategy basis S exists", re.M)

NEW = """# Ensure strategy basis S exists before estimation (and ensure X first)
import numpy as _np
if 'X' not in globals():
    if 'X_share' in globals():
        X = _np.asarray(X_share, dtype=float)
    elif 'counts' in globals():
        X = _np.asarray(counts, dtype=float)
    elif 'v_growth' in globals():
        X = _np.asarray(v_growth, dtype=float)
    else:
        raise NameError("X is undefined; expected X_share, counts, or v_growth earlier in the notebook.")

try:
    _ = S  # noqa: F821
except NameError:
    K = int(globals().get('K', 3)) if 'K' in globals() else 3
    S, H = nmf_on_X(X, k=K, iters=50, seed=1, normalize='l2')

print('S shape:', S.shape)
"""

def main():
    nb = nbf.read(NB, as_version=4)
    replaced = 0
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = (c.get("source") or "")
        first = src.splitlines()[0].strip() if src.strip() else ""
        if first.startswith("#") and HEAD.match(first):
            c["source"] = NEW
            replaced += 1
    if replaced == 0:
        raise SystemExit("[abort] No matching basis cells found")
    with NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[patched] SOCIO_DEMO: replaced {replaced} basis cell(s)")

if __name__ == "__main__":
    main()
