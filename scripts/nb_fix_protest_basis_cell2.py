#!/usr/bin/env python
from pathlib import Path
import nbformat as nbf, re

NB = Path("notebooks/PROTEST_DEMO.ipynb")

HEADER_LINE = re.compile(r"^\s*#\s*Ensure strategy basis S exists", re.M)

NEW_SRC = """# Ensure strategy basis S exists before estimation (and ensure X first)
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
        src = c.get("source","") or ""
        # Replace any cell whose FIRST non-empty line matches the header
        first_line = src.splitlines()[0].strip() if src.strip() else ""
        if first_line.startswith("#") and HEADER_LINE.match(first_line):
            c["source"] = NEW_SRC
            replaced += 1
    if replaced == 0:
        raise SystemExit("[abort] No matching basis cells found")
    with NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[patched] PROTEST_DEMO: replaced {replaced} basis cell(s)")

if __name__ == "__main__":
    main()
