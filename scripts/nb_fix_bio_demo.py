#!/usr/bin/env python
from __future__ import annotations
import re, sys
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/BIO_DEMO.ipynb")

# remove the "ensure X exists" block from the bootstrap cell (cell 0)
RM_X_BLOCK = re.compile(r"\n#\s*ensure X exists.*", re.S)

# insert an explicit X definition *after* v_growth is computed
X_FROM_COUNTS = """# CI: define X explicitly from counts to keep shapes consistent
import numpy as np
X = np.asarray(counts, dtype=float)
"""

def main():
    nb = nbf.read(NB, as_version=4)

    # --- 1) edit cell 0: drop the premature X assignment in bootstrap ---
    c0 = nb.cells[0]
    if c0.get("cell_type") == "code":
        src = c0.get("source","")
        new = RM_X_BLOCK.sub("", src)
        if new != src:
            c0["source"] = new

    # --- 2) insert X := counts right after v_growth cell ---
    # find first code cell that defines v_growth = ...
    idx_v = None
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        src = c.get("source","")
        if re.search(r"^\s*v_growth\s*=", src, re.M):
            idx_v = i
            break
    if idx_v is None:
        print("Could not locate v_growth cell; aborting.", file=sys.stderr)
        sys.exit(1)

    # avoid duplicating if an identical X-from-counts cell already exists next
    already = False
    if idx_v + 1 < len(nb.cells):
        nxt = nb.cells[idx_v + 1]
        if nxt.get("cell_type") == "code" and "X = np.asarray(counts" in (nxt.get("source","") or ""):
            already = True

    if not already:
        xcell = nbf.v4.new_code_cell(X_FROM_COUNTS, metadata={"tags": ["ci-x-from-counts"]})
        nb.cells.insert(idx_v + 1, xcell)

    nbf.write(nb, NB)
    print("[patched] BIO_DEMO.ipynb: removed early X; added X:=counts after v_growth")

if __name__ == "__main__":
    main()
