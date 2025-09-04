#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/FINANCE_DEMO.ipynb")

RE_DEF_NMF = re.compile(r"^\s*def\s+nmf_on_X\s*\(", re.M)
RE_NORM_LINE = re.compile(r"^\s*S\s*=\s*S\s*/.*np\.linalg\.norm\(S.*$", re.M)
RE_CALL_NMF = re.compile(r"nmf_on_X\s*\(", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    # (a) remove/neutralize local nmf_on_X def cell
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source","") or ""
        if RE_DEF_NMF.search(src):
            c["source"] = "# (removed) local nmf_on_X shadowed core; using ts2eg.core.nmf_on_X"
            changed = True

    # (b) drop redundant manual normalization of S after nmf_on_X
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source","") or ""
        new = RE_NORM_LINE.sub("# (removed) nmf_on_X already normalizes columns (normalize='l2')", src)
        if new != src:
            c["source"] = new
            changed = True

    # (c) ensure normalize='l2' is passed in first nmf_on_X call that lacks it
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source","") or ""
        if "nmf_on_X(" in src and "normalize=" not in src:
            src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)
            c["source"] = src
            changed = True
            break

    if changed:
        nbf.write(nb, NB)
        print("[patched] FINANCE_DEMO: removed local nmf_on_X, dropped manual norm, ensured normalize='l2'")
    else:
        print("[ok] FINANCE_DEMO: no changes")

if __name__ == "__main__":
    main()
