#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/CYBER_DEMO.ipynb")

CALL_PAT = re.compile(r"nmf_multiplicative\s*\(", re.M)
NORM_LINE_PAT = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""

        # replace nmf_multiplicative(...) -> nmf_on_X(...), r= -> k=, ensure normalize='l2'
        if CALL_PAT.search(src):
            src = CALL_PAT.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")  # kw rename
            if "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)
            changed = True

        # remove manual normalization line after S_null computation
        if NORM_LINE_PAT.search(src):
            src = NORM_LINE_PAT.sub("# (removed) nmf_on_X already normalizes S columns", src)
            changed = True

        if changed:
            c["source"] = src

    if changed:
        nbf.write(nb, NB)
        print("[patched] CYBER_DEMO: nmf_multiplicative -> nmf_on_X; dropped manual normalization")
    else:
        print("[ok] CYBER_DEMO: no legacy nmf_multiplicative found")

if __name__ == "__main__":
    main()
