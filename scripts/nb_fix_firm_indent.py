#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/FIRM_DEMO.ipynb")

RE_INNER_EST = re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+estimate_A_from_series\s*$", re.M)
RE_DUP_NPPLT = re.compile(r"^\s*import\s+numpy\s+as\s+np,\s*matplotlib\.pyplot\s+as\s+plt\s*$", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src0 = c.get("source","") or ""
        if "def rolling_window_metrics" in src0:
            src = RE_INNER_EST.sub("", src0)  # remove stray inner import
            src = RE_DUP_NPPLT.sub("# (removed duplicate) numpy/matplotlib already imported", src)
            # normalize blank lines (collapse 3+ newlines)
            src = re.sub(r"\n{3,}", "\n\n", src)
            if src != src0:
                c["source"] = src
                changed = True
    if changed:
        nbf.write(nb, NB)
        print("[patched] FIRM_DEMO: removed stray inner import and duplicate np/plt line")
    else:
        print("[ok] FIRM_DEMO: no changes")
if __name__ == "__main__":
    main()
