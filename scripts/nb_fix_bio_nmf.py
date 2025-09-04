#!/usr/bin/env python
from __future__ import annotations
import re, sys
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/BIO_DEMO.ipynb")

CALL_PAT = re.compile(r"nmf_multiplicative\s*\(", re.M)
NORM_LINE_PAT = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "") or ""

        # Replace nmf_multiplicative(...) call with nmf_on_X(...)
        if CALL_PAT.search(src):
            src = CALL_PAT.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")  # r->k kwarg
            # ensure normalize is passed (idempotent if already present)
            # crude: if this call assigns to S_null, tack on normalize='l2' if missing
            if "S_null" in src and "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)",
                             r"nmf_on_X(\1, normalize='l2')", src, count=1)
            changed = True

        # Drop manual normalization after S_null computation
        if NORM_LINE_PAT.search(src):
            src = NORM_LINE_PAT.sub("# (removed) nmf_on_X already normalizes S columns", src)
            changed = True

        if changed:
            c["source"] = src

    if not changed:
        print("[ok] BIO_DEMO: no legacy nmf_multiplicative call found")
    else:
        nbf.write(nb, NB)
        print("[patched] BIO_DEMO: replaced nmf_multiplicative -> nmf_on_X and removed manual normalization")

if __name__ == "__main__":
    main()
