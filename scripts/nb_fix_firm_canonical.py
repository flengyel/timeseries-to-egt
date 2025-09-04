#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/FIRM_DEMO.ipynb")

CANON = """# --- Canonical imports (ts2eg only) ---
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ts2eg as gm
from ts2eg.core import nmf_on_X, estimate_A_from_series, find_ESS, value_gradient_payoffs
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None  # optional
"""

RE_CANON_BAD = re.compile(
    r"(?s)#\s*---\s*Canonical imports.*?(?:\n\s*\n|$)"
)
RE_BAD_LINES = re.compile(r"^\s*sys\.path\.append\([^)]*\)\s*$", re.M)
RE_WILDCARD  = re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*$", re.M)
RE_INNER_EST = re.compile(r"^\s*from\s+ts2eg\s+import\s+estimate_A_from_series\s*$", re.M)
RE_LAMBDA    = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")
RE_NMF_CALL  = re.compile(r"nmf_multiplicative\s*\(", re.M)
RE_S_NORM    = re.compile(r"^\s*S\s*=\s*S\s*/.*np\.linalg\.norm\(S.*$", re.M)
RE_SNULL_N   = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src0 = c.get("source","") or ""
        src  = src0

        # Replace the whole bad canonical block with clean ts2eg-only imports
        if "# --- Canonical imports" in src or "standalone canonical file" in src or "gamify_timeseries" in src:
            src = RE_CANON_BAD.sub(CANON + "\n", src)

        # Cleanups anywhere in the notebook
        src = RE_BAD_LINES.sub("", src)                     # drop sys.path hacks
        src = RE_WILDCARD.sub("", src)                     # drop wildcard import line (if any)
        src = RE_INNER_EST.sub("from ts2eg.core import estimate_A_from_series", src)  # fix inner import
        src = RE_LAMBDA.sub("ridge=", src)                 # lambda_ -> ridge

        # Legacy NMF -> nmf_on_X (and r->k, ensure normalize once)
        if RE_NMF_CALL.search(src):
            src = RE_NMF_CALL.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")
            if "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)
        # Drop redundant manual normalization lines
        src = RE_S_NORM.sub("# (removed) nmf_on_X normalizes S columns", src)
        src = RE_SNULL_N.sub("# (removed) nmf_on_X normalizes S_null columns", src)

        if src != src0:
            c["source"] = src
            changed = True

    # Ensure CI tag
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    if changed:
        nbf.write(nb, NB)
        print("[patched] FIRM_DEMO: canonical imports replaced; inner import fixed; legacy nmf cleaned")
    else:
        print("[ok] FIRM_DEMO: no changes applied")

if __name__ == "__main__":
    main()
