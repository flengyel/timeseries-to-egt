#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import re, nbformat as nbf

NB = Path("notebooks/PROTEST_DEMO.ipynb")

CANON = """# --- Canonical imports (ts2eg only) ---
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ts2eg as gm
from ts2eg.core import nmf_on_X, growth_payoffs, estimate_A_from_series, find_ESS, info_gain_payoffs
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None  # optional
"""

# heuristics: any cell mentioning these is treated as a broken canonical-imports cell
TRIGGERS = (
    "Canonical imports", "standalone canonical file", "gamify_timeseries",
    "from ts2eg.core import *", "extensions as ext, info_gain_payoffs"
)

BAD_LINE_PATTERNS = [
    re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*,?.*$", re.M),
    re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext.*$", re.M),
    re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+nmf_on_X\s*,\s*info_gain_payoffs\s*$", re.M),
]

def main():
    nb = nbf.read(NB, as_version=4)
    replaced = False
    for c in nb.cells:
        if c.get("cell_type") != "code": 
            continue
        src = c.get("source","") or ""
        if any(t in src for t in TRIGGERS):
            c["source"] = CANON.strip()+"\n"
            replaced = True
            break

    # cleanup of stray bad lines in other cells (if any)
    for c in nb.cells:
        if c.get("cell_type") != "code": 
            continue
        src = c.get("source","") or ""
        new = src
        for pat in BAD_LINE_PATTERNS:
            new = pat.sub("", new)
        if new != src:
            # collapse excess blank lines
            new = re.sub(r"\n{3,}", "\n\n", new)
            c["source"] = new

    if replaced:
        nbf.write(nb, NB)
        print("[patched] PROTEST_DEMO: canonical imports replaced; stray lines cleaned")
    else:
        print("[warn] PROTEST_DEMO: no canonical block replaced")
if __name__ == "__main__":
    main()
