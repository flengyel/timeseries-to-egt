#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import nbformat as nbf
import re

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

TRIGGERS = (
    "Canonical imports", "standalone canonical file", "gamify_timeseries",
    "SimpleNamespace(", "egt_extensions", "extensions as ext, info_gain_payoffs"
)

def main():
    nb = nbf.read(NB, as_version=4)
    canon_idxs = []
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code":
            continue
        src = (c.get("source") or "")
        if any(t in src for t in TRIGGERS):
            canon_idxs.append(i)

    if not canon_idxs:
        print("[warn] PROTEST_DEMO: no canonical block found")
    else:
        # keep the first, replace it with clean canonical imports
        first = canon_idxs[0]
        nb.cells[first].source = CANON.strip() + "\n"
        # neuter all other triggered cells
        for i in canon_idxs[1:]:
            nb.cells[i].source = "# (removed legacy fallback imports)\n"

    # scrub any stray bad lines in remaining code cells
    BAD_LINE_PATTERNS = [
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*,?.*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext.*$", re.M),
        re.compile(r"^\s*try:\s*$", re.M),
        re.compile(r"^\s*except\s+Exception:\s*$", re.M),
        re.compile(r"^\s*raise\s+ImportError\(.*gamify.*\)\s*$", re.M),
    ]
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = (c.get("source") or "")
        new = src
        for pat in BAD_LINE_PATTERNS:
            new = pat.sub("", new)
        if new != src:
            # collapse excessive blank lines
            new = re.sub(r"\n{3,}", "\n\n", new)
            c.source = new

    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")
    nbf.write(nb, NB)
    print("[patched] PROTEST_DEMO: canonical imports normalized; fallbacks removed")

if __name__ == "__main__":
    main()
