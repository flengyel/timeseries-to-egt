#!/usr/bin/env python
from pathlib import Path
import nbformat as nbf, re

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

HDR = re.compile(r"^\s*#\s*---\s*Canonical imports", re.M)

nb = nbf.read(NB, as_version=4)
changed = False
for c in nb.cells:
    if c.get("cell_type") != "code": 
        continue
    src = c.get("source", "") or ""
    if HDR.search(src):
        c["source"] = CANON.strip() + "\n"
        changed = True
# extra hygiene: remove any lone, indented leftovers in other cells
SCRUBS = [
    re.compile(r"^\s+from\s+ts2eg\s+import\s+extensions\s+as\s+ext\s*$", re.M),
    re.compile(r"^\s+ext\s*=\s*None\s+#\s*optional\s*$", re.M),
]
for c in nb.cells:
    if c.get("cell_type") != "code": 
        continue
    src = c.get("source","") or ""
    new = src
    for pat in SCRUBS:
        new = pat.sub("", new)
    if new != src:
        c["source"] = re.sub(r"\n{3,}", "\n\n", new)

if changed:
    nbf.write(NB, nb)
    print("[patched] PROTEST_DEMO: canonical imports cell replaced cleanly")
else:
    print("[ok] PROTEST_DEMO: canonical cell not found (no changes)")
