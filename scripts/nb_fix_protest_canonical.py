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

PAT = re.compile(r"(?s)^#\s*---\s*Canonical imports.*?(?=\Z|\n\s*\n)", re.M)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""
        if "# --- Canonical imports" in src:
            new = PAT.sub(CANON.strip()+"\n", src)
            if new != src:
                c["source"] = new
                changed = True
                break
    if changed:
        nbf.write(nb, NB)
        print("[patched] PROTEST_DEMO: canonical imports cell replaced")
    else:
        print("[warn] PROTEST_DEMO: canonical imports cell not found / unchanged")

if __name__ == "__main__":
    main()
