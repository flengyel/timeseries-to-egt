#!/usr/bin/env python
from pathlib import Path
import re, nbformat as nbf

NB = Path("notebooks/SOCIO_DEMO.ipynb")

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

def main():
    nb = nbf.read(NB, as_version=4)
    cells = nb.cells

    # 1) replace the first canonical-imports cell strictly; remove duplicates
    first_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code": 
            continue
        src = (c.get("source") or "")
        if HDR.search(src) or "Canonical imports" in src:
            if first_idx is None:
                c["source"] = CANON.strip() + "\n"
                first_idx = i
            else:
                c["source"] = "# (removed duplicate canonical imports)\n"

    # if none found, insert at top
    if first_idx is None:
        cells.insert(0, nbf.v4.new_code_cell(CANON.strip() + "\n"))

    # 2) scrub stray bad lines across all code cells
    SCRUBS = [
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*,?.*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext.*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext\s*,\s*info_gain_payoffs\s*$", re.M),
        re.compile(r"^\s*try:\s*$", re.M),
        re.compile(r"^\s*except\s+Exception:\s*$", re.M),
    ]
    for c in cells:
        if c.get("cell_type") != "code": 
            continue
        src = (c.get("source") or "")
        new = src
        for pat in SCRUBS:
            new = pat.sub("", new)
        if new != src:
            new = re.sub(r"\n{3,}", "\n\n", new)
            c["source"] = new

    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")
    with NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[patched] SOCIO_DEMO: canonical imports normalized and leftovers scrubbed")

if __name__ == "__main__":
    main()
