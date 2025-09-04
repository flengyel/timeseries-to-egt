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

def main():
    nb = nbf.read(NB, as_version=4)
    replaced_idx = None

    # Replace the first canonical-imports cell strictly
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "") or ""
        if HDR.search(src):
            c["source"] = CANON.strip() + "\n"
            replaced_idx = i
            break

    # Remove any *other* canonical-imports cells if present
    if replaced_idx is not None:
        for j, c in enumerate(nb.cells):
            if j == replaced_idx or c.get("cell_type") != "code":
                continue
            src = c.get("source", "") or ""
            if HDR.search(src):
                c["source"] = "# (removed duplicate canonical imports)\n"

    # Scrub stray indented leftovers across all code cells
    SCRUBS = [
        re.compile(r"^\s+from\s+ts2eg\s+import\s+extensions\s+as\s+ext\s*$", re.M),
        re.compile(r"^\s+ext\s*=\s*None\s+#\s*optional\s*$", re.M),
        re.compile(r"^\s*try:\s*$", re.M),
        re.compile(r"^\s*except\s+Exception:\s*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*$", re.M),
    ]
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""
        new = src
        for pat in SCRUBS:
            new = pat.sub("", new)
        if new != src:
            # collapse 3+ blank lines
            new = re.sub(r"\n{3,}", "\n\n", new)
            c["source"] = new

    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    # Write (correct order: write(nb, fp))
    with NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[patched] PROTEST_DEMO: canonical imports replaced; leftovers scrubbed")

if __name__ == "__main__":
    main()
