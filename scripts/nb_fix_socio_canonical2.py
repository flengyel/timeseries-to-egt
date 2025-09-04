#!/usr/bin/env python
from pathlib import Path
import re, nbformat as nbf

NB = Path("notebooks/SOCIO_DEMO.ipynb")
HDR = re.compile(r"^\s*#\s*---\s*Canonical imports", re.M)
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

def main():
    nb = nbf.read(NB, as_version=4)
    cells = nb.cells

    # find all canonical cells
    canon_idxs = [i for i,c in enumerate(cells)
                  if c.get("cell_type")=="code" and HDR.search(c.get("source") or "")]
    if not canon_idxs:
        # insert at top if missing
        cells.insert(0, nbf.v4.new_code_cell(CANON.strip() + "\n"))
        keep_idx = 0
    else:
        keep_idx = canon_idxs[0]
        cells[keep_idx].source = CANON.strip() + "\n"
        # remove duplicates
        for j in reversed(canon_idxs[1:]):
            del cells[j]
            if j < keep_idx:
                keep_idx -= 1

    # scrub ONLY outside the canonical cell
    SCRUBS = [
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext.*$", re.M),
        re.compile(r"^\s+from\s+ts2eg\s+import\s+extensions\s+as\s+ext\s*$", re.M),
        re.compile(r"^\s+ext\s*=\s*None\s+#\s*optional\s*$", re.M),
        re.compile(r"^\s*try:\s*$", re.M),
        re.compile(r"^\s*except\s+Exception:\s*$", re.M),
    ]
    for i,c in enumerate(cells):
        if i == keep_idx or c.get("cell_type")!="code":
            continue
        s = c.get("source","") or ""
        s2 = s
        for pat in SCRUBS:
            s2 = pat.sub("", s2)
        if s2 != s:
            c["source"] = re.sub(r"\n{3,}", "\n\n", s2)

    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")
    with NB.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[patched] SOCIO_DEMO: canonical imports fixed; scrubs applied outside canonical cell")

if __name__ == "__main__":
    main()
