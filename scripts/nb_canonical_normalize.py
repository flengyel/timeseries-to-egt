#!/usr/bin/env python
from __future__ import annotations
import sys, re
from pathlib import Path
import nbformat as nbf

CANON_HDR = re.compile(r"^\s*#\s*---\s*Canonical imports.*$", re.M)

def normalize_one(nb_path: Path) -> bool:
    nb = nbf.read(nb_path, as_version=4)
    cells = nb.cells

    # 1) find all canonical-imports cells
    canon_idx = [i for i,c in enumerate(cells)
                 if c.get("cell_type")=="code" and CANON_HDR.search(c.get("source") or "")]
    if not canon_idx:
        return False

    # 2) keep the first; remove the rest
    keep = canon_idx[0]
    keep_cell = cells[keep]
    # strict clean content for the kept cell (no stray indentation)
    keep_cell.source = (
        "# --- Canonical imports (ts2eg only) ---\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import ts2eg as gm\n"
        "from ts2eg.core import nmf_on_X, growth_payoffs, estimate_A_from_series, find_ESS, info_gain_payoffs\n"
        "try:\n"
        "    from ts2eg import extensions as ext\n"
        "except Exception:\n"
        "    ext = None  # optional\n"
    )
    for j in reversed(canon_idx[1:]):
        del cells[j]

    # 3) move the canonical cell to top (index 0)
    if keep != 0:
        keep_cell = cells.pop(keep)
        cells.insert(0, keep_cell)

    # 4) scrub stray “try:”/“except:” or wildcard/core-extensions lines in other cells
    SCRUBS = [
        re.compile(r"^\s*try:\s*$", re.M),
        re.compile(r"^\s*except\s+Exception:\s*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+\*\s*$", re.M),
        re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+extensions\s+as\s+ext.*$", re.M),
        re.compile(r"^\s+from\s+ts2eg\s+import\s+extensions\s+as\s+ext\s*$", re.M),
        re.compile(r"^\s+ext\s*=\s*None\s+#\s*optional\s*$", re.M),
    ]
    for c in cells:
        if c.get("cell_type") != "code": continue
        s = c.get("source","") or ""
        s2 = s
        for pat in SCRUBS:
            s2 = pat.sub("", s2)
        if s2 != s:
            c["source"] = re.sub(r"\n{3,}", "\n\n", s2)

    # 5) ensure CI tag
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    with nb_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return True

def main(argv):
    if len(argv) == 1:
        paths = [Path("notebooks/PROTEST_DEMO.ipynb")]
    else:
        paths = [Path(p) for p in argv[1:]]
    changed = False
    for p in paths:
        if normalize_one(p):
            print(f"[patched] {p}: canonical imports normalized")
            changed = True
        else:
            print(f"[ok] {p}: no canonical imports found")
    sys.exit(0 if changed else 0)

if __name__ == "__main__":
    main(sys.argv)
