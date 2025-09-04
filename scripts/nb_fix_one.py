#!/usr/bin/env python
"""
Fix ONE notebook in-place:

- Insert a top Bootstrap cell (deterministic, offline, no-net, tiny data stub).
- Replace legacy "gamify_timeseries" fallback imports with ts2eg-only imports.
- Rename lambda_= -> ridge=.
- Set metadata ts2eg_ci = "fast".

Usage:
  python scripts/nb_fix_one.py notebooks/BIO_DEMO.ipynb
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import nbformat as nbf

BOOTSTRAP = """# Bootstrap (CI/offline, deterministic)
import os, random, numpy as np
# headless plotting (some cells call plt.*)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# core API we need
from ts2eg.core import growth_payoffs, nmf_on_X, estimate_A_from_series, find_ESS
import ts2eg as gm
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None

# seeds
os.environ.setdefault("TS2EG_CI", "1")
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0)

# tiny offline data stub (only if notebook didn't define data yet)
g = globals()
if "counts" not in g and "v_growth" not in g and "X" not in g:
    rng = np.random.default_rng(0)
    N, T = 4, 80
    counts = np.maximum(rng.lognormal(mean=0.0, sigma=0.4, size=(N, T)), 1e-8)

# default K if unspecified
if "K" not in g:
    K = 3

# ensure X exists for notebooks that reference it early
if "X" not in g:
    X = np.asarray(g.get("v_growth", g.get("counts")), dtype=float)
"""

CANONICAL_IMPORTS = """# --- Canonical imports (ts2eg only) ---
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from ts2eg.core import nmf_on_X, growth_payoffs, estimate_A_from_series, find_ESS
import ts2eg as gm
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None  # optional
"""

RE_CANON_MARK   = re.compile(r"^\s*#\s*---\s*Canonical imports", re.I|re.M)
RE_LEGACY       = re.compile(r"\bgamify_timeseries\b|egt_extensions\b", re.I|re.M)
RE_LAMBDA_KW    = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")

def fix_one(path: Path) -> None:
    nb = nbf.read(path, as_version=4)

    # 1) ensure metadata tag
    nb.metadata["ts2eg_ci"] = "fast"

    # 2) insert Bootstrap cell at very top (before any code)
    first_code_ix = 0
    pre = nbf.v4.new_code_cell(BOOTSTRAP, metadata={"tags": ["ci-bootstrap"]})
    nb.cells.insert(first_code_ix, pre)

    # 3) clean canonical imports block / legacy fallbacks
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source", "") or ""
        if RE_CANON_MARK.search(src) or RE_LEGACY.search(src):
            c["source"] = CANONICAL_IMPORTS

    # 4) rename lambda_= -> ridge=
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source", "") or ""
        new = RE_LAMBDA_KW.sub("ridge=", src)
        if new != src:
            c["source"] = new

    nbf.write(nb, path)

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/nb_fix_one.py notebooks/<NOTEBOOK>.ipynb", file=sys.stderr)
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Not found: {p}", file=sys.stderr); sys.exit(1)
    fix_one(p)
    print(f"[patched] {p}")

if __name__ == "__main__":
    main()
