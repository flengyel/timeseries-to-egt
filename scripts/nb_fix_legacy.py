#!/usr/bin/env python
"""
Fix notebooks by:
  1) Removing legacy 'gamify_timeseries.py' / 'egt_extensions' fallbacks.
  2) Using ts2eg-only imports.
  3) Ensuring X exists before the first nmf_on_X(...) call.

Idempotent and safe to re-run.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# Heuristics
IMPORT_MARKER = re.compile(r"^\s*#\s*---\s*Canonical imports", re.I|re.M)
HAS_LEGACY = re.compile(r"\bgamify_timeseries\b|egt_extensions\b", re.I|re.M)
CALLS_NMF = re.compile(r"\bnmf_on_X\s*\(", re.M)
DEF_X = re.compile(r"^\s*X\s*=", re.M)

CLEAN_IMPORTS = """# --- Canonical imports (ts2eg only) ---
import ts2eg as gm
from ts2eg import nmf_on_X, growth_payoffs
try:
    from ts2eg import extensions as ext
except Exception:
    ext = None  # optional
"""

X_GUARD = """# Ensure feature matrix X exists before nmf_on_X
import numpy as _np
if 'X' not in globals():
    if 'v_growth' in globals():
        X = _np.asarray(v_growth, dtype=float)
    elif 'counts' in globals():
        X = _np.asarray(counts, dtype=float)
    else:
        raise NameError("X is undefined; expected v_growth or counts earlier in the notebook.")
"""

def notebooks():
    return sorted(NB_DIR.glob("*.ipynb"))

def rewrite_imports(nb) -> bool:
    """Replace legacy 'canonical imports with fallbacks' with CLEAN_IMPORTS."""
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""
        if IMPORT_MARKER.search(src) or HAS_LEGACY.search(src):
            c["source"] = CLEAN_IMPORTS
            return True
    return False

def insert_x_guard(nb) -> bool:
    """Insert X_GUARD immediately before the first nmf_on_X call if X not assigned anywhere."""
    # if X is assigned anywhere already, skip
    for c in nb.cells:
        if c.get("cell_type") == "code" and DEF_X.search(c.get("source","") or ""):
            return False
    idx = None
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code":
            continue
        if CALLS_NMF.search(c.get("source","") or ""):
            idx = i
            break
    if idx is None:
        return False
    guard = nbf.v4.new_code_cell(X_GUARD, metadata={"tags": ["ci-ensure-X"]})
    nb.cells.insert(idx, guard)
    return True

def main():
    changed_any = False
    for p in notebooks():
        nb = nbf.read(p, as_version=4)
        ci = rewrite_imports(nb)
        cx = insert_x_guard(nb)
        if ci or cx:
            nbf.write(nb, p)
            changed_any = True
            flags = []
            if ci: flags.append("imports")
            if cx: flags.append("X-guard")
            print(f"[patched] {p.name}: {', '.join(flags)}")
        else:
            print(f"[ok] {p.name}: no change")
    if not changed_any:
        print("No modifications needed.")
if __name__ == "__main__":
    main()
