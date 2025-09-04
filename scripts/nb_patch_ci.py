#!/usr/bin/env python
"""
Normalize notebook cell IDs (no validator.normalize), ensure a top 'Bootstrap' cell,
add deterministic seeds, introduce TS2EG_CI flag, and rename keyword 'lambda_=' -> 'ridge='.
Sets notebook metadata ts2eg_ci="fast" (adjust later if needed).

Idempotent and safe to re-run. Requires: nbformat>=5.1.4
"""
from __future__ import annotations
import re, os, sys, uuid
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = [ROOT / "notebooks"]

BOOTSTRAP_TAG = "bootstrap"
CI_META_KEY = "ts2eg_ci"
CI_META_VAL = "fast"

BOOTSTRAP_SRC = """# Bootstrap (CI-safe)
# tags: [bootstrap]
import os, random, numpy as np
import ts2eg

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# Downshift knob: set TS2EG_CI=1 in CI to keep runtime small.
TS2EG_CI = os.getenv("TS2EG_CI") == "1"
"""

# rename lambda_ only when used as a keyword argument (avoid touching Python 'lambda')
RE_LAMBDA_KW = re.compile(r"(?<![A-Za-z0-9_])lambda_\\s*=", re.M)

def normalize_ids_inplace(nb) -> None:
    """Assign a stable-ish id to any cell missing one."""
    for cell in nb.cells:
        if "id" not in cell or not cell["id"]:
            cell["id"] = uuid.uuid4().hex[:12]

def has_bootstrap(nb) -> bool:
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        md = c.get("metadata", {}) or {}
        if "tags" in md and BOOTSTRAP_TAG in (md.get("tags") or []):
            return True
        src = c.get("source", "") or ""
        if "Bootstrap" in src and "tags:" in src:
            return True
    return False

def ensure_bootstrap(nb) -> bool:
    if has_bootstrap(nb):
        return False
    cell = nbf.v4.new_code_cell(BOOTSTRAP_SRC)
    cell.metadata = {"tags": [BOOTSTRAP_TAG]}
    nb.cells.insert(0, cell)
    return True

def ensure_ci_meta(nb) -> bool:
    if nb.metadata.get(CI_META_KEY) != CI_META_VAL:
        nb.metadata[CI_META_KEY] = CI_META_VAL
        return True
    return False

def patch_lambda_kw(nb) -> bool:
    changed = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        s = c.get("source", "") or ""
        new = RE_LAMBDA_KW.sub("ridge=", s)
        if new != s:
            c["source"] = new
            changed = True
    return changed

def main():
    nb_paths = []
    for d in NB_DIRS:
        if d.is_dir():
            nb_paths.extend(sorted(d.rglob("*.ipynb")))
    if not nb_paths:
        print("No notebooks found.", file=sys.stderr)
        sys.exit(2)

    ins_cnt = kw_cnt = meta_cnt = norm_cnt = 0
    for p in nb_paths:
        nb = nbf.read(p, as_version=4)
        normalize_ids_inplace(nb); norm_cnt += 1
        inserted = ensure_bootstrap(nb); ins_cnt += (1 if inserted else 0)
        kw = patch_lambda_kw(nb); kw_cnt += (1 if kw else 0)
        meta = ensure_ci_meta(nb); meta_cnt += (1 if meta else 0)
        nbf.write(nb, p)
        print(f"[ok] {p.relative_to(ROOT)}"
              f"{' +bootstrap' if inserted else ''}"
              f"{' +lambda_->ridge' if kw else ''}"
              f"{' +ts2eg_ci=fast' if meta else ''}")

    print(f"\nSummary: normalized={norm_cnt}, inserted_bootstrap={ins_cnt}, "
          f"renamed_lambda_kw_in={kw_cnt}, set_ci_meta_in={meta_cnt}")

if __name__ == "__main__":
    main()
