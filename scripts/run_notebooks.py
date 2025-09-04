#!/usr/bin/env python
"""
Execute notebooks tagged ts2eg_ci == 'fast' by default, headless, no network.
Writes executed copies under artifacts/notebooks/*.executed.ipynb.

Usage:
  python scripts/run_notebooks.py [--tag fast|slow|skip] [--timeout 120]

Notes:
- Injects a CI preamble cell (seeds, TS2EG_CI=1, network block).
- Inserts recovery cells after any `%reset`-like use to restore imports/seeds.
- Fails on any execution error (return code != 0).
"""
from __future__ import annotations
import argparse, os, sys, socket, time, platform, asyncio, re
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# Windows: ensure ZMQ works with asyncio by using the Selector policy
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "artifacts" / "notebooks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_notebooks():
    return sorted(NB_DIR.glob("*.ipynb"))

def wants(nb, wanted_tag: str) -> bool:
    tag = (nb.metadata or {}).get("ts2eg_ci", "fast")
    return tag == wanted_tag

# Detect resets that clear namespace
RESET_PATTERNS = (
    r"^\s*%reset\b",                                # IPython magic
    r"get_ipython\(\)\.magic\(\s*[\"']reset",       # explicit call
    r"run_line_magic\(\s*[\"']reset",               # run_line_magic('reset', ...)
)

def inject_recovery_cells(nb) -> int:
    """
    After any reset-like cell, insert a tiny recovery cell that restores
    imports, seeds, and the network block.
    """
    inserted = 0
    i = 0
    while i < len(nb.cells):
        c = nb.cells[i]
        if c.get("cell_type") == "code":
            src = c.get("source", "") or ""
            if any(re.search(p, src, flags=re.M) for p in RESET_PATTERNS):
                rec_src = (
                    "import os, random, numpy as np, pandas as pd, socket as _socket\n"
                    "os.environ.setdefault('TS2EG_CI','1')\n"
                    "os.environ['PYTHONHASHSEED']='0'\n"
                    "random.seed(0); np.random.seed(0)\n"
                    "def _deny(*a, **k): raise RuntimeError('network disabled in CI')\n"
                    "_socket.create_connection = _deny\n"
                )
                rec = nbf.v4.new_code_cell(rec_src, metadata={"tags": ["ci-recover-imports"]})
                nb.cells.insert(i + 1, rec)
                inserted += 1
                i += 1
        i += 1
    return inserted

PREAMBLE = r"""# CI preamble (injected)
import os, random, numpy as np, socket
import pandas as pd
os.environ.setdefault("TS2EG_CI", "1")
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0)

# deny network calls
def _deny(*a, **k): raise RuntimeError("network disabled in CI")
socket.create_connection = _deny
"""

def execute_one(path: Path, timeout: int) -> None:
    nb = nbf.read(path, as_version=4)
    # Inject preamble as the very first cell (non-destructive to repo)
    pre = nbf.v4.new_code_cell(PREAMBLE, metadata={"tags": ["ci-preamble"]})
    nb.cells.insert(0, pre)
    # Insert recovery cells after any %reset / reset magic
    inject_recovery_cells(nb)

    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=False
    )
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})

    out = OUT_DIR / (path.stem + ".executed.ipynb")
    nbf.write(nb, out)
    print(f"[ok] executed: {path} -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fast", choices=["fast","slow","skip"])
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    # CI-friendly environment
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TS2EG_CI", "1")

    notebooks = load_notebooks()
    selected = []
    for p in notebooks:
        nb = nbf.read(p, as_version=4)
        if wants(nb, args.tag):
            selected.append(p)
    if not selected:
        print(f"No notebooks with ts2eg_ci == '{args.tag}'", file=sys.stderr)
        sys.exit(0)

    t0 = time.time()
    for p in selected:
        execute_one(p, timeout=args.timeout)
    print(f"Done {len(selected)} notebooks in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
