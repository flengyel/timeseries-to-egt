#!/usr/bin/env python
from __future__ import annotations
import argparse, os, sys, socket, time, platform, asyncio, re, traceback
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# Host-side Windows event loop policy (ZMQ)
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
    return (nb.metadata or {}).get("ts2eg_ci", "fast") == wanted_tag

RESET_PATTERNS = (
    r"^\s*%reset\b",
    r"get_ipython\(\)\.magic\(\s*[\"']reset",
    r"run_line_magic\(\s*[\"']reset",
)

def inject_recovery_cells(nb) -> int:
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
                nb.cells.insert(i + 1, rec); inserted += 1; i += 1
        i += 1
    return inserted

def inject_x_guards(nb) -> int:
    inserted = 0
    pat_use = re.compile(r'(?<![A-Za-z0-9_])X(?![A-Za-z0-9_])')
    for i in range(len(nb.cells) - 1, -1, -1):
        c = nb.cells[i]
        if c.get("cell_type") != "code": continue
        src = c.get("source", "") or ""
        if not pat_use.search(src): continue
        guard_src = (
            "import numpy as _np\n"
            "if 'X' not in globals():\n"
            "    if 'counts' in globals():\n"
            "        X = _np.asarray(counts, dtype=float)\n"
            "    elif 'v_growth' in globals():\n"
            "        X = _np.asarray(v_growth, dtype=float)\n"
            "    else:\n"
            "        raise NameError('X is undefined (no counts/v_growth available)')\n"
        )
        guard = nbf.v4.new_code_cell(guard_src, metadata={"tags": ["ci-ensure-X"]})
        nb.cells.insert(i, guard); inserted += 1
    return inserted

PREAMBLE = r"""# CI preamble (injected)
import os, random, numpy as np, socket
import pandas as pd
import asyncio, platform

# Kernel-side Windows selector loop for ZMQ
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Headless plotting and plt shim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TS2EG_CI", "1")
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0)

# CI synthetic stub: if notebook has no data yet, create tiny offline counts + default K
if os.getenv("TS2EG_CI","0") == "1":
    g = globals()
    if ("counts" not in g) and ("v_growth" not in g) and ("X" not in g):
        rng = np.random.default_rng(0)
        N, T = 4, 80
        counts = np.maximum(rng.lognormal(mean=0.0, sigma=0.4, size=(N, T)), 1e-8)
    if "K" not in g:
        K = 3

# deny network calls
def _deny(*a, **k): raise RuntimeError("network disabled in CI")
socket.create_connection = _deny
"""

def execute_one(path: Path, timeout: int) -> None:
    nb = nbf.read(path, as_version=4)
    pre = nbf.v4.new_code_cell(PREAMBLE, metadata={"tags": ["ci-preamble"]})
    nb.cells.insert(0, pre)
    inject_recovery_cells(nb)
    inject_x_guards(nb)
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
        allow_errors=False
    )
    try:
        ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    except Exception as e:
        failed = OUT_DIR / (path.stem + ".failed.ipynb")
        try:
            nbf.write(nb, failed)
            print(f"[error] saved failed state to {failed}", file=sys.stderr)
        except Exception:
            print("[warn] could not save failed notebook state", file=sys.stderr)
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        raise
    out = OUT_DIR / (path.stem + ".executed.ipynb")
    nbf.write(nb, out)
    print(f"[ok] executed: {path} -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fast", choices=["fast","slow","skip"])
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--only", default="", help="Substring filter on notebook filename")
    ap.add_argument("--list", action="store_true", help="List selected notebooks and exit")
    args = ap.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TS2EG_CI", "1")

    candidates = load_notebooks()
    selected = []
    for p in candidates:
        nb = nbf.read(p, as_version=4)
        if wants(nb, args.tag) and (args.only in p.name):
            selected.append(p)

    if args.list:
        for p in selected:
            print(p.name)
        sys.exit(0)

    if not selected:
        print(f"No notebooks match tag='{args.tag}' and only='{args.only}'", file=sys.stderr)
        sys.exit(0)

    t0 = time.time()
    for p in selected:
        rel = p.relative_to(ROOT)
        print(f">>> Running: {rel}", flush=True)
        try:
            execute_one(p, timeout=args.timeout)
        except Exception:
            print(f"[fail] {rel}", file=sys.stderr, flush=True)
            raise
    print(f"Done {len(selected)} notebooks in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
