#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
VALID_TAGS = {"fast", "slow", "skip"}

# Execute the exact committed Python cells in order, in an isolated subprocess.
# The notebook audit rejects non-Python cells, shell escapes, and magics.  This
# avoids both CI-time source injection and a local Jupyter/ZMQ socket dependency.
DIRECT_EXECUTOR = r"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
notebook = json.loads(path.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__", "__file__": str(path)}
for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(str(part) for part in source)
    else:
        source = str(source)
    exec(compile(source, f"{path.as_posix()}#cell-{cell_number}", "exec"), namespace)
"""


def load_notebooks() -> list[Path]:
    return sorted(NB_DIR.glob("*.ipynb"))


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_tag(nb: dict, path: Path) -> str:
    tag = (nb.get("metadata", {}) or {}).get("ts2eg_ci")
    if not isinstance(tag, str) or tag not in VALID_TAGS:
        allowed = ", ".join(sorted(VALID_TAGS))
        raise ValueError(
            f"{path.relative_to(ROOT)} has invalid metadata.ts2eg_ci={tag!r}; "
            f"expected one of: {allowed}"
        )
    return tag


def execute_one(path: Path, timeout: int) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("TS2EG_CI", "1")
    environment.setdefault("PYTHONHASHSEED", "0")
    try:
        subprocess.run(
            [sys.executable, "-c", DIRECT_EXECUTOR, str(path)],
            cwd=ROOT,
            env=environment,
            timeout=timeout,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(f"[ok] executed committed cells: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fast", choices=sorted(VALID_TAGS))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--only", default="", help="Substring filter on notebook filename")
    ap.add_argument("--list", action="store_true", help="List selected notebooks and exit")
    args = ap.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TS2EG_CI", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    candidates = load_notebooks()
    selected = []
    for p in candidates:
        nb = read_notebook(p)
        if notebook_tag(nb, p) == args.tag and args.only in p.name:
            selected.append(p)

    if not selected:
        print(f"No notebooks match tag='{args.tag}' and only='{args.only}'", file=sys.stderr)
        return 2

    if args.list:
        for p in selected:
            print(p.name)
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
