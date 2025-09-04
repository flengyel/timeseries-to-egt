#!/usr/bin/env python
from __future__ import annotations
import re, sys
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/CYBER_DEMO.ipynb")

RE_BAD_PATH = re.compile(r"^\s*sys\.path\.append\([^)]*\)\s*$", re.M)
RE_BAD_IMPORT = re.compile(r"^\s*from\s+(?:gameify|gamify)_timeseries\s+import\s+([^\n#]+)", re.M)
RE_LAMBDA_KW = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source","") or ""

        # 1) drop legacy sys.path tweak
        new = RE_BAD_PATH.sub("", src)

        # 2) rewrite legacy import -> ts2eg.core ; preserve imported names, add nmf_on_X
        def _rewrite(m):
            names = [p.strip() for p in m.group(1).split(",")]
            if "nmf_on_X" not in names:
                names.append("nmf_on_X")
            # dedupe preserving order
            out, seen = [], set()
            for n in names:
                if n and n not in seen:
                    out.append(n); seen.add(n)
            return "from ts2eg.core import " + ", ".join(out)
        newer = RE_BAD_IMPORT.sub(_rewrite, new)

        # 3) rename lambda_ -> ridge
        newest = RE_LAMBDA_KW.sub("ridge=", newer)

        if newest != src:
            c["source"] = newest
            changed = True

    # set/confirm CI tag
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    if changed:
        nbf.write(nb, NB)
        print("[patched] CYBER_DEMO.ipynb imports (ts2eg.core) + ridge kw")
    else:
        print("[ok] CYBER_DEMO.ipynb: no changes")

if __name__ == "__main__":
    main()
