#!/usr/bin/env python
from __future__ import annotations
import re, sys
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/BIO_DEMO.ipynb")

RE_IMPORT = re.compile(r"^(\s*from\s+ts2eg\.core\s+import\s+)([^\n#]+)", re.M)

def ensure_info_gain_import(src: str) -> str:
    """
    If there's a 'from ts2eg.core import ...' line, ensure 'info_gain_payoffs' is present once.
    Preserve existing names and order where possible.
    """
    def repl(m):
        prefix, names = m.groups()
        parts = [p.strip() for p in names.split(",")]
        if "info_gain_payoffs" not in parts:
            # Try to insert right after growth_payoffs if present, else append
            try:
                gi = parts.index("growth_payoffs") + 1
                parts.insert(gi, "info_gain_payoffs")
            except ValueError:
                parts.append("info_gain_payoffs")
        # Deduplicate while preserving order
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                out.append(p); seen.add(p)
        return prefix + ", ".join(out)
    return RE_IMPORT.sub(repl, src)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False
    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", "") or ""
        new = ensure_info_gain_import(src)
        if new != src:
            c["source"] = new
            changed = True
    if changed:
        nbf.write(nb, NB)
        print("[patched] BIO_DEMO.ipynb: ensured 'info_gain_payoffs' import from ts2eg.core")
    else:
        print("[ok] BIO_DEMO.ipynb: imports already include info_gain_payoffs")

if __name__ == "__main__":
    main()
