#!/usr/bin/env python
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

NB = Path("notebooks/FIRM_DEMO.ipynb")

def ensure_valgrad_import():
    nb = nbf.read(NB, as_version=4)
    inserted = False
    updated  = False

    # Try to add to an existing "from ts2eg.core import ..." line
    for c in nb.cells:
        if c.get("cell_type") != "code": continue
        src = c.get("source","") or ""
        if "from ts2eg.core import" in src:
            # crude, order-preserving dedupe
            lines = src.splitlines()
            for i, line in enumerate(lines):
                if "from ts2eg.core import" in line:
                    head, names = line.split("import", 1)
                    parts = [p.strip() for p in names.split(",")]
                    if "value_gradient_payoffs" not in parts:
                        parts.append("value_gradient_payoffs")
                        # dedupe preserving order
                        seen, out = set(), []
                        for p in parts:
                            if p and p not in seen:
                                out.append(p); seen.add(p)
                        lines[i] = head + "import " + ", ".join(out)
                        c["source"] = "\n".join(lines)
                        updated = True
                        break
            if updated:
                break

    # If not found anywhere, insert a minimal import cell at the top
    if not updated:
        cell = nbf.v4.new_code_cell(
            "from ts2eg.core import value_gradient_payoffs",
            metadata={"tags": ["ci-fix-imports"]}
        )
        nb.cells.insert(0, cell)
        inserted = True

    nbf.write(nb, NB)
    print("[patched] FIRM_DEMO: " + ("inserted import cell" if inserted else "updated existing import line"))

if __name__ == "__main__":
    ensure_valgrad_import()
