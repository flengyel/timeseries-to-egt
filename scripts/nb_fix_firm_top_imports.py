#!/usr/bin/env python
from pathlib import Path
import nbformat as nbf, re

NB = Path("notebooks/FIRM_DEMO.ipynb")
BAD = re.compile(r"^\s*from\s+ts2eg\s+import\s+nmf_on_X\s*,\s*value_gradient_payoffs\s*,\s*estimate_A_from_series\s*,\s*find_ESS\s*$", re.M)
GOOD= "from ts2eg.core import nmf_on_X, value_gradient_payoffs, estimate_A_from_series, find_ESS"

nb = nbf.read(NB, as_version=4)
changed = False
for c in nb.cells:
    if c.get("cell_type") != "code": 
        continue
    src = c.get("source","") or ""
    new = BAD.sub(GOOD, src)
    if new != src:
        c["source"] = new
        changed = True
if changed:
    nbf.write(nb, NB)
    print("[patched] FIRM_DEMO: ts2eg.core import fixed")
else:
    print("[ok] FIRM_DEMO: no change")
