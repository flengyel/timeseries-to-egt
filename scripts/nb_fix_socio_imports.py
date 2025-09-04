#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import re, nbformat as nbf

NB = Path("notebooks/SOCIO_DEMO.ipynb")

# patterns
RE_TS2_TOP    = re.compile(r"^\s*from\s+ts2eg\s+import\s+([^\n#]+)", re.M)
RE_CORE_LINE  = re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+([^\n#]+)", re.M)
RE_GAMIFY_IMP = re.compile(r"^\s*from\s+(?:gameify|gamify)_timeseries\s+import\s+([^\n#]+)", re.M)
RE_BAD_PATH   = re.compile(r"^\s*sys\.path\.append\([^)]*\)\s*$", re.M)
RE_LAMBDA_KW  = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")
RE_NMF_CALL   = re.compile(r"nmf_multiplicative\s*\(", re.M)
RE_S_NORM     = re.compile(r"^\s*S\s*=\s*S\s*/.*np\.linalg\.norm\(S.*$", re.M)
RE_SNULL_NORM = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

CANON_HEADER = "# --- Canonical imports (ts2eg only) ---"
CANON_BODY = (
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import ts2eg as gm\n"
    "from ts2eg.core import nmf_on_X, growth_payoffs, estimate_A_from_series, find_ESS, info_gain_payoffs\n"
    "try:\n"
    "    from ts2eg import extensions as ext\n"
    "except Exception:\n"
    "    ext = None  # optional\n"
)

def ensure_names(line: str, extra: list[str]) -> str:
    m = RE_CORE_LINE.search(line)
    if not m: return line
    names = [p.strip() for p in m.group(1).split(",")]
    for x in extra:
        if x not in names:
            names.append(x)
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            out.append(n); seen.add(n)
    return RE_CORE_LINE.sub("from ts2eg.core import " + ", ".join(out), line, count=1)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False
    # 1) Replace/insert a canonical imports cell at top if we see any “Canonical imports” marker; else leave content alone.
    inserted = False
    for i, c in enumerate(nb.cells):
        if c.get("cell_type") != "code": continue
        src = c.get("source","") or ""
        if "Canonical imports" in src:
            c["source"] = CANON_HEADER + "\n" + CANON_BODY
            inserted = True
            changed = True
            break
    if not inserted:
        # prepend a clean imports cell
        nb.cells.insert(0, nbf.v4.new_code_cell(CANON_HEADER + "\n" + CANON_BODY))
        changed = True

    # 2) Sweep other cells for legacy lines and import fixes
    for c in nb.cells[1:]:
        if c.get("cell_type") != "code": continue
        src0 = c.get("source","") or ""
        s = src0
        s = RE_BAD_PATH.sub("", s)
        # gamify -> core (ensure info_gain_payoffs present)
        def _gamify_to_core(m):
            names = [p.strip() for p in m.group(1).split(",")]
            for need in ("nmf_on_X", "info_gain_payoffs", "estimate_A_from_series", "find_ESS"):
                if need not in names: names.append(need)
            seen, out = set(), []
            for n in names:
                if n and n not in seen: out.append(n); seen.add(n)
            return "from ts2eg.core import " + ", ".join(out)
        s = RE_GAMIFY_IMP.sub(_gamify_to_core, s)
        # top-level ts2eg import -> core; ensure info_gain_payoffs
        def _ts2_to_core(m):
            names = [p.strip() for p in m.group(1).split(",")]
            if "info_gain_payoffs" not in names: names.append("info_gain_payoffs")
            seen, out = set(), []
            for n in names:
                if n and n not in seen: out.append(n); seen.add(n)
            return "from ts2eg.core import " + ", ".join(out)
        s = RE_TS2_TOP.sub(_ts2_to_core, s)
        if RE_CORE_LINE.search(s):
            s = ensure_names(s, ["info_gain_payoffs"])
        # lambda_ -> ridge
        s = RE_LAMBDA_KW.sub("ridge=", s)
        # nmf_multiplicative -> nmf_on_X; r->k; ensure normalize='l2'
        if RE_NMF_CALL.search(s):
            s = RE_NMF_CALL.sub("nmf_on_X(", s)
            s = s.replace(" r=", " k=")
            if "nmf_on_X(" in s and "normalize=" not in s:
                s = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", s, count=1)
        # drop redundant normalization lines
        s = RE_S_NORM.sub("# (removed) nmf_on_X already normalizes S columns", s)
        s = RE_SNULL_NORM.sub("# (removed) nmf_on_X already normalizes S_null columns", s)

        if s != src0:
            c["source"] = s
            changed = True

    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")
    if changed:
        with NB.open("w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("[patched] SOCIO_DEMO: imports fixed (ts2eg.core), ridge kw, NMF legacy removed")
    else:
        print("[ok] SOCIO_DEMO: no changes")

if __name__ == "__main__":
    main()
