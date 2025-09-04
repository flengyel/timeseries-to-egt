#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/FIRM_DEMO.ipynb")

RE_BAD_PATH   = re.compile(r"^\s*sys\.path\.append\([^)]*\)\s*$", re.M)
RE_GAMIFY_IMP = re.compile(r"^\s*from\s+(?:gameify|gamify)_timeseries\s+import\s+([^\n#]+)", re.M)
RE_T2_TOPBAD  = re.compile(r"^\s*from\s+ts2eg\s+import\s+value_gradient_payoffs\s*$", re.M)

RE_LAMBDA_KW  = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")

RE_NMF_CALL   = re.compile(r"nmf_multiplicative\s*\(", re.M)
RE_S_NORM     = re.compile(r"^\s*S\s*=\s*S\s*/.*np\.linalg\.norm\(S.*$", re.M)
RE_SNULL_NORM = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def rewrite_import_block(src: str) -> str:
    """Ensure ts2eg.core imports incl. value_gradient_payoffs; drop legacy lines."""
    src = RE_BAD_PATH.sub("", src)
    # rewrite gamify imports -> ts2eg.core (preserve names, add nmf_on_X)
    def _gamify_to_core(m):
        names = [p.strip() for p in m.group(1).split(",")]
        if "nmf_on_X" not in names:
            names.append("nmf_on_X")
        seen, out = set(), []
        for n in names:
            if n and n not in seen:
                out.append(n); seen.add(n)
        return "from ts2eg.core import " + ", ".join(out)
    src = RE_GAMIFY_IMP.sub(_gamify_to_core, src)
    # fix top-level bad ts2eg import
    src = RE_T2_TOPBAD.sub("from ts2eg.core import value_gradient_payoffs", src)
    return src

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src0 = c.get("source","") or ""
        src  = rewrite_import_block(src0)
        # lambda_ -> ridge
        src  = RE_LAMBDA_KW.sub("ridge=", src)
        # nmf_multiplicative -> nmf_on_X, r->k, ensure normalize='l2'
        if RE_NMF_CALL.search(src):
            src = RE_NMF_CALL.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")
            if "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)
        # drop manual normalization lines
        src = RE_S_NORM.sub("# (removed) nmf_on_X already normalizes S columns", src)
        src = RE_SNULL_NORM.sub("# (removed) nmf_on_X already normalizes S_null columns", src)
        if src != src0:
            c["source"] = src
            changed = True

    # ensure notebook metadata tag
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    if changed:
        nbf.write(nb, NB)
        print("[patched] FIRM_DEMO imports/kw/NMF updated")
    else:
        print("[ok] FIRM_DEMO: no change needed")

if __name__ == "__main__":
    main()
