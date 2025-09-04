#!/usr/bin/env python
from __future__ import annotations
import re
from pathlib import Path
import nbformat as nbf

NB = Path("notebooks/PROTEST_DEMO.ipynb")

RE_BAD_PATH   = re.compile(r"^\s*sys\.path\.append\([^)]*\)\s*$", re.M)
RE_GAMIFY_IMP = re.compile(r"^\s*from\s+(?:gameify|gamify)_timeseries\s+import\s+([^\n#]+)", re.M)
RE_TS2_TOP    = re.compile(r"^\s*from\s+ts2eg\s+import\s+([^\n#]+)", re.M)

RE_CORE_LINE  = re.compile(r"^\s*from\s+ts2eg\.core\s+import\s+([^\n#]+)", re.M)
RE_LAMBDA_KW  = re.compile(r"(?<![A-Za-z0-9_])lambda_\s*=")

RE_NMF_CALL   = re.compile(r"nmf_multiplicative\s*\(", re.M)
RE_S_NORM     = re.compile(r"^\s*S\s*=\s*S\s*/.*np\.linalg\.norm\(S.*$", re.M)
RE_SNULL_NORM = re.compile(r"^\s*S_null\s*=\s*S_null\s*/.*np\.linalg\.norm\(S_null.*$", re.M)

def _ensure_names(line: str, extra: list[str]) -> str:
    m = RE_CORE_LINE.search(line)
    if not m: return line
    names = [p.strip() for p in m.group(1).split(",")]
    for x in extra:
        if x not in names:
            names.append(x)
    # dedupe preserving order
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            out.append(n); seen.add(n)
    return RE_CORE_LINE.sub("from ts2eg.core import " + ", ".join(out), line, count=1)

def main():
    nb = nbf.read(NB, as_version=4)
    changed = False

    for c in nb.cells:
        if c.get("cell_type") != "code":
            continue
        src0 = c.get("source","") or ""
        src  = src0

        # Drop sys.path hacks
        src = RE_BAD_PATH.sub("", src)

        # Legacy: from gameify_timeseries import ...  -> ts2eg.core
        def _gamify_to_core(m):
            names = [p.strip() for p in m.group(1).split(",")]
            # make sure nmf_on_X and info_gain_payoffs are present
            for need in ("nmf_on_X", "info_gain_payoffs", "estimate_A_from_series", "find_ESS"):
                if need not in names:
                    names.append(need)
            # dedupe
            seen, out = set(), []
            for n in names:
                if n and n not in seen:
                    out.append(n); seen.add(n)
            return "from ts2eg.core import " + ", ".join(out)
        src = RE_GAMIFY_IMP.sub(_gamify_to_core, src)

        # Top-level ts2eg import -> core; ensure info_gain_payoffs included
        def _ts2_to_core(m):
            names = [p.strip() for p in m.group(1).split(",")]
            if "info_gain_payoffs" not in names:
                names.append("info_gain_payoffs")
            seen, out = set(), []
            for n in names:
                if n and n not in seen:
                    out.append(n); seen.add(n)
            return "from ts2eg.core import " + ", ".join(out)
        src = RE_TS2_TOP.sub(_ts2_to_core, src)

        # If there is already a ts2eg.core import line, ensure info_gain_payoffs is present
        if RE_CORE_LINE.search(src):
            src = _ensure_names(src, ["info_gain_payoffs"])

        # lambda_ -> ridge
        src = RE_LAMBDA_KW.sub("ridge=", src)

        # Legacy NMF -> nmf_on_X; r->k; ensure normalize='l2'
        if RE_NMF_CALL.search(src):
            src = RE_NMF_CALL.sub("nmf_on_X(", src)
            src = src.replace(" r=", " k=")
            if "nmf_on_X(" in src and "normalize=" not in src:
                src = re.sub(r"nmf_on_X\(([^)]*)\)", r"nmf_on_X(\1, normalize='l2')", src, count=1)

        # Drop redundant manual normalization
        src = RE_S_NORM.sub("# (removed) nmf_on_X already normalizes S columns", src)
        src = RE_SNULL_NORM.sub("# (removed) nmf_on_X already normalizes S_null columns", src)

        if src != src0:
            c["source"] = src
            changed = True

    # Set CI tag if missing
    nb.metadata["ts2eg_ci"] = (nb.metadata or {}).get("ts2eg_ci", "fast")

    if changed:
        nbf.write(nb, NB)
        print("[patched] PROTEST_DEMO: imports fixed (ts2eg.core), ridge kw, NMF legacy removed")
    else:
        print("[ok] PROTEST_DEMO: no changes")
if __name__ == "__main__":
    main()
