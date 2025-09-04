#!/usr/bin/env python
import sys, re, os
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = [ROOT / "notebooks"]

CHECKS = {
    "has_import_ts2eg": re.compile(r"^\s*(from\s+ts2eg\s+import|import\s+ts2eg)\b", re.M),
    "has_seed_numpy": re.compile(r"\b(np\.random\.seed\(|np\.random\.default_rng\(\s*\d+\s*\))"),
    "has_seed_python": re.compile(r"\brandom\.seed\("),
    "uses_TS2EG_CI": re.compile(r"TS2EG_CI"),
    "uses_lambda_param": re.compile(r"\blambda_\s*="),
    "uses_np_ptp_func": re.compile(r"\bnp\.ptp\s*\("),
    "uses_attr_ptp": re.compile(r"\.ptp\s*\("),  # will greenlight if np.ptp also present in same cell
    "internet_hints": re.compile(
        r"\b(yfinance|pandas_datareader|requests|urllib|httpx|pd\.read_csv\(['\"]https?://)", re.I
    ),
    "bootstrap_marker": re.compile(r"(#\s*Bootstrap|\bbootstrap\b)", re.I),
}

def audit_nb(path: Path):
    nb = nbf.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    src_all = "\n".join(c.get("source", "") for c in code_cells)

    def _ok(pattern, source):
        return bool(pattern.search(source))

    # First non-empty code cell heuristics
    first_code = next((c for c in code_cells if c.get("source", "").strip()), None)
    first_has_bootstrap = False
    if first_code:
        first_has_bootstrap = _ok(CHECKS["bootstrap_marker"], first_code.get("source","")) or \
                              ("tags" in first_code.get("metadata", {}) and \
                               "bootstrap" in first_code["metadata"]["tags"])

    # Attribute .ptp flagged only if there's no np.ptp in same cell
    attr_ptp_bad = False
    for c in code_cells:
        s = c.get("source","")
        if CHECKS["uses_attr_ptp"].search(s) and not CHECKS["uses_np_ptp_func"].search(s):
            attr_ptp_bad = True
            break

    md = nb.metadata if hasattr(nb, "metadata") else {}
    tag = md.get("ts2eg_ci")  # fast|slow|skip

    return {
        "file": str(path.relative_to(ROOT)),
        "ci_tag": tag,
        "import_ts2eg": _ok(CHECKS["has_import_ts2eg"], src_all),
        "seed_numpy": _ok(CHECKS["has_seed_numpy"], src_all),
        "seed_python": _ok(CHECKS["has_seed_python"], src_all),
        "ts2eg_ci_env": _ok(CHECKS["uses_TS2EG_CI"], src_all),
        "bootstrap_top": first_has_bootstrap,
        "lambda_param": _ok(CHECKS["uses_lambda_param"], src_all),
        "np_ptp_func": _ok(CHECKS["uses_np_ptp_func"], src_all),
        "attr_ptp_bad": attr_ptp_bad,
        "internet": _ok(CHECKS["internet_hints"], src_all),
    }

def main():
    nbs = []
    for d in NB_DIRS:
        if d.is_dir():
            nbs.extend(sorted(d.rglob("*.ipynb")))
    if not nbs:
        print("No notebooks found.", file=sys.stderr)
        sys.exit(2)

    print("file,ci_tag,import_ts2eg,seed_numpy,seed_python,ts2eg_ci_env,bootstrap_top,"
          "lambda_param,np_ptp_func,attr_ptp_bad,internet")
    for p in nbs:
        rep = audit_nb(p)
        print(",".join(str(rep[k]) for k in [
            "file","ci_tag","import_ts2eg","seed_numpy","seed_python","ts2eg_ci_env",
            "bootstrap_top","lambda_param","np_ptp_func","attr_ptp_bad","internet"
        ]))

if __name__ == "__main__":
    main()
