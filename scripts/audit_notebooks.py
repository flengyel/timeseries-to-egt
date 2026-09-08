#!/usr/bin/env python
"""Validate committed notebooks for deterministic, offline CI execution."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
VALID_TAGS = {"fast", "slow", "skip"}

CHECKS = {
    "has_import_ts2eg": re.compile(
        r"^\s*(?:from\s+ts2eg\s+import|from\s+ts2eg\.|import\s+ts2eg)\b",
        re.MULTILINE,
    ),
    "has_seed_numpy": re.compile(
        r"\b(?:np\.random\.seed\(|np\.random\.default_rng\(\s*\d+\s*\))"
    ),
    "has_seed_python": re.compile(r"\brandom\.seed\("),
    "uses_ts2eg_ci": re.compile(r"\bTS2EG_CI\b"),
    "uses_lambda_param": re.compile(r"\blambda_\s*="),
    "uses_np_ptp_func": re.compile(r"\bnp\.ptp\s*\("),
    "uses_attr_ptp": re.compile(r"\.ptp\s*\("),
    "bootstrap_marker": re.compile(r"#\s*Bootstrap\b", re.IGNORECASE),
    "network_import": re.compile(
        r"^\s*(?:"
        r"from\s+(?:aiohttp|ftplib|httpx|pandas_datareader|requests|socket|"
        r"urllib|yfinance)(?:\.|\s)|"
        r"import\s+[^\n#]*\b(?:aiohttp|ftplib|httpx|pandas_datareader|"
        r"requests|socket|urllib|yfinance)\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "network_url": re.compile(r"(?:https?|ftp)://", re.IGNORECASE),
    "network_shell": re.compile(
        r"^\s*!\s*(?:curl|git\s+clone|pip\s+install|wget)\b|"
        r"^\s*%\s*(?:conda|mamba|pip)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
}


def _matches(name: str, source: str) -> bool:
    return bool(CHECKS[name].search(source))


def _cell_source(cell: Dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def audit_notebook(path: Path) -> Dict[str, object]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"
    ]
    source = "\n".join(_cell_source(cell) for cell in code_cells)

    has_bootstrap = any(
        "bootstrap" in ((cell.get("metadata", {}) or {}).get("tags") or [])
        or _matches("bootstrap_marker", _cell_source(cell))
        for cell in code_cells
    )

    attr_ptp_bad = any(
        _matches("uses_attr_ptp", _cell_source(cell))
        and not _matches("uses_np_ptp_func", _cell_source(cell))
        for cell in code_cells
    )

    committed_outputs = sum(len(cell.get("outputs", [])) for cell in code_cells)
    execution_counts = sum(
        cell.get("execution_count") is not None for cell in code_cells
    )
    invalid_python_cells = []
    for index, cell in enumerate(code_cells, start=1):
        try:
            compile(_cell_source(cell), f"{path.as_posix()}#code-cell-{index}", "exec")
        except SyntaxError:
            invalid_python_cells.append(index)
    metadata = nb.get("metadata", {}) or {}

    return {
        "file": path.relative_to(ROOT).as_posix(),
        "ci_tag": metadata.get("ts2eg_ci"),
        "import_ts2eg": _matches("has_import_ts2eg", source),
        "seed_numpy": _matches("has_seed_numpy", source),
        "seed_python": _matches("has_seed_python", source),
        "ts2eg_ci_env": _matches("uses_ts2eg_ci", source),
        "bootstrap": has_bootstrap,
        "lambda_param": _matches("uses_lambda_param", source),
        "np_ptp_func": _matches("uses_np_ptp_func", source),
        "attr_ptp_bad": attr_ptp_bad,
        "internet": any(
            _matches(name, source)
            for name in ("network_import", "network_url", "network_shell")
        ),
        "committed_outputs": committed_outputs,
        "execution_counts": execution_counts,
        "invalid_python_cells": invalid_python_cells,
    }


def validation_errors(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    tag = report["ci_tag"]
    if not isinstance(tag, str) or tag not in VALID_TAGS:
        errors.append(f"metadata.ts2eg_ci must be one of {sorted(VALID_TAGS)}")

    if report["internet"]:
        errors.append("contains a network import, URL, or package/download command")
    if report["lambda_param"]:
        errors.append("uses retired lambda_= keyword")
    if report["attr_ptp_bad"]:
        errors.append("uses ndarray.ptp instead of np.ptp")
    if report["committed_outputs"] or report["execution_counts"]:
        errors.append("contains committed execution counts or outputs")
    if report["invalid_python_cells"]:
        errors.append(
            "contains non-Python code, an IPython magic, or a syntax error in "
            f"code cells {report['invalid_python_cells']}"
        )

    if tag == "fast":
        required = {
            "import_ts2eg": "does not import ts2eg",
            "seed_numpy": "does not set a deterministic NumPy seed",
            "seed_python": "does not set a deterministic Python seed",
            "ts2eg_ci_env": "does not honor TS2EG_CI",
            "bootstrap": "does not contain a tagged or labeled bootstrap cell",
        }
        for field, message in required.items():
            if not report[field]:
                errors.append(message)
    return errors


def main() -> int:
    notebook_paths = sorted(NB_DIR.glob("*.ipynb"))
    if not notebook_paths:
        print("No notebooks found.", file=sys.stderr)
        return 2

    reports = [audit_notebook(path) for path in notebook_paths]
    fields = [
        "file",
        "ci_tag",
        "import_ts2eg",
        "seed_numpy",
        "seed_python",
        "ts2eg_ci_env",
        "bootstrap",
        "lambda_param",
        "np_ptp_func",
        "attr_ptp_bad",
        "internet",
        "committed_outputs",
        "execution_counts",
        "invalid_python_cells",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: report[field] for field in fields} for report in reports)

    failures = []
    for report in reports:
        errors = validation_errors(report)
        if errors:
            failures.append((report["file"], errors))
    if failures:
        print("Notebook audit failed:", file=sys.stderr)
        for path, errors in failures:
            for error in errors:
                print(f"  {path}: {error}", file=sys.stderr)
        return 1

    print(f"Notebook audit passed ({len(reports)} notebooks checked).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
