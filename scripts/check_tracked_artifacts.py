#!/usr/bin/env python
"""Fail when repository-local or generated artifacts are tracked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_COMPONENTS = {
    ".venv": "virtual environment",
    "venv": "virtual environment",
    ".tox": "test environment",
    ".nox": "test environment",
    "__pycache__": "Python bytecode cache",
    ".pytest_cache": "pytest cache",
    ".mypy_cache": "mypy cache",
    ".ruff_cache": "ruff cache",
    ".ipynb_checkpoints": "notebook checkpoint",
    "artifacts": "generated artifact directory",
    "build": "build directory",
    "dist": "distribution directory",
    ".vscode": "editor-local settings",
    "Codex": "historical assistant transcript",
}

FORBIDDEN_ENDINGS = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".lot",
    ".log",
    ".nav",
    ".nbconvert.ipynb",
    ".executed.ipynb",
    ".failed.ipynb",
    ".out",
    ".run.xml",
    ".snm",
    ".synctex",
    ".synctex.gz",
    ".toc",
    ".vrb",
)

FORBIDDEN_EXACT = {
    ".github/workflows/ci-notebooks.yml": "duplicate notebook workflow",
    "A.csv": "generated notebook output",
    "Xk.csv": "generated notebook output",
    "audit_notebooks.csv": "generated audit snapshot",
    "docs/ts2eg_math_background.pdf": "compiled documentation",
    "gitattributes": "inert non-dot configuration",
    "gitignore": "inert non-dot configuration",
    "dev/open-code.sh": "machine-specific launcher",
    "dev/patch_readme_intro.sh": "obsolete README rewriter",
    "scripts/init_git.ps1": "obsolete repository initializer",
    "scripts/init_git.sh": "obsolete repository initializer",
    "scripts/inject_readme_intro_exact.sh": "obsolete README rewriter",
    "scripts/nb_canonical_normalize.py": "obsolete notebook rewriter",
    "scripts/nb_patch_ci.py": "obsolete notebook rewriter",
}


def _git_paths(arguments: Sequence[str]) -> List[str]:
    completed = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [
        os.fsdecode(raw)
        for raw in completed.stdout.split(b"\0")
        if raw
    ]


def forbidden_reasons(path: str) -> Set[str]:
    """Return policy reasons that make a tracked path forbidden."""
    reasons: Set[str] = set()
    exact_reason = FORBIDDEN_EXACT.get(path)
    if exact_reason is not None:
        reasons.add(exact_reason)

    parts = path.split("/")
    for part in parts:
        component_reason = FORBIDDEN_COMPONENTS.get(part)
        if component_reason is not None:
            reasons.add(component_reason)
        if part.endswith(".egg-info"):
            reasons.add("generated package metadata")

    if path.endswith(FORBIDDEN_ENDINGS):
        reasons.add("generated file type")
    return reasons


def collect_violations(
    tracked: Iterable[str], ignored_tracked: Iterable[str]
) -> Dict[str, Set[str]]:
    violations: Dict[str, Set[str]] = {}
    for path in tracked:
        reasons = forbidden_reasons(path)
        if reasons:
            violations.setdefault(path, set()).update(reasons)
    for path in ignored_tracked:
        violations.setdefault(path, set()).add("tracked despite .gitignore")
    return violations


def main() -> int:
    try:
        tracked = _git_paths(["ls-files", "--cached"])
        ignored_tracked = _git_paths(
            ["ls-files", "--cached", "--ignored", "--exclude-standard"]
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"repository hygiene audit could not run: {exc}", file=sys.stderr)
        return 2

    violations = collect_violations(tracked, ignored_tracked)
    if violations:
        print("Tracked-artifact policy violations:", file=sys.stderr)
        for path in sorted(violations):
            reasons = "; ".join(sorted(violations[path]))
            print(f"  {path}: {reasons}", file=sys.stderr)
        return 1

    print(f"Tracked-artifact audit passed ({len(tracked)} paths checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
