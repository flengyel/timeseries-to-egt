# Repository artifact policy

This repository tracks reviewable source material and reproducible fixtures. It
does not track local environments, caches, or products that can be regenerated
from committed sources.

## Tracked source

The following material belongs in the repository:

- Python package source, tests, examples, and maintenance scripts;
- Markdown documentation, the authoritative LaTeX source, and active or
  completed execution plans;
- notebooks with explicit `ts2eg_ci` metadata and no committed execution
  counts or outputs;
- CI configuration and portable project configuration; and
- small, deterministic test fixtures with their provenance documented. Such
  fixtures belong under `tests/fixtures/`, rather than among generated
  notebook outputs.

`data/README.md` and `data/.gitkeep` are retained to document the optional local
data location. The local data formats listed in `.gitignore` remain untracked.

## Generated and local material

The following material must remain untracked:

- virtual environments, installed packages, Python bytecode, test caches,
  coverage output, and build products;
- notebook checkpoints and executed or failed notebook artifacts;
- generated CSV output, logs, TeX auxiliary files, and compiled documentation;
- editor-specific settings and machine-specific launchers; and
- one-off repository initialization, notebook-rewriting, and README-injection
  scripts.

The source `docs/ts2eg_math_background.tex` is authoritative. No compiled PDF
is currently designated as a tracked release artifact. A future compiled
document may be tracked only after this policy names it explicitly and the
repository supplies a reproducible build and a source-drift check.

Generated notebook results belong under `artifacts/`, which is ignored. CI may
upload that directory as an ephemeral workflow artifact.

## Notebook execution

CI executes the committed Python cells directly and in order, with one isolated
subprocess per notebook. The runner may set process environment variables, but
it must not inject, reorder, or rewrite source cells. It does not require a
Jupyter kernel or local network socket. Fast notebooks must be deterministic,
self-contained, headless, offline, and valid as plain Python cells. Each
notebook declares one of `fast`, `slow`, or `skip` in `metadata.ts2eg_ci`;
missing or unknown values are errors.

## Historical material

Historical Codex prompts and one-off development helpers were removed from the
current tree because they contain obsolete instructions and are not project
documentation. Git history preserves them. The cleanup intentionally does not
rewrite history, so previously committed generated files and environments also
remain recoverable from older commits.

Active implementation instructions belong in `AGENTS.md` and
`docs/exec-plans/`. A task file moves to `docs/exec-plans/completed/` only when
its acceptance conditions have been met.

## Enforcement

Run the tracked-artifact audit from the repository root:

```bash
python scripts/check_tracked_artifacts.py
```

The audit checks both files that are simultaneously ignored and tracked and an
independent set of forbidden path classes. The independent rules ensure that
weakening `.gitignore` cannot silently permit virtual environments, bytecode,
or generated TeX products.
