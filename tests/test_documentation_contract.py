import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CURRENT_DOCUMENTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "data" / "README.md",
    REPOSITORY_ROOT / "docs" / "ARCHITECTURE.md",
    REPOSITORY_ROOT / "docs" / "BIOLOGY.md",
    REPOSITORY_ROOT / "docs" / "CAPITAL_LABOR.md",
    REPOSITORY_ROOT / "docs" / "CYBERSECURITY.md",
    REPOSITORY_ROOT / "docs" / "FINANCE.md",
    REPOSITORY_ROOT / "docs" / "REPOSITORY_POLICY.md",
    REPOSITORY_ROOT / "docs" / "SOCIOLOGY.md",
    REPOSITORY_ROOT / "docs" / "design" / "STATISTICAL_CONTRACT.md",
    REPOSITORY_ROOT / "docs" / "design" / "VALIDATION_MATRIX.md",
    REPOSITORY_ROOT / "docs" / "exec-plans" / "TS2EG_V2_REDESIGN.md",
    REPOSITORY_ROOT / "docs" / "ts2eg_math_background.tex",
)

CURRENT_SCRIPTS = (
    REPOSITORY_ROOT / "scripts" / "audit_notebooks.py",
    REPOSITORY_ROOT / "scripts" / "check_tracked_artifacts.py",
    REPOSITORY_ROOT / "scripts" / "run_notebooks.py",
)


def _notebook_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _current_public_sources():
    for path in (*CURRENT_DOCUMENTS, *CURRENT_SCRIPTS):
        yield path, path.read_text(encoding="utf-8")
    for path in sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb")):
        yield path, _notebook_text(path)


def test_current_sources_do_not_restore_obsolete_names_or_inferential_claims():
    banned_phrases = (
        "gameify_" + "timeseries",
        "gamify-" + "timeseries",
        "ess " + "p-value",
        "ess " + "indicator",
        "reject " + "h0",
        "reject " + "h_0",
        "surrogate " + "significance",
        "evidence for " + "strategic interaction",
        "held-out " + "r2",
        "held out " + "r2",
        "r2 (" + "held-out)",
    )

    for path, source in _current_public_sources():
        lowered = source.lower()
        for phrase in banned_phrases:
            assert phrase not in lowered, f"{path} still contains {phrase!r}"


def test_readme_states_the_actual_legacy_scope_and_unimplemented_repair():
    readme = _normalized(
        (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    )

    required = (
        "scientifically flawed as an empirical detector",
        "no validated capability to detect",
        "repair is specified",
        "motion estimator is not implemented",
        "contemporaneous payoff signals",
        "not a motion estimator",
        "in-sample zero-field score",
        "discard the aggregate channel",
        "right-centering is not an identification normalization",
        "no replacement inferential procedure",
    )
    for phrase in required:
        assert phrase in readme


def test_every_notebook_disclaims_detection_and_motion_estimation():
    notebooks = sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks

    for path in notebooks:
        text = _notebook_text(path).lower()
        assert "legacy compatibility" in text, path
        assert "no validated detection capability" in text, path
        assert "not a mixture-motion estimator" in text, path
        if "r2" in text:
            assert "in-sample" in text, path


def test_capital_labor_documents_the_exact_row_shift_degeneracy():
    text = _normalized(
        (REPOSITORY_ROOT / "docs" / "CAPITAL_LABOR.md").read_text(
            encoding="utf-8"
        )
    )

    required = (
        "circular shifts preserve row means",
        "every surrogate satisfies",
        "point mass at the observed statistic",
        "supplies no null distribution",
        "each row is constant in time",
        "circular shift leaves the entire payoff array unchanged",
        "no observed surplus-capture test is validated",
    )
    for phrase in required:
        assert phrase in text

    compact = text.replace(" ", "")
    assert "v_c^z(t)-v_l^z(t)" in compact
    assert "v_c(t)-v_l(t)" in compact

    firm_notebook = _normalized(
        _notebook_text(REPOSITORY_ROOT / "notebooks" / "FIRM_DEMO.ipynb")
    )
    assert "supplies no null distribution or test" in firm_notebook
    assert "cannot create a null distribution" in firm_notebook
    for obsolete_statistic in ("p_" + "value", "z_" + "score", "ess_" + "p"):
        assert obsolete_statistic not in firm_notebook


def test_obsolete_document_and_notebook_rewriters_are_absent():
    obsolete_paths = (
        REPOSITORY_ROOT / "dev" / "patch_readme_intro.sh",
        REPOSITORY_ROOT / "scripts" / "inject_readme_intro_exact.sh",
        REPOSITORY_ROOT / "scripts" / "nb_canonical_normalize.py",
        REPOSITORY_ROOT / "scripts" / "nb_patch_ci.py",
    )
    for path in obsolete_paths:
        assert not path.exists(), path

    hygiene_audit = (CURRENT_SCRIPTS[1]).read_text(encoding="utf-8")
    for path in obsolete_paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        assert relative in hygiene_audit
