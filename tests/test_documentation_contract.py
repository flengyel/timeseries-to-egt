from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_primary_docs_do_not_present_ess_as_calibrated_inference():
    paths = [
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "ts2eg_math_background.tex",
    ]
    banned_phrases = [
        "ess " + "p-value",
        "ess " + "indicator",
        "reject " + "h0",
        "reject $h_0$",
        "surrogate " + "significance",
        "evidence for " + "strategic interaction",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned_phrases:
            assert phrase not in text, f"{path} still contains {phrase!r}"
        assert "uncalibrated" in text
        assert "descriptive" in text
        assert "no replacement inferential test" in text
