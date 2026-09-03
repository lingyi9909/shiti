from __future__ import annotations

from pathlib import Path


def test_task7_matching_modules_and_verifier_prompt_exist() -> None:
    required = (
        Path("src/question_builder/matching/alignment.py"),
        Path("src/question_builder/matching/scoring.py"),
        Path("src/question_builder/matching/verifier.py"),
        Path("prompts/answer_verify/v1.txt"),
    )

    missing = [str(path) for path in required if not path.exists()]
    assert missing == []
