from __future__ import annotations

import importlib
import json
from pathlib import Path

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.question import QuestionCandidate


def test_duplicate_numbers_and_tempting_wrong_mappings_never_displace_exact_alignment() -> None:
    fixture = json.loads(
        Path("fixtures/gold/task7_answer_matching_precision.json").read_text(encoding="utf-8")
    )
    scoring = importlib.import_module("question_builder.matching.scoring")
    verifier = importlib.import_module("question_builder.matching.verifier")

    questions = tuple(
        QuestionCandidate(
            question_candidate_id=item["id"],
            document_id="doc_question",
            content_blocks=(item["block"],),
            question_number=item["number"],
            question_type_candidate=item["type"],
            split_score=1.0,
        )
        for item in fixture["questions"]
    )
    answers = tuple(
        AnswerCandidate(
            answer_candidate_id=item["id"],
            document_id="doc_answer",
            question_number=item["number"],
            answer=item["answer"],
            analysis="略",
            source_blocks=(item["block"],),
            extract_score=1.0,
        )
        for item in fixture["answers"]
    )
    signals = {
        (item["q"], item["a"]): scoring.MatchSignals(**item["values"])
        for item in fixture["signals"]
    }
    verifier_results = {
        (item["q"], item["a"]): verifier.VerifierResult(
            decision=verifier.VerifierDecision.PASS,
            score=item["score"],
            reason="golden source evidence confirms mapping",
            cited_blocks=tuple(item["cited_blocks"]),
        )
        for item in fixture["verifiers"]
    }

    result = verifier.match_exam_cluster(
        questions,
        answers,
        signals_by_pair=signals,
        verifier_results=verifier_results,
        thresholds=QualityThresholds(),
    )

    accepted = [
        [item.question.question_candidate_id, item.answer.answer_candidate_id]
        for item in result.matched_questions
    ]
    assert result.alignment.ambiguous is False
    assert result.rejections == ()
    assert accepted == fixture["expected"]
