from __future__ import annotations

import importlib
import json
from pathlib import Path

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.recognition.calibration import CalibrationProfile, CalibrationRegistry
from question_builder.recognition.contracts import RecognitionTask
from question_builder.understanding.clustering import ExamCluster


def _fixture() -> dict[str, object]:
    return json.loads(
        Path("fixtures/gold/task7_answer_matching_precision.json").read_text(encoding="utf-8")
    )


def _candidates(
    fixture: dict[str, object],
) -> tuple[tuple[QuestionCandidate, ...], tuple[AnswerCandidate, ...]]:
    question_items = fixture["questions"]
    answer_items = fixture["answers"]
    assert isinstance(question_items, list)
    assert isinstance(answer_items, list)
    questions = tuple(
        QuestionCandidate(
            question_candidate_id=item["id"],
            document_id="doc_question",
            content_blocks=(item["block"],),
            question_number=item["number"],
            question_type_candidate=item["type"],
            split_score=1.0,
        )
        for item in question_items
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
        for item in answer_items
    )
    return questions, answers


def _cluster() -> ExamCluster:
    return ExamCluster(
        cluster_id="golden-task7",
        document_ids=("doc_question", "doc_answer"),
        accepted=True,
        evidence=("golden_accepted_cluster",),
    )


def _registry() -> CalibrationRegistry:
    return CalibrationRegistry(
        (
            CalibrationProfile(
                provider="golden-verifier",
                model="golden-model",
                task=RecognitionTask.LLM,
                version="task7-golden-v1",
            ),
        )
    )


def _signals(scoring, fixture, questions, answers):
    signal_items = fixture["signals"]
    assert isinstance(signal_items, list)
    signals = {
        (item["q"], item["a"]): scoring.MatchSignals(**item["values"])
        for item in signal_items
    }
    for question in questions:
        for answer in answers:
            key = (question.question_candidate_id, answer.answer_candidate_id)
            if key not in signals:
                signals[key] = scoring.MatchSignals(
                    same_cluster=True,
                    cluster_conflict=True,
                    document_relation=0.0,
                    number_consistency=0.0,
                    sequence_consistency=0.0,
                    type_compatibility=0.0,
                    count_compatibility=0.0,
                    answer_format_compatibility=0.0,
                    filename_title_consistency=0.0,
                    semantic_consistency=0.0,
                )
    return signals


def _verifier_results(verifier, fixture):
    verifier_items = fixture["verifiers"]
    assert isinstance(verifier_items, list)
    return {
        (item["q"], item["a"]): verifier.VerifierExecution(
            result=verifier.VerifierResult(
                decision=verifier.VerifierDecision.PASS,
                score=item["score"],
                reason="golden source evidence confirms mapping",
                cited_blocks=tuple(item["cited_blocks"]),
            ),
            provider="golden-verifier",
            model="golden-model",
        )
        for item in verifier_items
    }


def test_duplicate_numbers_and_tempting_wrong_mappings_never_displace_exact_alignment() -> None:
    fixture = _fixture()
    scoring = importlib.import_module("question_builder.matching.scoring")
    verifier = importlib.import_module("question_builder.matching.verifier")
    questions, answers = _candidates(fixture)
    signals = _signals(scoring, fixture, questions, answers)

    result = verifier.match_exam_cluster(
        _cluster(),
        questions,
        answers,
        signals_by_pair=signals,
        verifier_results=_verifier_results(verifier, fixture),
        calibration_registry=_registry(),
        thresholds=QualityThresholds(),
    )

    accepted = [
        [item.question.question_candidate_id, item.answer.answer_candidate_id]
        for item in result.matched_questions
    ]
    assert result.alignment.ambiguous is False
    assert result.rejections == ()
    assert accepted == fixture["expected"]


def test_missing_plausible_wrong_pair_evidence_cannot_manufacture_golden_pass() -> None:
    fixture = _fixture()
    scoring = importlib.import_module("question_builder.matching.scoring")
    verifier = importlib.import_module("question_builder.matching.verifier")
    questions, answers = _candidates(fixture)
    signals = _signals(scoring, fixture, questions, answers)
    signals.pop(("q_a1", "a_b1"))

    result = verifier.match_exam_cluster(
        _cluster(),
        questions,
        answers,
        signals_by_pair=signals,
        verifier_results=_verifier_results(verifier, fixture),
        calibration_registry=_registry(),
        thresholds=QualityThresholds(),
    )

    assert result.matched_questions == ()
    assert result.rejections
    assert all(
        item.reason_code is RejectReason.ANSWER_MATCH_AMBIGUOUS
        for item in result.rejections
    )
    assert any("missing pair evidence" in item.message for item in result.rejections)
