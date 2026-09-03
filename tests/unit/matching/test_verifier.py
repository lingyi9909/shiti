from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.recognition.calibration import CalibrationProfile, CalibrationRegistry
from question_builder.recognition.contracts import RecognitionTask


def _question() -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="q1",
        document_id="doc_question",
        content_blocks=("qb1", "qb2"),
        question_number="1",
        question_type_candidate="选择题",
        split_score=1.0,
    )


def _answer(answer_id: str, block_id: str) -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id=answer_id,
        document_id="doc_answer",
        question_number="1",
        answer="A",
        analysis="略",
        source_blocks=(block_id,),
        extract_score=1.0,
    )


def _apis():
    scoring = importlib.import_module("question_builder.matching.scoring")
    verifier = importlib.import_module("question_builder.matching.verifier")
    return scoring, verifier


def _scored(scoring, question, answer, score: float):
    return scoring.ScoredMatch(
        question=question,
        answer=answer,
        score=score,
        eligible=True,
        version="answer_match_v1",
        evidence={"fixture_score": score},
    )


def _verifier_execution(
    verifier,
    score: float,
    *,
    decision: str = "PASS",
    cited_blocks: tuple[str, ...] = ("qb1", "ab1"),
):
    return verifier.VerifierExecution(
        result=verifier.VerifierResult(
            decision=verifier.VerifierDecision(decision),
            score=score,
            reason="source evidence supports the mapping",
            cited_blocks=cited_blocks,
        ),
        provider="test-verifier",
        model="test-model",
    )


def _registry() -> CalibrationRegistry:
    return CalibrationRegistry(
        (
            CalibrationProfile(
                provider="test-verifier",
                model="test-model",
                task=RecognitionTask.LLM,
                version="task7-unit-v1",
            ),
        )
    )


def test_verifier_json_contract_is_exact_and_rejects_replacement_answer() -> None:
    _, verifier = _apis()
    valid = json.dumps(
        {
            "decision": "PASS",
            "score": 0.999,
            "reason": "number and source context agree",
            "cited_blocks": ["qb1", "ab1"],
        }
    )

    parsed = verifier.parse_verifier_output(
        valid,
        allowed_source_blocks=frozenset({"qb1", "ab1"}),
    )
    assert parsed.decision is verifier.VerifierDecision.PASS
    assert parsed.score == pytest.approx(0.999)

    invalid = json.dumps(
        {
            "decision": "PASS",
            "score": 0.999,
            "reason": "looks right",
            "cited_blocks": ["qb1", "ab1"],
            "replacement_answer": "B",
        }
    )
    with pytest.raises(verifier.VerifierContractError):
        verifier.parse_verifier_output(
            invalid,
            allowed_source_blocks=frozenset({"qb1", "ab1"}),
        )


def test_verifier_pass_must_cite_both_question_and_answer_source() -> None:
    scoring, verifier = _apis()
    question = _question()
    answer = _answer("a1", "ab1")
    ranked = (_scored(scoring, question, answer, 0.999),)

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            ranked,
            _verifier_execution(verifier, 0.999, cited_blocks=("qb1",)),
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )
    assert exc.value.reason_code is RejectReason.ANSWER_VERIFICATION_FAILED


def test_abstention_accepts_clear_margin_and_high_verifier_score() -> None:
    scoring, verifier = _apis()
    question = _question()
    top = _answer("a1", "ab1")
    second = _answer("a2", "ab2")

    matched = verifier.select_verified_match(
        question,
        (
            _scored(scoring, question, top, 0.996),
            _scored(scoring, question, second, 0.80),
        ),
        _verifier_execution(verifier, 0.997),
        calibration_registry=_registry(),
        thresholds=QualityThresholds(),
    )

    assert matched.answer.answer_candidate_id == "a1"
    assert matched.evidence.match_score == pytest.approx(0.996)
    assert matched.evidence.second_best_score == pytest.approx(0.80)
    assert matched.evidence.verifier_score == pytest.approx(0.997)


def test_abstention_rejects_high_but_close_top_two() -> None:
    scoring, verifier = _apis()
    question = _question()
    top = _answer("a1", "ab1")
    second = _answer("a2", "ab2")

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            (
                _scored(scoring, question, top, 0.996),
                _scored(scoring, question, second, 0.991),
            ),
            _verifier_execution(verifier, 0.999),
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )
    assert exc.value.reason_code is RejectReason.ANSWER_MATCH_AMBIGUOUS


def test_abstention_rejects_below_match_threshold_even_with_large_margin() -> None:
    scoring, verifier = _apis()
    question = _question()
    top = _answer("a1", "ab1")
    second = _answer("a2", "ab2")

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            (
                _scored(scoring, question, top, 0.994),
                _scored(scoring, question, second, 0.20),
            ),
            _verifier_execution(verifier, 0.999),
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )
    assert exc.value.reason_code is RejectReason.ANSWER_MATCH_AMBIGUOUS


def test_abstention_rejects_low_verifier_score() -> None:
    scoring, verifier = _apis()
    question = _question()
    top = _answer("a1", "ab1")

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            (_scored(scoring, question, top, 0.999),),
            _verifier_execution(verifier, 0.994),
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )
    assert exc.value.reason_code is RejectReason.ANSWER_VERIFICATION_FAILED


def test_answer_verify_prompt_forbids_generation_and_has_independent_contract() -> None:
    prompt = Path("prompts/answer_verify/v1.txt").read_text(encoding="utf-8")

    assert "PASS" in prompt and "FAIL" in prompt
    assert "cited_blocks" in prompt
    assert "replacement_answer" in prompt
    assert "禁止" in prompt
    assert "生成答案" in prompt or "修改答案" in prompt
