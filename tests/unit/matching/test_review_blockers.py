from __future__ import annotations

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.matching import scoring, verifier
from question_builder.recognition.calibration import CalibrationProfile, CalibrationRegistry
from question_builder.recognition.contracts import RecognitionTask
from question_builder.understanding.clustering import ExamCluster


def _question() -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="q1",
        document_id="question_doc",
        content_blocks=("qb1",),
        question_number="1",
        question_type_candidate="选择题",
        split_score=1.0,
    )


def _answer(answer_id: str, number: str, block_id: str) -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id=answer_id,
        document_id="answer_doc",
        question_number=number,
        answer="A",
        analysis="略",
        source_blocks=(block_id,),
        extract_score=1.0,
    )


def _cluster(
    *,
    accepted: bool = True,
    document_ids: tuple[str, ...] = ("question_doc", "answer_doc"),
) -> ExamCluster:
    return ExamCluster(
        cluster_id="cluster_test",
        document_ids=document_ids,
        accepted=accepted,
        evidence=("test_cluster",),
    )


def _signals(**overrides: object) -> scoring.MatchSignals:
    values: dict[str, object] = {
        "same_cluster": True,
        "cluster_conflict": False,
        "document_relation": 1.0,
        "number_consistency": 1.0,
        "sequence_consistency": 1.0,
        "type_compatibility": 1.0,
        "count_compatibility": 1.0,
        "answer_format_compatibility": 1.0,
        "filename_title_consistency": 1.0,
        "semantic_consistency": 1.0,
    }
    values.update(overrides)
    return scoring.MatchSignals(**values)


def _raw_verifier(score: float, *, answer_block: str = "ab1") -> verifier.VerifierResult:
    return verifier.VerifierResult(
        decision=verifier.VerifierDecision.PASS,
        score=score,
        reason="source evidence supports the mapping",
        cited_blocks=("qb1", answer_block),
    )


def _execution(score: float, *, answer_block: str = "ab1") -> verifier.VerifierExecution:
    return verifier.VerifierExecution(
        result=_raw_verifier(score, answer_block=answer_block),
        provider="verifier-provider",
        model="verifier-model",
    )


def _registry(*, scale: float = 1.0, offset: float = 0.0) -> CalibrationRegistry:
    return CalibrationRegistry(
        (
            CalibrationProfile(
                provider="verifier-provider",
                model="verifier-model",
                task=RecognitionTask.LLM,
                version="task7-test-v1",
                scale=scale,
                offset=offset,
            ),
        )
    )


def _scored(
    question: QuestionCandidate,
    answer: AnswerCandidate,
    score: float = 0.999,
) -> scoring.ScoredMatch:
    return scoring.ScoredMatch(
        question=question,
        answer=answer,
        score=score,
        eligible=True,
        version=scoring.MATCH_SCORE_VERSION,
        evidence={"fixture_score": score},
    )


def test_matcher_rejects_candidate_outside_real_cluster_even_if_signals_claim_same() -> None:
    question = _question()
    answer = _answer("a1", "1", "ab1")

    with pytest.raises(verifier.ClusterContextError):
        verifier.match_exam_cluster(
            _cluster(document_ids=("question_doc",)),
            (question,),
            (answer,),
            signals_by_pair={("q1", "a1"): _signals()},
            verifier_results={("q1", "a1"): _execution(0.999)},
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )


def test_matcher_rejects_unaccepted_exam_cluster_before_matching() -> None:
    question = _question()
    answer = _answer("a1", "1", "ab1")

    with pytest.raises(verifier.ClusterContextError):
        verifier.match_exam_cluster(
            _cluster(accepted=False),
            (question,),
            (answer,),
            signals_by_pair={("q1", "a1"): _signals()},
            verifier_results={("q1", "a1"): _execution(0.999)},
            calibration_registry=_registry(),
            thresholds=QualityThresholds(),
        )


def test_missing_competing_pair_evidence_fails_closed_before_margin_can_accept_top1() -> None:
    question = _question()
    answer1 = _answer("a1", "1", "ab1")
    answer2 = _answer("a2", "1", "ab2")

    result = verifier.match_exam_cluster(
        _cluster(),
        (question,),
        (answer1, answer2),
        signals_by_pair={("q1", "a1"): _signals()},
        verifier_results={("q1", "a1"): _execution(0.999)},
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


def test_explicit_deterministic_hard_conflict_may_exclude_competitor_from_margin() -> None:
    question = _question()
    answer1 = _answer("a1", "1", "ab1")
    answer2 = _answer("a2", "2", "ab2")

    result = verifier.match_exam_cluster(
        _cluster(),
        (question,),
        (answer1, answer2),
        signals_by_pair={
            ("q1", "a1"): _signals(),
            ("q1", "a2"): _signals(cluster_conflict=True),
        },
        verifier_results={("q1", "a1"): _execution(0.999)},
        calibration_registry=_registry(),
        thresholds=QualityThresholds(),
    )

    assert [item.answer.answer_candidate_id for item in result.matched_questions] == ["a1"]


def test_verifier_gate_uses_calibrated_normalized_score_not_raw_model_score() -> None:
    question = _question()
    answer = _answer("a1", "1", "ab1")

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            (_scored(question, answer),),
            _execution(0.999),
            calibration_registry=_registry(offset=-0.005),
            thresholds=QualityThresholds(),
        )

    assert exc.value.reason_code is RejectReason.ANSWER_VERIFICATION_FAILED


def test_approved_normalization_preserves_verifier_provenance() -> None:
    question = _question()
    answer = _answer("a1", "1", "ab1")

    matched = verifier.select_verified_match(
        question,
        (_scored(question, answer),),
        _execution(0.900),
        calibration_registry=_registry(offset=0.097),
        thresholds=QualityThresholds(),
    )

    assert matched.evidence.verifier_score == pytest.approx(0.997)
    assert matched.evidence.evidence["verifier_raw_score"] == pytest.approx(0.900)
    assert matched.evidence.evidence["verifier_normalized_score"] == pytest.approx(0.997)
    assert matched.evidence.evidence["verifier_provider"] == "verifier-provider"
    assert matched.evidence.evidence["verifier_model"] == "verifier-model"
    assert matched.evidence.evidence["verifier_calibration_id"] == (
        "verifier-provider/verifier-model/llm@task7-test-v1"
    )


def test_missing_verifier_calibration_fails_closed() -> None:
    question = _question()
    answer = _answer("a1", "1", "ab1")

    with pytest.raises(verifier.MatchRejected) as exc:
        verifier.select_verified_match(
            question,
            (_scored(question, answer),),
            _execution(0.999),
            calibration_registry=CalibrationRegistry(),
            thresholds=QualityThresholds(),
        )

    assert exc.value.reason_code is RejectReason.ANSWER_VERIFICATION_FAILED
