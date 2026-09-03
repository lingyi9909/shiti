from __future__ import annotations

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.matching import MatchedQuestion
from question_builder.domain.question import QuestionCandidate
from question_builder.matching import scoring, verifier
from question_builder.recognition.calibration import CalibrationProfile, CalibrationRegistry
from question_builder.recognition.contracts import RecognitionTask


def _question(*, document_id: str = "question_doc") -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="q1",
        document_id=document_id,
        content_blocks=("qb1",),
        question_number="1",
        question_type_candidate="选择题",
        split_score=1.0,
    )


def _answer(answer_id: str, block_id: str) -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id=answer_id,
        document_id="answer_doc",
        question_number="1",
        answer="A",
        analysis="略",
        source_blocks=(block_id,),
        extract_score=1.0,
    )


def _scored(
    question: QuestionCandidate,
    answer: AnswerCandidate,
    score: float,
) -> scoring.ScoredMatch:
    return scoring.ScoredMatch(
        question=question,
        answer=answer,
        score=score,
        eligible=True,
        version=scoring.MATCH_SCORE_VERSION,
        evidence={"fixture_score": score},
    )


def _execution(score: float, *, answer_block: str = "ab1") -> verifier.VerifierExecution:
    return verifier.VerifierExecution(
        result=verifier.VerifierResult(
            decision=verifier.VerifierDecision.PASS,
            score=score,
            reason="source evidence supports the mapping",
            cited_blocks=("qb1", answer_block),
        ),
        provider="verifier-provider",
        model="verifier-model",
    )


def _registry() -> CalibrationRegistry:
    return CalibrationRegistry(
        (
            CalibrationProfile(
                provider="verifier-provider",
                model="verifier-model",
                task=RecognitionTask.LLM,
                version="task7-path-v1",
            ),
        )
    )


def _invoke_legacy_select(
    question: QuestionCandidate,
    ranked: tuple[scoring.ScoredMatch, ...],
) -> object | None:
    select = getattr(verifier, "select_verified_match", None)
    if select is None:
        return None
    return select(
        question,
        ranked,
        _execution(0.999),
        calibration_registry=_registry(),
        thresholds=QualityThresholds(),
    )


def test_out_of_cluster_manual_scored_match_cannot_produce_formal_matched_question() -> None:
    question = _question(document_id="outside_cluster_doc")
    answer = _answer("a1", "ab1")

    result = _invoke_legacy_select(
        question,
        (_scored(question, answer, 0.999),),
    )

    assert not isinstance(result, MatchedQuestion)


def test_partial_ranked_subset_cannot_bypass_competing_answer_margin() -> None:
    question = _question()
    answer1 = _answer("a1", "ab1")
    _answer2 = _answer("a2", "ab2")

    result = _invoke_legacy_select(
        question,
        (_scored(question, answer1, 0.999),),
    )

    assert not isinstance(result, MatchedQuestion)
