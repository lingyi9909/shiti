from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.matching import MatchEvidence, MatchedQuestion
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.matching.alignment import AlignmentResult, align_sequences
from question_builder.matching.scoring import MATCH_SCORE_VERSION, MatchSignals, ScoredMatch, score_pair


class VerifierDecision(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class VerifierResult:
    decision: VerifierDecision
    score: float
    reason: str
    cited_blocks: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not 0.0 <= self.score <= 1.0:
            raise ValueError("verifier score must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("verifier reason must be non-empty")
        if not self.cited_blocks or any(not block_id for block_id in self.cited_blocks):
            raise ValueError("verifier cited_blocks must contain non-empty block ids")
        if len(self.cited_blocks) != len(set(self.cited_blocks)):
            raise ValueError("verifier cited_blocks must be unique")


class VerifierContractError(ValueError):
    pass


class MatchRejected(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class MatchRejection:
    question_candidate_id: str
    reason_code: RejectReason
    message: str


@dataclass(frozen=True, slots=True)
class ExamMatchResult:
    alignment: AlignmentResult
    matched_questions: tuple[MatchedQuestion, ...]
    rejections: tuple[MatchRejection, ...]
    unmatched_question_ids: tuple[str, ...]
    unmatched_answer_ids: tuple[str, ...]


def parse_verifier_output(
    payload: str,
    *,
    allowed_source_blocks: frozenset[str],
) -> VerifierResult:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerifierContractError("verifier output must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise VerifierContractError("verifier output must be a JSON object")
    allowed_keys = {"decision", "score", "reason", "cited_blocks"}
    if set(parsed) != allowed_keys:
        raise VerifierContractError(
            "verifier output may contain only decision, score, reason, cited_blocks"
        )

    raw_decision = parsed.get("decision")
    try:
        decision = VerifierDecision(raw_decision)
    except (TypeError, ValueError) as exc:
        raise VerifierContractError("decision must be PASS or FAIL") from exc

    raw_score = parsed.get("score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise VerifierContractError("score must be a number between 0 and 1")
    score = float(raw_score)
    if not 0.0 <= score <= 1.0:
        raise VerifierContractError("score must be a number between 0 and 1")

    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise VerifierContractError("reason must be a non-empty string")

    raw_cited_blocks = parsed.get("cited_blocks")
    if not isinstance(raw_cited_blocks, list) or not raw_cited_blocks:
        raise VerifierContractError("cited_blocks must be a non-empty list")
    cited_blocks: list[str] = []
    for block_id in raw_cited_blocks:
        if not isinstance(block_id, str) or not block_id:
            raise VerifierContractError("cited_blocks must contain non-empty strings")
        if block_id not in allowed_source_blocks:
            raise VerifierContractError(f"unknown cited source block: {block_id}")
        cited_blocks.append(block_id)
    if len(cited_blocks) != len(set(cited_blocks)):
        raise VerifierContractError("cited_blocks must not contain duplicates")

    return VerifierResult(
        decision=decision,
        score=score,
        reason=reason.strip(),
        cited_blocks=tuple(cited_blocks),
    )


def _eligible_ranked_matches(
    question: QuestionCandidate,
    ranked_matches: Sequence[ScoredMatch],
) -> tuple[ScoredMatch, ...]:
    if not ranked_matches:
        raise MatchRejected(
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            "no answer match candidates are available",
        )
    for item in ranked_matches:
        if item.question.question_candidate_id != question.question_candidate_id:
            raise ValueError("ranked matches must all reference the target question")
    eligible = tuple(
        sorted(
            (item for item in ranked_matches if item.eligible),
            key=lambda item: item.score,
            reverse=True,
        )
    )
    if not eligible:
        raise MatchRejected(
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            "no eligible same-cluster answer match remains",
        )
    return eligible


def _verifier_cites_both_sides(
    question: QuestionCandidate,
    answer: AnswerCandidate,
    verifier_result: VerifierResult,
) -> bool:
    cited = set(verifier_result.cited_blocks)
    return bool(cited & set(question.content_blocks)) and bool(cited & set(answer.source_blocks))


def select_verified_match(
    question: QuestionCandidate,
    ranked_matches: Sequence[ScoredMatch],
    verifier_result: VerifierResult,
    *,
    thresholds: QualityThresholds,
) -> MatchedQuestion:
    eligible = _eligible_ranked_matches(question, ranked_matches)
    top = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None

    if top.score < thresholds.answer_match_accept:
        raise MatchRejected(
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            f"top answer match score {top.score:.6f} is below acceptance threshold",
        )
    if second is not None and top.score - second.score < thresholds.answer_match_margin:
        raise MatchRejected(
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            "top answer candidates do not satisfy the required score margin",
        )

    if (
        verifier_result.decision is not VerifierDecision.PASS
        or verifier_result.score < thresholds.answer_verify_accept
        or not _verifier_cites_both_sides(question, top.answer, verifier_result)
    ):
        raise MatchRejected(
            RejectReason.ANSWER_VERIFICATION_FAILED,
            "independent answer verifier did not provide sufficient source-backed confirmation",
        )

    evidence: dict[str, float | str | bool] = dict(top.evidence)
    evidence["verifier_decision"] = verifier_result.decision.value
    evidence["verifier_reason"] = verifier_result.reason
    evidence["verifier_source_backed"] = True
    evidence["match_score_version"] = top.version

    match_evidence = MatchEvidence(
        question_candidate_id=question.question_candidate_id,
        answer_candidate_id=top.answer.answer_candidate_id,
        match_score=top.score,
        second_best_score=second.score if second is not None else None,
        verifier_score=verifier_result.score,
        question_source_blocks=question.content_blocks,
        answer_source_blocks=top.answer.source_blocks,
        evidence=evidence,
    )
    return MatchedQuestion(question=question, answer=top.answer, evidence=match_evidence)


def _missing_pair_score(
    question: QuestionCandidate,
    answer: AnswerCandidate,
) -> ScoredMatch:
    return ScoredMatch(
        question=question,
        answer=answer,
        score=0.0,
        eligible=False,
        version=MATCH_SCORE_VERSION,
        evidence={
            "ineligible_reason": "missing_pair_evidence",
            "score_version": MATCH_SCORE_VERSION,
        },
    )


def _rank_question_answers(
    question: QuestionCandidate,
    answers: Sequence[AnswerCandidate],
    signals_by_pair: Mapping[tuple[str, str], MatchSignals],
) -> tuple[ScoredMatch, ...]:
    scored: list[ScoredMatch] = []
    for answer in answers:
        key = (question.question_candidate_id, answer.answer_candidate_id)
        signals = signals_by_pair.get(key)
        if signals is None:
            scored.append(_missing_pair_score(question, answer))
        else:
            scored.append(score_pair(question, answer, signals))
    return tuple(sorted(scored, key=lambda item: item.score, reverse=True))


def match_exam_cluster(
    questions: Sequence[QuestionCandidate],
    answers: Sequence[AnswerCandidate],
    *,
    signals_by_pair: Mapping[tuple[str, str], MatchSignals],
    verifier_results: Mapping[tuple[str, str], VerifierResult],
    thresholds: QualityThresholds,
) -> ExamMatchResult:
    alignment = align_sequences(questions, answers)
    if alignment.ambiguous:
        return ExamMatchResult(
            alignment=alignment,
            matched_questions=(),
            rejections=tuple(
                MatchRejection(
                    question_candidate_id=question.question_candidate_id,
                    reason_code=RejectReason.ANSWER_MATCH_AMBIGUOUS,
                    message="sequence alignment has multiple equally optimal mappings",
                )
                for question in questions
            ),
            unmatched_question_ids=tuple(
                question.question_candidate_id for question in questions
            ),
            unmatched_answer_ids=tuple(answer.answer_candidate_id for answer in answers),
        )

    matched_questions: list[MatchedQuestion] = []
    rejections: list[MatchRejection] = []
    for question_index, answer_index in alignment.matched_pairs:
        question = questions[question_index]
        aligned_answer = answers[answer_index]
        ranked = _rank_question_answers(question, answers, signals_by_pair)
        eligible = tuple(item for item in ranked if item.eligible)
        if not eligible or eligible[0].answer.answer_candidate_id != aligned_answer.answer_candidate_id:
            rejections.append(
                MatchRejection(
                    question_candidate_id=question.question_candidate_id,
                    reason_code=RejectReason.ANSWER_MATCH_AMBIGUOUS,
                    message="sequence alignment conflicts with highest supported answer evidence",
                )
            )
            continue

        pair_key = (question.question_candidate_id, aligned_answer.answer_candidate_id)
        verifier_result = verifier_results.get(pair_key)
        if verifier_result is None:
            rejections.append(
                MatchRejection(
                    question_candidate_id=question.question_candidate_id,
                    reason_code=RejectReason.ANSWER_VERIFICATION_FAILED,
                    message="independent verifier evidence is missing for aligned answer",
                )
            )
            continue

        try:
            matched_questions.append(
                select_verified_match(
                    question,
                    ranked,
                    verifier_result,
                    thresholds=thresholds,
                )
            )
        except MatchRejected as exc:
            rejections.append(
                MatchRejection(
                    question_candidate_id=question.question_candidate_id,
                    reason_code=exc.reason_code,
                    message=str(exc),
                )
            )

    unmatched_question_ids = tuple(
        questions[index].question_candidate_id for index in alignment.skipped_questions
    )
    unmatched_answer_ids = tuple(
        answers[index].answer_candidate_id for index in alignment.skipped_answers
    )
    return ExamMatchResult(
        alignment=alignment,
        matched_questions=tuple(matched_questions),
        rejections=tuple(rejections),
        unmatched_question_ids=unmatched_question_ids,
        unmatched_answer_ids=unmatched_answer_ids,
    )
