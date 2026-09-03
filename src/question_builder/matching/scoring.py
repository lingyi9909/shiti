from __future__ import annotations

from dataclasses import dataclass

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.question import QuestionCandidate

MATCH_SCORE_VERSION = "answer_match_v1"

_WEIGHTS = {
    "same_cluster": 0.22,
    "document_relation": 0.08,
    "number_consistency": 0.18,
    "sequence_consistency": 0.18,
    "type_compatibility": 0.06,
    "count_compatibility": 0.06,
    "answer_format_compatibility": 0.06,
    "filename_title_consistency": 0.06,
    "semantic_consistency": 0.10,
}


@dataclass(frozen=True, slots=True)
class MatchSignals:
    same_cluster: bool
    cluster_conflict: bool
    document_relation: float
    number_consistency: float
    sequence_consistency: float
    type_compatibility: float
    count_compatibility: float
    answer_format_compatibility: float
    filename_title_consistency: float
    semantic_consistency: float

    def __post_init__(self) -> None:
        for name in (
            "document_relation",
            "number_consistency",
            "sequence_consistency",
            "type_compatibility",
            "count_compatibility",
            "answer_format_compatibility",
            "filename_title_consistency",
            "semantic_consistency",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a number between 0 and 1")


@dataclass(frozen=True, slots=True)
class ScoredMatch:
    question: QuestionCandidate
    answer: AnswerCandidate
    score: float
    eligible: bool
    version: str
    evidence: dict[str, float | str | bool]

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if not self.version:
            raise ValueError("version must be non-empty")


def _evidence(signals: MatchSignals) -> dict[str, float | str | bool]:
    return {
        "same_cluster": signals.same_cluster,
        "cluster_conflict": signals.cluster_conflict,
        "document_relation": signals.document_relation,
        "number_consistency": signals.number_consistency,
        "sequence_consistency": signals.sequence_consistency,
        "type_compatibility": signals.type_compatibility,
        "count_compatibility": signals.count_compatibility,
        "answer_format_compatibility": signals.answer_format_compatibility,
        "filename_title_consistency": signals.filename_title_consistency,
        "semantic_consistency": signals.semantic_consistency,
        "score_version": MATCH_SCORE_VERSION,
    }


def score_pair(
    question: QuestionCandidate,
    answer: AnswerCandidate,
    signals: MatchSignals,
) -> ScoredMatch:
    evidence = _evidence(signals)
    if signals.cluster_conflict:
        evidence["ineligible_reason"] = "cluster_conflict"
        return ScoredMatch(
            question=question,
            answer=answer,
            score=0.0,
            eligible=False,
            version=MATCH_SCORE_VERSION,
            evidence=evidence,
        )
    if not signals.same_cluster:
        evidence["ineligible_reason"] = "cross_cluster_disabled"
        return ScoredMatch(
            question=question,
            answer=answer,
            score=0.0,
            eligible=False,
            version=MATCH_SCORE_VERSION,
            evidence=evidence,
        )

    score = (
        _WEIGHTS["same_cluster"]
        + _WEIGHTS["document_relation"] * signals.document_relation
        + _WEIGHTS["number_consistency"] * signals.number_consistency
        + _WEIGHTS["sequence_consistency"] * signals.sequence_consistency
        + _WEIGHTS["type_compatibility"] * signals.type_compatibility
        + _WEIGHTS["count_compatibility"] * signals.count_compatibility
        + _WEIGHTS["answer_format_compatibility"] * signals.answer_format_compatibility
        + _WEIGHTS["filename_title_consistency"] * signals.filename_title_consistency
        + _WEIGHTS["semantic_consistency"] * signals.semantic_consistency
    )
    normalized_score = min(1.0, max(0.0, score))
    evidence["weighted_score"] = normalized_score
    return ScoredMatch(
        question=question,
        answer=answer,
        score=normalized_score,
        eligible=True,
        version=MATCH_SCORE_VERSION,
        evidence=evidence,
    )
