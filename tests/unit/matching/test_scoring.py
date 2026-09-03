from __future__ import annotations

import importlib

import pytest

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.question import QuestionCandidate


def _question() -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="q1",
        document_id="doc_question",
        content_blocks=("qb1",),
        question_number="1",
        question_type_candidate="选择题",
        split_score=1.0,
    )


def _answer() -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id="a1",
        document_id="doc_answer",
        question_number="1",
        answer="A",
        analysis="略",
        source_blocks=("ab1",),
        extract_score=1.0,
    )


def _scoring_api():
    module = importlib.import_module("question_builder.matching.scoring")
    return (
        getattr(module, "MatchSignals"),
        getattr(module, "score_pair"),
        getattr(module, "MATCH_SCORE_VERSION"),
    )


def _signals(MatchSignals, **overrides):
    values = {
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
    return MatchSignals(**values)


def test_perfect_same_cluster_evidence_scores_one_and_records_all_components() -> None:
    MatchSignals, score_pair, version = _scoring_api()

    scored = score_pair(_question(), _answer(), _signals(MatchSignals))

    assert scored.eligible is True
    assert scored.score == pytest.approx(1.0)
    assert scored.version == version == "answer_match_v1"
    assert scored.evidence["same_cluster"] is True
    for key in (
        "document_relation",
        "number_consistency",
        "sequence_consistency",
        "type_compatibility",
        "count_compatibility",
        "answer_format_compatibility",
        "filename_title_consistency",
        "semantic_consistency",
    ):
        assert scored.evidence[key] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("same_cluster", "cluster_conflict", "expected_reason"),
    [
        (False, False, "cross_cluster_disabled"),
        (True, True, "cluster_conflict"),
    ],
)
def test_cross_cluster_or_conflicting_cluster_evidence_is_ineligible(
    same_cluster: bool,
    cluster_conflict: bool,
    expected_reason: str,
) -> None:
    MatchSignals, score_pair, _ = _scoring_api()

    scored = score_pair(
        _question(),
        _answer(),
        _signals(
            MatchSignals,
            same_cluster=same_cluster,
            cluster_conflict=cluster_conflict,
        ),
    )

    assert scored.eligible is False
    assert scored.score == 0.0
    assert scored.evidence["ineligible_reason"] == expected_reason


def test_same_number_alone_cannot_reach_acceptance_score() -> None:
    MatchSignals, score_pair, _ = _scoring_api()

    scored = score_pair(
        _question(),
        _answer(),
        _signals(
            MatchSignals,
            document_relation=0.0,
            sequence_consistency=0.0,
            type_compatibility=0.0,
            count_compatibility=0.0,
            answer_format_compatibility=0.0,
            filename_title_consistency=0.0,
            semantic_consistency=0.0,
        ),
    )

    assert scored.eligible is True
    assert scored.score < 0.995
