from __future__ import annotations

import importlib

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.question import QuestionCandidate


def _question(index: int, number: str | None) -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id=f"q{index}",
        document_id="doc_question",
        content_blocks=(f"qb{index}",),
        question_number=number,
        question_type_candidate="选择题",
        split_score=1.0,
    )


def _answer(index: int, number: str | None) -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id=f"a{index}",
        document_id="doc_answer",
        question_number=number,
        answer="A",
        analysis="略",
        source_blocks=(f"ab{index}",),
        extract_score=1.0,
    )


def _alignment_api():
    module = importlib.import_module("question_builder.matching.alignment")
    return getattr(module, "align_sequences"), getattr(module, "AlignmentOp")


def test_perfect_sequence_aligns_every_pair_without_skips() -> None:
    align_sequences, AlignmentOp = _alignment_api()
    questions = tuple(_question(index, str(index)) for index in range(1, 4))
    answers = tuple(_answer(index, str(index)) for index in range(1, 4))

    result = align_sequences(questions, answers)

    assert result.matched_pairs == ((0, 0), (1, 1), (2, 2))
    assert result.skipped_questions == ()
    assert result.skipped_answers == ()
    assert result.ambiguous is False
    assert tuple(step.operation for step in result.steps) == (
        AlignmentOp.MATCH,
        AlignmentOp.MATCH,
        AlignmentOp.MATCH,
    )


def test_missing_answer_does_not_shift_following_question() -> None:
    align_sequences, _ = _alignment_api()
    questions = tuple(_question(index, str(index)) for index in range(1, 5))
    answers = (
        _answer(1, "1"),
        _answer(2, "2"),
        _answer(4, "4"),
    )

    result = align_sequences(questions, answers)

    assert result.matched_pairs == ((0, 0), (1, 1), (3, 2))
    assert result.skipped_questions == (2,)
    assert result.skipped_answers == ()
    assert result.ambiguous is False


def test_duplicate_number_with_two_equally_good_positions_is_ambiguous() -> None:
    align_sequences, _ = _alignment_api()
    questions = (_question(1, "1"), _question(2, "1"))
    answers = (_answer(1, "1"),)

    result = align_sequences(questions, answers)

    assert result.ambiguous is True
    assert len(result.matched_pairs) == 1
    assert len(result.skipped_questions) == 1


def test_inserted_unnumbered_commentary_is_skipped_without_shifting() -> None:
    align_sequences, AlignmentOp = _alignment_api()
    questions = (_question(1, "1"), _question(2, "2"))
    answers = (
        _answer(1, "1"),
        _answer(99, None),
        _answer(2, "2"),
    )

    result = align_sequences(questions, answers)

    assert result.matched_pairs == ((0, 0), (1, 2))
    assert result.skipped_answers == (1,)
    assert any(
        step.operation is AlignmentOp.SKIP_ANSWER and step.answer_index == 1
        for step in result.steps
    )
    assert result.ambiguous is False
