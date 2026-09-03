from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isclose

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.question import QuestionCandidate

_MATCH_EQUAL = 4.0
_MATCH_BOTH_UNNUMBERED = 1.0
_MATCH_ONE_UNNUMBERED = -2.0
_MATCH_MISMATCH = -4.0
_SKIP_QUESTION = -1.5
_SKIP_ANSWER = -1.5
_EPSILON = 1e-12


class AlignmentOp(StrEnum):
    MATCH = "MATCH"
    SKIP_QUESTION = "SKIP_QUESTION"
    SKIP_ANSWER = "SKIP_ANSWER"


@dataclass(frozen=True, slots=True)
class AlignmentStep:
    operation: AlignmentOp
    question_index: int | None
    answer_index: int | None
    question_number: str | None
    answer_number: str | None
    local_score: float


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    steps: tuple[AlignmentStep, ...]
    matched_pairs: tuple[tuple[int, int], ...]
    skipped_questions: tuple[int, ...]
    skipped_answers: tuple[int, ...]
    score: float
    ambiguous: bool


def _match_score(question_number: str | None, answer_number: str | None) -> float:
    if question_number is None and answer_number is None:
        return _MATCH_BOTH_UNNUMBERED
    if question_number is None or answer_number is None:
        return _MATCH_ONE_UNNUMBERED
    if question_number == answer_number:
        return _MATCH_EQUAL
    return _MATCH_MISMATCH


def _best_score(candidates: tuple[float, ...]) -> float:
    return max(candidates)


def _is_best(value: float, best: float) -> bool:
    return isclose(value, best, rel_tol=0.0, abs_tol=_EPSILON)


def align_sequences(
    questions: Sequence[QuestionCandidate],
    answers: Sequence[AnswerCandidate],
) -> AlignmentResult:
    question_count = len(questions)
    answer_count = len(answers)

    scores = [[float("-inf")] * (answer_count + 1) for _ in range(question_count + 1)]
    path_counts = [[0] * (answer_count + 1) for _ in range(question_count + 1)]
    choices: list[list[AlignmentOp | None]] = [
        [None] * (answer_count + 1) for _ in range(question_count + 1)
    ]

    scores[0][0] = 0.0
    path_counts[0][0] = 1
    for question_index in range(1, question_count + 1):
        scores[question_index][0] = scores[question_index - 1][0] + _SKIP_QUESTION
        path_counts[question_index][0] = 1
        choices[question_index][0] = AlignmentOp.SKIP_QUESTION
    for answer_index in range(1, answer_count + 1):
        scores[0][answer_index] = scores[0][answer_index - 1] + _SKIP_ANSWER
        path_counts[0][answer_index] = 1
        choices[0][answer_index] = AlignmentOp.SKIP_ANSWER

    preference = (
        AlignmentOp.MATCH,
        AlignmentOp.SKIP_QUESTION,
        AlignmentOp.SKIP_ANSWER,
    )
    for question_index in range(1, question_count + 1):
        for answer_index in range(1, answer_count + 1):
            question = questions[question_index - 1]
            answer = answers[answer_index - 1]
            candidate_scores = {
                AlignmentOp.MATCH: scores[question_index - 1][answer_index - 1]
                + _match_score(question.question_number, answer.question_number),
                AlignmentOp.SKIP_QUESTION: scores[question_index - 1][answer_index]
                + _SKIP_QUESTION,
                AlignmentOp.SKIP_ANSWER: scores[question_index][answer_index - 1]
                + _SKIP_ANSWER,
            }
            best = _best_score(tuple(candidate_scores.values()))
            scores[question_index][answer_index] = best

            optimal_ops = tuple(
                operation
                for operation in preference
                if _is_best(candidate_scores[operation], best)
            )
            choices[question_index][answer_index] = optimal_ops[0]

            total_paths = 0
            for operation in optimal_ops:
                if operation is AlignmentOp.MATCH:
                    previous_paths = path_counts[question_index - 1][answer_index - 1]
                elif operation is AlignmentOp.SKIP_QUESTION:
                    previous_paths = path_counts[question_index - 1][answer_index]
                else:
                    previous_paths = path_counts[question_index][answer_index - 1]
                total_paths = min(2, total_paths + previous_paths)
            path_counts[question_index][answer_index] = total_paths

    steps_reversed: list[AlignmentStep] = []
    question_index = question_count
    answer_index = answer_count
    while question_index > 0 or answer_index > 0:
        operation = choices[question_index][answer_index]
        if operation is None:
            raise RuntimeError("alignment backtrace is incomplete")

        if operation is AlignmentOp.MATCH:
            question = questions[question_index - 1]
            answer = answers[answer_index - 1]
            steps_reversed.append(
                AlignmentStep(
                    operation=operation,
                    question_index=question_index - 1,
                    answer_index=answer_index - 1,
                    question_number=question.question_number,
                    answer_number=answer.question_number,
                    local_score=_match_score(question.question_number, answer.question_number),
                )
            )
            question_index -= 1
            answer_index -= 1
        elif operation is AlignmentOp.SKIP_QUESTION:
            question = questions[question_index - 1]
            steps_reversed.append(
                AlignmentStep(
                    operation=operation,
                    question_index=question_index - 1,
                    answer_index=None,
                    question_number=question.question_number,
                    answer_number=None,
                    local_score=_SKIP_QUESTION,
                )
            )
            question_index -= 1
        else:
            answer = answers[answer_index - 1]
            steps_reversed.append(
                AlignmentStep(
                    operation=operation,
                    question_index=None,
                    answer_index=answer_index - 1,
                    question_number=None,
                    answer_number=answer.question_number,
                    local_score=_SKIP_ANSWER,
                )
            )
            answer_index -= 1

    steps = tuple(reversed(steps_reversed))
    matched_pairs = tuple(
        (step.question_index, step.answer_index)
        for step in steps
        if step.operation is AlignmentOp.MATCH
        and step.question_index is not None
        and step.answer_index is not None
    )
    skipped_questions = tuple(
        step.question_index
        for step in steps
        if step.operation is AlignmentOp.SKIP_QUESTION and step.question_index is not None
    )
    skipped_answers = tuple(
        step.answer_index
        for step in steps
        if step.operation is AlignmentOp.SKIP_ANSWER and step.answer_index is not None
    )

    return AlignmentResult(
        steps=steps,
        matched_pairs=matched_pairs,
        skipped_questions=skipped_questions,
        skipped_answers=skipped_answers,
        score=scores[question_count][answer_count],
        ambiguous=path_counts[question_count][answer_count] > 1,
    )
