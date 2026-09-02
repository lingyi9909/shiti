from __future__ import annotations

from dataclasses import dataclass

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import DocumentIR
from question_builder.domain.quality import RejectReason


class AnswerExtractContractError(ValueError):
    pass


class AnswerExtractionError(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class LLMAnswerSelection:
    payload: str


def parse_llm_answer_extract(payload: str, document: DocumentIR) -> LLMAnswerSelection:
    del document
    return LLMAnswerSelection(payload=payload)


def extract_answer_candidates(
    document: DocumentIR,
    *,
    llm_selection: LLMAnswerSelection | None = None,
) -> list[AnswerCandidate]:
    del document, llm_selection
    raise NotImplementedError("Task 6 answer extraction is not implemented")
