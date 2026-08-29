from __future__ import annotations

from pydantic import Field, field_validator

from question_builder.domain.document import DomainModel


class AnswerCandidate(DomainModel):
    answer_candidate_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    question_number: str | None = None
    answer: str = Field(min_length=1)
    analysis: str = "略"
    source_blocks: list[str] = Field(min_length=1)
    extract_score: float = Field(ge=0.0, le=1.0)

    @field_validator("source_blocks")
    @classmethod
    def validate_source_blocks(cls, value: list[str]) -> list[str]:
        if any(not block_id for block_id in value):
            raise ValueError("answer source block ids must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("answer source block ids must be unique")
        return value
