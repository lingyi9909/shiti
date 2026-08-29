from __future__ import annotations

from pydantic import Field, field_validator

from question_builder.domain.document import DomainModel


class QuestionCandidate(DomainModel):
    question_candidate_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content_blocks: tuple[str, ...] = Field(min_length=1)
    question_number: str | None = None
    question_type_candidate: str | None = None
    split_score: float = Field(ge=0.0, le=1.0)

    @field_validator("content_blocks")
    @classmethod
    def validate_content_blocks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not block_id for block_id in value):
            raise ValueError("content block ids must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("content block ids must be unique")
        return value
