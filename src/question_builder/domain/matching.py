from __future__ import annotations

from typing import Any, cast

from pydantic import Field, field_serializer, field_validator, model_validator

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import DomainModel, freeze_value, thaw_value
from question_builder.domain.question import QuestionCandidate


class MatchEvidence(DomainModel):
    question_candidate_id: str = Field(min_length=1)
    answer_candidate_id: str = Field(min_length=1)
    match_score: float = Field(ge=0.0, le=1.0)
    second_best_score: float | None = Field(default=None, ge=0.0, le=1.0)
    verifier_score: float | None = Field(default=None, ge=0.0, le=1.0)
    question_source_blocks: tuple[str, ...] = Field(min_length=1)
    answer_source_blocks: tuple[str, ...] = Field(min_length=1)
    evidence: dict[str, float | str | bool] = Field(default_factory=dict)

    @field_validator("evidence", mode="after")
    @classmethod
    def freeze_evidence(cls, value: dict[str, float | str | bool]) -> dict[str, Any]:
        return freeze_value(value)

    @field_serializer("evidence")
    def serialize_evidence(self, value: dict[str, float | str | bool]) -> dict[str, Any]:
        return cast(dict[str, Any], thaw_value(value))


class MatchedQuestion(DomainModel):
    question: QuestionCandidate
    answer: AnswerCandidate
    evidence: MatchEvidence

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> MatchedQuestion:
        if self.evidence.question_candidate_id != self.question.question_candidate_id:
            raise ValueError("match evidence must reference the matched question candidate")
        if self.evidence.answer_candidate_id != self.answer.answer_candidate_id:
            raise ValueError("match evidence must reference the matched answer candidate")
        if self.evidence.question_source_blocks != self.question.content_blocks:
            raise ValueError("match evidence must preserve question source blocks")
        if self.evidence.answer_source_blocks != self.answer.source_blocks:
            raise ValueError("match evidence must preserve answer source blocks")
        return self
