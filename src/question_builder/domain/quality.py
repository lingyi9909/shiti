from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from question_builder.domain.document import DomainModel


class RejectReason(StrEnum):
    DOCUMENT_PARSE_FAILED = "DOCUMENT_PARSE_FAILED"
    DOCUMENT_RELATION_BROKEN = "DOCUMENT_RELATION_BROKEN"
    QUESTION_SPLIT_LOW_CONFIDENCE = "QUESTION_SPLIT_LOW_CONFIDENCE"
    QUESTION_CONTENT_INCOMPLETE = "QUESTION_CONTENT_INCOMPLETE"
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    FORMULA_UNRESOLVED = "FORMULA_UNRESOLVED"
    TABLE_UNRESOLVED = "TABLE_UNRESOLVED"
    IMAGE_MISSING = "IMAGE_MISSING"
    ANSWER_NOT_FOUND = "ANSWER_NOT_FOUND"
    ANSWER_MATCH_AMBIGUOUS = "ANSWER_MATCH_AMBIGUOUS"
    ANSWER_VERIFICATION_FAILED = "ANSWER_VERIFICATION_FAILED"
    LANGUAGE_UNRESOLVED = "LANGUAGE_UNRESOLVED"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"


class RejectedRecord(DomainModel):
    candidate_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    reason_code: RejectReason
    details: dict[str, Any] = Field(default_factory=dict)
    source_files: list[str] = Field(default_factory=list)
