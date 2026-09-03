from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from question_builder.domain.document import DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion


class MetadataSource(StrEnum):
    EXPLICIT_DOCUMENT = "explicit_document"
    FILENAME = "filename"
    TITLE_HEADER = "title_header"
    SAME_CLUSTER = "same_cluster"
    LLM = "llm"
    DEFAULT = "default"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class MetadataCandidate:
    field: str
    value: str
    source: MetadataSource
    score: float


@dataclass(frozen=True, slots=True)
class MetadataValue:
    value: str
    source: MetadataSource
    score: float


@dataclass(frozen=True, slots=True)
class NormalizedMetadata:
    text_course: MetadataValue
    text_grade_level: MetadataValue
    text_grade: MetadataValue
    knowledge_points: MetadataValue
    exam_points: MetadataValue
    publisher: MetadataValue
    text_paper: MetadataValue
    textbook_version: MetadataValue
    language: MetadataValue
    text_year: MetadataValue
    entrance_exam_type: MetadataValue
    text_city: MetadataValue
    question_type: MetadataValue
    competition_event: MetadataValue


class MetadataNormalizationError(ValueError):
    pass


class FinalBuildError(ValueError):
    pass


def normalize_metadata(candidates: tuple[MetadataCandidate, ...]) -> NormalizedMetadata:
    raise NotImplementedError("Task 8 metadata normalization is not implemented")


def slim_question_md5(text_question: str) -> str:
    raise NotImplementedError("Task 8 slim_md5_v1 is not implemented")


def build_final_question(
    matched: MatchedQuestion,
    *,
    question_document: DocumentIR,
    answer_document: DocumentIR,
    metadata: NormalizedMetadata,
    copyright: str,
    pipeline_version: str,
) -> FinalQuestionRecord:
    raise NotImplementedError("Task 8 final builder is not implemented")
