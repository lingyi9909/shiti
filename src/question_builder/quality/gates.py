from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from question_builder.config.models import QualityThresholds
from question_builder.domain.document import DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion
from question_builder.domain.quality import RejectedRecord
from question_builder.domain.question import QuestionCandidate
from question_builder.recognition.contracts import RecognitionRequest
from question_builder.recognition.router import RecognitionRoutingResult


@dataclass(frozen=True, slots=True)
class FileGateEvidence:
    source_file: str
    parse_succeeded: bool = True
    zip_xml_complete: bool = True
    relations_resolved: bool = True
    encrypted_or_corrupt: bool = False


@dataclass(frozen=True, slots=True)
class RecognitionGateEvidence:
    block_id: str
    request: RecognitionRequest
    routing: RecognitionRoutingResult


@dataclass(frozen=True, slots=True)
class SplitGateEvidence:
    question: QuestionCandidate | None
    boundary_unique: bool = True
    structure_consistent: bool = True
    unassigned_critical_blocks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerGateEvidence:
    matched: MatchedQuestion | None
    answer_candidates_present: bool = True
    match_ambiguous: bool = False
    alignment_consistent: bool = True


@dataclass(frozen=True, slots=True)
class QualityGateContext:
    candidate_id: str
    file_evidence: tuple[FileGateEvidence, ...]
    documents: tuple[DocumentIR, ...]
    recognition: tuple[RecognitionGateEvidence, ...]
    split: SplitGateEvidence
    answer: AnswerGateEvidence
    final_record: FinalQuestionRecord | Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    rejection: RejectedRecord | None = None


def run_quality_gates(
    context: QualityGateContext,
    *,
    thresholds: QualityThresholds,
    image_dir: Path | None = None,
) -> QualityGateResult:
    raise NotImplementedError
