from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from question_builder.config.models import QualityThresholds
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion
from question_builder.domain.quality import QualityStage, RejectedRecord, RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.metadata.normalizer import slim_question_md5
from question_builder.recognition.contracts import (
    ImageClass,
    RecognitionRequest,
)
from question_builder.recognition.router import (
    RecognitionDecision,
    RecognitionRoutingResult,
)

_IMAGE_REF = re.compile(r'<img\s+src=["\']image/([^"\']+)["\'][^>]*>')
_ACCEPTED_RECOGNITION_REASONS = frozenset(
    {
        "primary_accept",
        "fallback_verified",
        "question_screenshot_multimodal_verified",
        "multimodal_fallback_verified",
    }
)
_RELATION_REASONS = frozenset(
    {
        "missing_image_relationship",
        "missing_image_relationship_target",
        "missing_image_package_member",
    }
)


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


def _source_files(context: QualityGateContext) -> tuple[str, ...]:
    ordered: list[str] = []
    for evidence in context.file_evidence:
        if evidence.source_file and evidence.source_file not in ordered:
            ordered.append(evidence.source_file)
    for document in context.documents:
        if document.source_file not in ordered:
            ordered.append(document.source_file)
    return tuple(ordered)


def _reject(
    context: QualityGateContext,
    stage: QualityStage,
    reason: RejectReason,
    details: Mapping[str, Any],
) -> QualityGateResult:
    return QualityGateResult(
        passed=False,
        rejection=RejectedRecord(
            candidate_id=context.candidate_id,
            stage=stage.value,
            reason_code=reason,
            details=dict(details),
            source_files=_source_files(context),
        ),
    )


def _file_gate(context: QualityGateContext) -> QualityGateResult | None:
    parse_failures = tuple(
        evidence
        for evidence in context.file_evidence
        if (
            not evidence.parse_succeeded
            or not evidence.zip_xml_complete
            or evidence.encrypted_or_corrupt
        )
    )
    if parse_failures:
        return _reject(
            context,
            QualityStage.FILE,
            RejectReason.DOCUMENT_PARSE_FAILED,
            {
                "source_files": [item.source_file for item in parse_failures],
                "checks": {
                    item.source_file: {
                        "parse_succeeded": item.parse_succeeded,
                        "zip_xml_complete": item.zip_xml_complete,
                        "encrypted_or_corrupt": item.encrypted_or_corrupt,
                    }
                    for item in parse_failures
                },
            },
        )

    relation_failures = tuple(
        evidence for evidence in context.file_evidence if not evidence.relations_resolved
    )
    if relation_failures:
        return _reject(
            context,
            QualityStage.FILE,
            RejectReason.DOCUMENT_RELATION_BROKEN,
            {"source_files": [item.source_file for item in relation_failures]},
        )
    return None


def _documents_by_id(context: QualityGateContext) -> dict[str, DocumentIR]:
    return {document.document_id: document for document in context.documents}


def _relevant_block_ids(context: QualityGateContext) -> frozenset[str]:
    block_ids: set[str] = set()
    if context.split.question is not None:
        block_ids.update(context.split.question.content_blocks)
    if context.answer.matched is not None:
        block_ids.update(context.answer.matched.answer.source_blocks)
    block_ids.update(item.block_id for item in context.recognition)
    return frozenset(block_ids)


def _selected_blocks(context: QualityGateContext) -> tuple[ContentBlock, ...]:
    relevant_ids = _relevant_block_ids(context)
    if not relevant_ids:
        return tuple(block for document in context.documents for block in document.blocks)
    return tuple(
        block
        for document in context.documents
        for block in document.blocks
        if block.block_id in relevant_ids
    )


def _document_gate(context: QualityGateContext) -> QualityGateResult | None:
    documents = _documents_by_id(context)
    question = context.split.question
    if question is not None:
        question_document = documents.get(question.document_id)
        if question_document is None:
            return _reject(
                context,
                QualityStage.DOCUMENT_IR,
                RejectReason.QUESTION_CONTENT_INCOMPLETE,
                {"check": "question_document_missing", "document_id": question.document_id},
            )
        known = {block.block_id for block in question_document.blocks}
        missing = [block_id for block_id in question.content_blocks if block_id not in known]
        if missing:
            return _reject(
                context,
                QualityStage.DOCUMENT_IR,
                RejectReason.QUESTION_CONTENT_INCOMPLETE,
                {"check": "question_source_blocks", "missing_blocks": missing},
            )

    for block in _selected_blocks(context):
        if block.type == "unresolved":
            raw_reason = block.metadata.get("reason")
            reason = raw_reason if isinstance(raw_reason, str) else ""
            if reason in _RELATION_REASONS:
                return _reject(
                    context,
                    QualityStage.DOCUMENT_IR,
                    RejectReason.DOCUMENT_RELATION_BROKEN,
                    {"block_id": block.block_id, "reason": reason},
                )
            if reason == RejectReason.FORMULA_UNRESOLVED.value:
                return _reject(
                    context,
                    QualityStage.DOCUMENT_IR,
                    RejectReason.FORMULA_UNRESOLVED,
                    {"block_id": block.block_id, "reason": reason},
                )
            if reason == RejectReason.TABLE_UNRESOLVED.value:
                return _reject(
                    context,
                    QualityStage.DOCUMENT_IR,
                    RejectReason.TABLE_UNRESOLVED,
                    {"block_id": block.block_id, "reason": reason},
                )
        if block.type == "image":
            asset_filename = block.metadata.get("asset_filename")
            if not isinstance(asset_filename, str) or not asset_filename:
                return _reject(
                    context,
                    QualityStage.DOCUMENT_IR,
                    RejectReason.IMAGE_MISSING,
                    {"block_id": block.block_id, "check": "asset_traceability"},
                )
    return None


def _canonical_content(content: str) -> str:
    return " ".join(content.split())


def _recognition_gate(
    context: QualityGateContext,
    thresholds: QualityThresholds,
) -> QualityGateResult | None:
    for evidence in context.recognition:
        request = evidence.request
        routing = evidence.routing
        threshold = (
            thresholds.critical_recognition_accept
            if request.critical
            else thresholds.noncritical_recognition_accept
        )
        primary = routing.primary_result
        result = routing.result

        if (
            routing.decision is not RecognitionDecision.ACCEPT
            or routing.reason not in _ACCEPTED_RECOGNITION_REASONS
            or result is None
            or result.normalized_score < threshold
        ):
            return _reject(
                context,
                QualityStage.RECOGNITION,
                RejectReason.OCR_LOW_CONFIDENCE,
                {
                    "block_id": evidence.block_id,
                    "routing_reason": routing.reason,
                    "normalized_score": (
                        result.normalized_score if result is not None else None
                    ),
                    "required_score": threshold,
                },
            )

        if primary is None or primary.normalized_score < thresholds.recognition_fallback_floor:
            return _reject(
                context,
                QualityStage.RECOGNITION,
                RejectReason.OCR_LOW_CONFIDENCE,
                {
                    "block_id": evidence.block_id,
                    "check": "primary_floor",
                    "normalized_score": (
                        primary.normalized_score if primary is not None else None
                    ),
                },
            )

        fallback = routing.fallback_result
        if routing.reason == "fallback_verified":
            if (
                fallback is None
                or fallback.normalized_score < threshold
                or (primary.provider, primary.model) == (fallback.provider, fallback.model)
                or _canonical_content(primary.content) != _canonical_content(fallback.content)
                or result != fallback
            ):
                return _reject(
                    context,
                    QualityStage.RECOGNITION,
                    RejectReason.OCR_LOW_CONFIDENCE,
                    {"block_id": evidence.block_id, "check": "fallback_verification"},
                )
        elif primary.normalized_score < threshold:
            return _reject(
                context,
                QualityStage.RECOGNITION,
                RejectReason.OCR_LOW_CONFIDENCE,
                {"block_id": evidence.block_id, "check": "fallback_required"},
            )

        if (
            fallback is not None
            and primary.normalized_score >= threshold
            and fallback.normalized_score >= threshold
            and _canonical_content(primary.content) != _canonical_content(fallback.content)
        ):
            return _reject(
                context,
                QualityStage.RECOGNITION,
                RejectReason.OCR_LOW_CONFIDENCE,
                {"block_id": evidence.block_id, "check": "conflicting_results"},
            )

        if request.image_class is ImageClass.QUESTION_SCREENSHOT:
            if routing.reason != "question_screenshot_multimodal_verified":
                return _reject(
                    context,
                    QualityStage.RECOGNITION,
                    RejectReason.OCR_LOW_CONFIDENCE,
                    {"block_id": evidence.block_id, "check": "multimodal_verification"},
                )
        if request.image_class in {ImageClass.MIXED, ImageClass.UNKNOWN}:
            if routing.reason != "multimodal_fallback_verified":
                return _reject(
                    context,
                    QualityStage.RECOGNITION,
                    RejectReason.OCR_LOW_CONFIDENCE,
                    {"block_id": evidence.block_id, "check": "multimodal_fallback"},
                )
    return None


def _split_gate(
    context: QualityGateContext,
    thresholds: QualityThresholds,
) -> QualityGateResult | None:
    split = context.split
    if (
        split.question is None
        or split.question.split_score < thresholds.split_accept
        or not split.boundary_unique
    ):
        return _reject(
            context,
            QualityStage.QUESTION_SPLIT,
            RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE,
            {
                "split_score": (
                    split.question.split_score if split.question is not None else None
                ),
                "required_score": thresholds.split_accept,
                "boundary_unique": split.boundary_unique,
            },
        )
    if not split.structure_consistent or split.unassigned_critical_blocks:
        return _reject(
            context,
            QualityStage.QUESTION_SPLIT,
            RejectReason.QUESTION_CONTENT_INCOMPLETE,
            {
                "structure_consistent": split.structure_consistent,
                "unassigned_critical_blocks": list(split.unassigned_critical_blocks),
            },
        )
    return None


def _answer_sources_exist(
    context: QualityGateContext,
    matched: MatchedQuestion,
) -> tuple[bool, tuple[str, ...]]:
    answer_document = _documents_by_id(context).get(matched.answer.document_id)
    if answer_document is None:
        return False, matched.answer.source_blocks
    known = {block.block_id for block in answer_document.blocks}
    missing = tuple(
        block_id for block_id in matched.answer.source_blocks if block_id not in known
    )
    return not missing, missing


def _answer_gate(
    context: QualityGateContext,
    thresholds: QualityThresholds,
) -> QualityGateResult | None:
    answer = context.answer
    matched = answer.matched
    if matched is None and not answer.answer_candidates_present:
        return _reject(
            context,
            QualityStage.ANSWER_MATCH,
            RejectReason.ANSWER_NOT_FOUND,
            {"check": "original_answer"},
        )
    if matched is None or answer.match_ambiguous:
        return _reject(
            context,
            QualityStage.ANSWER_MATCH,
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            {"check": "matched_question"},
        )

    sources_exist, missing_sources = _answer_sources_exist(context, matched)
    if not sources_exist:
        return _reject(
            context,
            QualityStage.ANSWER_MATCH,
            RejectReason.ANSWER_NOT_FOUND,
            {"check": "answer_source_blocks", "missing_blocks": list(missing_sources)},
        )

    evidence = matched.evidence
    details = evidence.evidence
    second = evidence.second_best_score
    margin_ok = second is None or evidence.match_score - second >= thresholds.answer_match_margin
    same_cluster = details.get("same_cluster") is True
    cluster_conflict = details.get("cluster_conflict") is True
    sequence_value = details.get("sequence_consistency")
    sequence_conflict = (
        isinstance(sequence_value, (int, float))
        and not isinstance(sequence_value, bool)
        and float(sequence_value) <= 0.0
    )
    if (
        evidence.match_score < thresholds.answer_match_accept
        or not margin_ok
        or not same_cluster
        or cluster_conflict
        or not answer.alignment_consistent
        or sequence_conflict
    ):
        return _reject(
            context,
            QualityStage.ANSWER_MATCH,
            RejectReason.ANSWER_MATCH_AMBIGUOUS,
            {
                "match_score": evidence.match_score,
                "second_best_score": second,
                "required_match_score": thresholds.answer_match_accept,
                "required_margin": thresholds.answer_match_margin,
                "same_cluster": same_cluster,
                "cluster_conflict": cluster_conflict,
                "alignment_consistent": answer.alignment_consistent,
            },
        )

    if (
        evidence.verifier_score is None
        or evidence.verifier_score < thresholds.answer_verify_accept
        or details.get("verifier_decision") != "PASS"
        or details.get("verifier_source_backed") is not True
    ):
        return _reject(
            context,
            QualityStage.ANSWER_MATCH,
            RejectReason.ANSWER_VERIFICATION_FAILED,
            {
                "verifier_score": evidence.verifier_score,
                "required_score": thresholds.answer_verify_accept,
                "verifier_decision": details.get("verifier_decision"),
                "verifier_source_backed": details.get("verifier_source_backed"),
            },
        )
    return None


def _language_only_validation_failure(payload: Mapping[str, Any]) -> bool:
    language = payload.get("language")
    if not isinstance(language, str):
        return False
    replacement = dict(payload)
    replacement["language"] = "zh"
    try:
        FinalQuestionRecord.model_validate(replacement)
    except ValidationError:
        return False
    try:
        FinalQuestionRecord.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors()
        return bool(errors) and all(
            error.get("loc") == ("language",) for error in errors
        )
    return False


def _validated_final_record(
    context: QualityGateContext,
) -> tuple[FinalQuestionRecord | None, QualityGateResult | None]:
    raw = context.final_record
    if raw is None:
        return None, _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "final_record_missing"},
        )
    if isinstance(raw, FinalQuestionRecord):
        return raw, None
    if not isinstance(raw, Mapping):
        return None, _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "final_record_type"},
        )

    expected_fields = frozenset(FinalQuestionRecord.model_fields)
    if frozenset(raw) != expected_fields:
        return None, _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {
                "check": "field_shape",
                "missing": sorted(expected_fields - frozenset(raw)),
                "extra": sorted(frozenset(raw) - expected_fields),
            },
        )
    if _language_only_validation_failure(raw):
        return None, _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.LANGUAGE_UNRESOLVED,
            {"check": "language", "value": raw.get("language")},
        )
    try:
        return FinalQuestionRecord.model_validate(raw), None
    except ValidationError as exc:
        return None, _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "schema", "errors": exc.errors(include_url=False)},
        )


def _safe_image_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and len(path.parts) == 1


def _final_gate(
    context: QualityGateContext,
    image_dir: Path | None,
) -> QualityGateResult | None:
    record, rejection = _validated_final_record(context)
    if rejection is not None:
        return rejection
    assert record is not None

    try:
        static_info = json.loads(record.static_info)
    except json.JSONDecodeError:
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "static_info"},
        )
    if not isinstance(static_info, dict):
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "static_info"},
        )
    if static_info.get("md5_version") != "slim_md5_v1":
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "md5_version", "value": static_info.get("md5_version")},
        )
    recomputed_md5 = slim_question_md5(record.text_question)
    if static_info.get("slim_question_md5") != recomputed_md5:
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {
                "check": "slim_question_md5",
                "expected": recomputed_md5,
                "actual": static_info.get("slim_question_md5"),
            },
        )

    image_refs = tuple(_IMAGE_REF.findall(record.text_question))
    expected_picture_flag = 1 if image_refs else 0
    if record.is_pic_included != expected_picture_flag:
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {
                "check": "is_pic_included",
                "expected": expected_picture_flag,
                "actual": record.is_pic_included,
            },
        )
    for image_name in image_refs:
        if not _safe_image_name(image_name):
            return _reject(
                context,
                QualityStage.FINAL_CONTRACT,
                RejectReason.IMAGE_MISSING,
                {"image": image_name, "check": "unsafe_image_reference"},
            )
        if image_dir is None or not (image_dir / image_name).is_file():
            return _reject(
                context,
                QualityStage.FINAL_CONTRACT,
                RejectReason.IMAGE_MISSING,
                {"image": image_name, "check": "image_file"},
            )

    try:
        serialized = json.loads(record.model_dump_json())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "json_serialization", "error": str(exc)},
        )
    if not isinstance(serialized, dict) or len(serialized) != 19:
        return _reject(
            context,
            QualityStage.FINAL_CONTRACT,
            RejectReason.SCHEMA_VALIDATION_FAILED,
            {"check": "json_line_shape"},
        )
    return None


def run_quality_gates(
    context: QualityGateContext,
    *,
    thresholds: QualityThresholds,
    image_dir: Path | None = None,
) -> QualityGateResult:
    for gate in (
        lambda: _file_gate(context),
        lambda: _document_gate(context),
        lambda: _recognition_gate(context, thresholds),
        lambda: _split_gate(context, thresholds),
        lambda: _answer_gate(context, thresholds),
        lambda: _final_gate(context, image_dir),
    ):
        rejection = gate()
        if rejection is not None:
            return rejection
    return QualityGateResult(passed=True)
