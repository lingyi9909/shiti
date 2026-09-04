from __future__ import annotations

import json
from pathlib import Path

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion, MatchEvidence
from question_builder.domain.quality import QualityStage, RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.metadata.normalizer import slim_question_md5
from question_builder.quality.gates import (
    AnswerGateEvidence,
    FileGateEvidence,
    QualityGateContext,
    RecognitionGateEvidence,
    SplitGateEvidence,
    run_quality_gates,
)
from question_builder.recognition.contracts import (
    ImageClass,
    RecognitionRequest,
    RecognitionResult,
    RecognitionTask,
)
from question_builder.recognition.router import (
    RecognitionDecision,
    RecognitionRoutingResult,
)


def _block(
    block_id: str,
    order: int,
    *,
    block_type: str = "paragraph",
    raw_text: str = "",
    metadata: dict[str, object] | None = None,
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type=block_type,
        raw_text=raw_text,
        metadata=metadata or {},
    )


def _question_document(
    *,
    blocks: tuple[ContentBlock, ...] | None = None,
) -> DocumentIR:
    return DocumentIR(
        document_id="qdoc",
        source_file="questions.docx",
        source_sha256="a" * 64,
        blocks=blocks or (_block("q1", 1, raw_text="1. 选择正确答案"),),
    )


def _answer_document() -> DocumentIR:
    return DocumentIR(
        document_id="adoc",
        source_file="answers.docx",
        source_sha256="b" * 64,
        blocks=(_block("a1", 1, raw_text="1. A"),),
    )


def _question(
    *,
    content_blocks: tuple[str, ...] = ("q1",),
    split_score: float = 0.999,
) -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id="qc1",
        document_id="qdoc",
        content_blocks=content_blocks,
        question_number="1",
        question_type_candidate="选择题",
        split_score=split_score,
    )


def _matched(
    *,
    question: QuestionCandidate | None = None,
    match_score: float = 0.999,
    second_best_score: float | None = 0.50,
    verifier_score: float | None = 0.999,
    same_cluster: bool = True,
    cluster_conflict: bool = False,
    sequence_consistency: float = 1.0,
    verifier_decision: str = "PASS",
    verifier_source_backed: bool = True,
) -> MatchedQuestion:
    current_question = question or _question()
    answer = AnswerCandidate(
        answer_candidate_id="ac1",
        document_id="adoc",
        question_number=current_question.question_number,
        answer="A",
        analysis="略",
        source_blocks=("a1",),
        extract_score=0.999,
    )
    evidence = MatchEvidence(
        question_candidate_id=current_question.question_candidate_id,
        answer_candidate_id="ac1",
        match_score=match_score,
        second_best_score=second_best_score,
        verifier_score=verifier_score,
        question_source_blocks=current_question.content_blocks,
        answer_source_blocks=("a1",),
        evidence={
            "same_cluster": same_cluster,
            "cluster_conflict": cluster_conflict,
            "sequence_consistency": sequence_consistency,
            "verifier_decision": verifier_decision,
            "verifier_source_backed": verifier_source_backed,
        },
    )
    return MatchedQuestion(question=current_question, answer=answer, evidence=evidence)


def _final_payload(
    *,
    text_question: str = "1. 选择正确答案",
    is_pic_included: int = 0,
    language: str = "zh",
    md5: str | None = None,
    source_question_blocks: tuple[str, ...] = ("q1",),
) -> dict[str, object]:
    static_info = json.dumps(
        {
            "copyright": "0",
            "md5_version": "slim_md5_v1",
            "pipeline_version": "question-builder-v1",
            "slim_question_md5": md5 or slim_question_md5(text_question),
            "source_answer_blocks": ["a1"],
            "source_files": ["questions.docx", "answers.docx"],
            "source_question_blocks": list(source_question_blocks),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "text_question": text_question,
        "is_pic_included": is_pic_included,
        "text_answer": "A",
        "answer_analysis": "略",
        "text_course": "数学",
        "text_grade_level": "初中一年级",
        "text_grade": "初中",
        "knowledge_points": "",
        "exam_points": "",
        "publisher": "",
        "text_paper": "测试试卷",
        "textbook_version": "",
        "static_info": static_info,
        "language": language,
        "text_year": "2024",
        "entrance_exam_type": "未知",
        "text_city": "",
        "question_type": "选择题",
        "competition_event": "",
    }


def _context(
    *,
    files: tuple[FileGateEvidence, ...] | None = None,
    question_document: DocumentIR | None = None,
    recognition: tuple[RecognitionGateEvidence, ...] = (),
    split: SplitGateEvidence | None = None,
    answer: AnswerGateEvidence | None = None,
    final_record: FinalQuestionRecord | dict[str, object] | None = None,
) -> QualityGateContext:
    qdoc = question_document or _question_document()
    return QualityGateContext(
        candidate_id="qc1",
        file_evidence=files
        or (
            FileGateEvidence(source_file="questions.docx"),
            FileGateEvidence(source_file="answers.docx"),
        ),
        documents=(qdoc, _answer_document()),
        recognition=recognition,
        split=split or SplitGateEvidence(question=_question()),
        answer=answer or AnswerGateEvidence(matched=_matched()),
        final_record=final_record or _final_payload(),
    )


def _result(
    context: QualityGateContext,
    *,
    image_dir: Path | None = None,
):
    return run_quality_gates(
        context,
        thresholds=QualityThresholds(),
        image_dir=image_dir,
    )


def _recognition_result(score: float, *, provider: str = "provider-a") -> RecognitionResult:
    return RecognitionResult(
        task=RecognitionTask.FORMULA_OCR,
        image_class=ImageClass.FORMULA_IMAGE,
        provider=provider,
        model="model-1",
        request_id=f"req-{provider}",
        latency_ms=1.0,
        raw_score=score,
        raw_score_reference="fixture",
        normalized_score=score,
        calibration_id=f"cal-{provider}",
        content="x=1",
    )


def _recognition_evidence(
    *,
    primary_score: float,
    decision: RecognitionDecision,
    reason: str,
    result_score: float | None = None,
    fallback_score: float | None = None,
) -> RecognitionGateEvidence:
    primary = _recognition_result(primary_score)
    fallback = (
        _recognition_result(fallback_score, provider="provider-b")
        if fallback_score is not None
        else None
    )
    result = None
    if result_score is not None:
        if fallback is not None and result_score == fallback.normalized_score:
            result = fallback
        else:
            result = _recognition_result(result_score)
    request = RecognitionRequest(
        task=RecognitionTask.FORMULA_OCR,
        image_class=ImageClass.FORMULA_IMAGE,
        input_ref="q1",
        critical=True,
    )
    routing = RecognitionRoutingResult(
        decision=decision,
        reason=reason,
        result=result,
        primary_result=primary,
        fallback_result=fallback,
    )
    return RecognitionGateEvidence(block_id="q1", request=request, routing=routing)


def test_gate_chain_returns_stable_earliest_failure() -> None:
    context = _context(
        files=(
            FileGateEvidence(source_file="questions.docx", parse_succeeded=False),
            FileGateEvidence(source_file="answers.docx", relations_resolved=False),
        ),
        split=SplitGateEvidence(question=_question(split_score=0.10)),
        answer=AnswerGateEvidence(matched=None, answer_candidates_present=False),
        final_record={"language": "invalid"},
    )

    result = _result(context)

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.FILE.value
    assert result.rejection.reason_code is RejectReason.DOCUMENT_PARSE_FAILED


def test_relation_failure_is_structured_before_later_gate_failures() -> None:
    context = _context(
        files=(
            FileGateEvidence(source_file="questions.docx", relations_resolved=False),
            FileGateEvidence(source_file="answers.docx"),
        ),
        split=SplitGateEvidence(question=_question(split_score=0.10)),
    )

    result = _result(context)

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.FILE.value
    assert result.rejection.reason_code is RejectReason.DOCUMENT_RELATION_BROKEN


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("FORMULA_UNRESOLVED", RejectReason.FORMULA_UNRESOLVED),
        ("TABLE_UNRESOLVED", RejectReason.TABLE_UNRESOLVED),
    ],
)
def test_document_gate_rejects_explicit_unresolved_critical_content(
    reason: str,
    expected: RejectReason,
) -> None:
    document = _question_document(
        blocks=(
            _block("q1", 1, raw_text="1. 题目"),
            _block(
                "u1",
                2,
                block_type="unresolved",
                metadata={"reason": reason},
            ),
        )
    )
    context = _context(
        question_document=document,
        split=SplitGateEvidence(question=_question(content_blocks=("q1", "u1"))),
    )

    result = _result(context)

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.DOCUMENT_IR.value
    assert result.rejection.reason_code is expected


def test_recognition_gate_rejects_low_confidence_or_unverified_fallback() -> None:
    evidence = _recognition_evidence(
        primary_score=0.95,
        decision=RecognitionDecision.ACCEPT,
        reason="primary_accept",
        result_score=0.95,
    )

    result = _result(_context(recognition=(evidence,)))

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.RECOGNITION.value
    assert result.rejection.reason_code is RejectReason.OCR_LOW_CONFIDENCE


def test_recognition_gate_accepts_verified_independent_fallback() -> None:
    evidence = _recognition_evidence(
        primary_score=0.95,
        decision=RecognitionDecision.ACCEPT,
        reason="fallback_verified",
        result_score=0.99,
        fallback_score=0.99,
    )

    result = _result(_context(recognition=(evidence,)))

    assert result.passed is True
    assert result.rejection is None


def test_split_gate_uses_configured_threshold() -> None:
    context = _context(split=SplitGateEvidence(question=_question(split_score=0.979)))

    result = _result(context)

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.QUESTION_SPLIT.value
    assert result.rejection.reason_code is RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE


def test_split_gate_rejects_unassigned_or_missing_critical_blocks() -> None:
    context = _context(
        split=SplitGateEvidence(
            question=_question(),
            unassigned_critical_blocks=("formula-2",),
        )
    )

    result = _result(context)

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.QUESTION_SPLIT.value
    assert result.rejection.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_answer_gate_distinguishes_missing_answer_from_ambiguous_match() -> None:
    missing = _result(
        _context(answer=AnswerGateEvidence(matched=None, answer_candidates_present=False))
    )
    ambiguous = _result(
        _context(
            answer=AnswerGateEvidence(
                matched=None,
                answer_candidates_present=True,
                match_ambiguous=True,
            )
        )
    )

    assert missing.rejection is not None
    assert missing.rejection.reason_code is RejectReason.ANSWER_NOT_FOUND
    assert ambiguous.rejection is not None
    assert ambiguous.rejection.reason_code is RejectReason.ANSWER_MATCH_AMBIGUOUS


def test_answer_gate_rechecks_score_margin_cluster_and_sequence() -> None:
    cases = (
        AnswerGateEvidence(matched=_matched(match_score=0.994)),
        AnswerGateEvidence(
            matched=_matched(match_score=0.999, second_best_score=0.95),
        ),
        AnswerGateEvidence(matched=_matched(same_cluster=False)),
        AnswerGateEvidence(matched=_matched(cluster_conflict=True)),
        AnswerGateEvidence(matched=_matched(), alignment_consistent=False),
    )

    for evidence in cases:
        result = _result(_context(answer=evidence))
        assert result.rejection is not None
        assert result.rejection.reason_code is RejectReason.ANSWER_MATCH_AMBIGUOUS


def test_answer_gate_requires_source_backed_passing_verifier() -> None:
    cases = (
        _matched(verifier_score=0.994),
        _matched(verifier_decision="FAIL"),
        _matched(verifier_source_backed=False),
    )

    for matched in cases:
        result = _result(_context(answer=AnswerGateEvidence(matched=matched)))
        assert result.rejection is not None
        assert result.rejection.reason_code is RejectReason.ANSWER_VERIFICATION_FAILED


def test_final_gate_classifies_unresolved_language_separately() -> None:
    payload = _final_payload(language="unknown")

    result = _result(_context(final_record=payload))

    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.FINAL_CONTRACT.value
    assert result.rejection.reason_code is RejectReason.LANGUAGE_UNRESOLVED


def test_final_gate_recomputes_md5_and_rejects_schema_tampering() -> None:
    payload = _final_payload(md5="0" * 32)

    result = _result(_context(final_record=payload))

    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.SCHEMA_VALIDATION_FAILED
    assert result.rejection.details["check"] == "slim_question_md5"


def test_final_gate_verifies_referenced_image_file_exists(tmp_path: Path) -> None:
    question = _question(content_blocks=("q1", "qimg"))
    question_document = _question_document(
        blocks=(
            _block("q1", 1, raw_text="1. 看图作答"),
            _block(
                "qimg",
                2,
                block_type="image",
                metadata={"asset_filename": "missing.png"},
            ),
        )
    )
    text_question = '1. 看图作答\n\n<img src="image/missing.png">'
    payload = _final_payload(
        text_question=text_question,
        is_pic_included=1,
        source_question_blocks=question.content_blocks,
    )
    image_dir = tmp_path / "image"
    image_dir.mkdir()

    result = _result(
        _context(
            question_document=question_document,
            split=SplitGateEvidence(question=question),
            answer=AnswerGateEvidence(matched=_matched(question=question)),
            final_record=payload,
        ),
        image_dir=image_dir,
    )

    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.IMAGE_MISSING


def test_full_valid_chain_passes_with_raw_or_typed_final_record(tmp_path: Path) -> None:
    question = _question(content_blocks=("q1", "qimg"))
    question_document = _question_document(
        blocks=(
            _block("q1", 1, raw_text="1. 看图作答"),
            _block(
                "qimg",
                2,
                block_type="image",
                metadata={"asset_filename": "present.png"},
            ),
        )
    )
    text_question = '1. 看图作答\n\n<img src="image/present.png">'
    payload = _final_payload(
        text_question=text_question,
        is_pic_included=1,
        source_question_blocks=question.content_blocks,
    )
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "present.png").write_bytes(b"png")
    context_kwargs = {
        "question_document": question_document,
        "split": SplitGateEvidence(question=question),
        "answer": AnswerGateEvidence(matched=_matched(question=question)),
    }

    raw_result = _result(
        _context(final_record=payload, **context_kwargs),
        image_dir=image_dir,
    )
    typed_result = _result(
        _context(
            final_record=FinalQuestionRecord.model_validate(payload),
            **context_kwargs,
        ),
        image_dir=image_dir,
    )

    assert raw_result.passed is True
    assert raw_result.rejection is None
    assert typed_result.passed is True
    assert typed_result.rejection is None


def test_every_reject_reason_serializes_required_structured_fields() -> None:
    from question_builder.domain.quality import RejectedRecord

    for reason in RejectReason:
        record = RejectedRecord(
            candidate_id="qc1",
            stage=QualityStage.FINAL_CONTRACT.value,
            reason_code=reason,
            details={"check": "fixture"},
            source_files=("questions.docx",),
        )
        payload = json.loads(record.model_dump_json())
        assert payload == {
            "candidate_id": "qc1",
            "stage": "final_contract",
            "reason_code": reason.value,
            "details": {"check": "fixture"},
            "source_files": ["questions.docx"],
        }
