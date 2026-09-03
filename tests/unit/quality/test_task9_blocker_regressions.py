from __future__ import annotations

import json
from pathlib import Path

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
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


def _block(block_id: str, order: int, text: str) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type="paragraph",
        raw_text=text,
        metadata={},
    )


def _question_document() -> DocumentIR:
    return DocumentIR(
        document_id="qdoc",
        source_file="questions.docx",
        source_sha256="a" * 64,
        blocks=(
            _block("q1", 1, "1. 选择正确答案"),
            _block("q2", 2, "2. 第二题"),
        ),
    )


def _answer_document() -> DocumentIR:
    return DocumentIR(
        document_id="adoc",
        source_file="answers.docx",
        source_sha256="b" * 64,
        blocks=(_block("a1", 1, "1. A"),),
    )


def _question(
    candidate_id: str = "qc1",
    *,
    block_id: str = "q1",
    number: str = "1",
) -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id=candidate_id,
        document_id="qdoc",
        content_blocks=(block_id,),
        question_number=number,
        question_type_candidate="选择题",
        split_score=0.999,
    )


def _matched(
    question: QuestionCandidate | None = None,
    *,
    answer_text: str = "A",
    analysis: str = "略",
) -> MatchedQuestion:
    current_question = question or _question()
    answer = AnswerCandidate(
        answer_candidate_id="ac1",
        document_id="adoc",
        question_number=current_question.question_number,
        answer=answer_text,
        analysis=analysis,
        source_blocks=("a1",),
        extract_score=0.999,
    )
    evidence = MatchEvidence(
        question_candidate_id=current_question.question_candidate_id,
        answer_candidate_id="ac1",
        match_score=0.999,
        second_best_score=0.50,
        verifier_score=0.999,
        question_source_blocks=current_question.content_blocks,
        answer_source_blocks=("a1",),
        evidence={
            "same_cluster": True,
            "cluster_conflict": False,
            "sequence_consistency": 1.0,
            "verifier_decision": "PASS",
            "verifier_source_backed": True,
        },
    )
    return MatchedQuestion(question=current_question, answer=answer, evidence=evidence)


def _final_payload(
    *,
    text_question: str = "1. 选择正确答案",
    text_answer: str = "A",
    answer_analysis: str = "略",
    source_question_blocks: list[str] | None = None,
    source_answer_blocks: list[str] | None = None,
    source_files: list[str] | None = None,
    pipeline_version: str = "question-builder-v1",
) -> dict[str, object]:
    static_info = json.dumps(
        {
            "copyright": "0",
            "md5_version": "slim_md5_v1",
            "pipeline_version": pipeline_version,
            "slim_question_md5": slim_question_md5(text_question),
            "source_answer_blocks": source_answer_blocks or ["a1"],
            "source_files": source_files or ["questions.docx", "answers.docx"],
            "source_question_blocks": source_question_blocks or ["q1"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "text_question": text_question,
        "is_pic_included": 0,
        "text_answer": text_answer,
        "answer_analysis": answer_analysis,
        "text_course": "数学",
        "text_grade_level": "初中一年级",
        "text_grade": "初中",
        "knowledge_points": "",
        "exam_points": "",
        "publisher": "",
        "text_paper": "测试试卷",
        "textbook_version": "",
        "static_info": static_info,
        "language": "zh",
        "text_year": "2024",
        "entrance_exam_type": "未知",
        "text_city": "",
        "question_type": "选择题",
        "competition_event": "",
    }


def _context(
    *,
    candidate_id: str = "qc1",
    split_question: QuestionCandidate | None = None,
    matched: MatchedQuestion | None = None,
    final_record: dict[str, object] | None = None,
    recognition: tuple[RecognitionGateEvidence, ...] = (),
) -> QualityGateContext:
    current_split = split_question or _question()
    current_matched = matched or _matched(current_split)
    return QualityGateContext(
        candidate_id=candidate_id,
        file_evidence=(
            FileGateEvidence(source_file="questions.docx"),
            FileGateEvidence(source_file="answers.docx"),
        ),
        documents=(_question_document(), _answer_document()),
        recognition=recognition,
        split=SplitGateEvidence(question=current_split),
        answer=AnswerGateEvidence(matched=current_matched),
        final_record=final_record or _final_payload(),
    )


def _run(context: QualityGateContext, *, image_dir: Path | None = None):
    return run_quality_gates(
        context,
        thresholds=QualityThresholds(),
        image_dir=image_dir,
    )


def _recognition_result(
    *,
    score: float,
    image_class: ImageClass,
    task: RecognitionTask,
    provider: str,
    model: str,
    content: str = "same content",
) -> RecognitionResult:
    return RecognitionResult(
        task=task,
        image_class=image_class,
        provider=provider,
        model=model,
        request_id=f"{provider}-{task.value}",
        latency_ms=1.0,
        raw_score=score,
        raw_score_reference="fixture",
        normalized_score=score,
        calibration_id=f"cal-{provider}-{task.value}",
        content=content,
    )


def _multimodal_evidence(
    image_class: ImageClass,
    reason: str,
    *,
    fallback_provider: str = "fallback-provider",
    include_fallback: bool = True,
    multimodal_score: float = 0.99,
) -> RecognitionGateEvidence:
    primary = _recognition_result(
        score=0.95,
        image_class=image_class,
        task=RecognitionTask.VISION,
        provider="primary-provider",
        model="vision-primary",
    )
    fallback = (
        _recognition_result(
            score=0.99,
            image_class=image_class,
            task=RecognitionTask.VISION,
            provider=fallback_provider,
            model=(
                "vision-primary"
                if fallback_provider == "primary-provider"
                else "vision-fallback"
            ),
        )
        if include_fallback
        else None
    )
    multimodal = _recognition_result(
        score=multimodal_score,
        image_class=image_class,
        task=RecognitionTask.LLM,
        provider="multimodal-provider",
        model="multimodal-model",
    )
    request = RecognitionRequest(
        task=RecognitionTask.VISION,
        image_class=image_class,
        input_ref="q1",
        critical=True,
    )
    routing = RecognitionRoutingResult(
        decision=RecognitionDecision.ACCEPT,
        reason=reason,
        result=multimodal,
        primary_result=primary,
        fallback_result=fallback,
    )
    return RecognitionGateEvidence(block_id="q1", request=request, routing=routing)


def test_candidate_id_must_match_split_question_identity() -> None:
    result = _run(_context(candidate_id="different-candidate"))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.QUESTION_SPLIT.value


def test_answer_matched_question_must_equal_split_authoritative_identity() -> None:
    q1 = _question()
    q2 = _question("qc2", block_id="q2", number="2")
    result = _run(_context(split_question=q1, matched=_matched(q2)))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.ANSWER_MATCH.value


def test_final_answer_must_be_bound_to_matched_answer() -> None:
    result = _run(_context(final_record=_final_payload(text_answer="B")))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.stage == QualityStage.FINAL_CONTRACT.value
    assert result.rejection.reason_code is RejectReason.SCHEMA_VALIDATION_FAILED


def test_final_question_cannot_self_validate_md5_for_different_question() -> None:
    payload = _final_payload(text_question="另一道题")
    result = _run(_context(final_record=payload))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_question_blocks", ["q2"]),
        ("source_answer_blocks", ["other-answer"]),
        ("source_files", ["other.docx"]),
    ],
)
def test_final_static_info_must_match_current_source_provenance(
    field: str,
    value: list[str],
) -> None:
    kwargs: dict[str, object] = {field: value}
    result = _run(_context(final_record=_final_payload(**kwargs)))  # type: ignore[arg-type]

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.SCHEMA_VALIDATION_FAILED


def test_final_pipeline_version_must_be_nonempty() -> None:
    result = _run(_context(final_record=_final_payload(pipeline_version="")))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.SCHEMA_VALIDATION_FAILED


def test_answer_image_reference_must_exist(tmp_path: Path) -> None:
    answer = '<img src="image/answer.png">'
    matched = _matched(answer_text=answer)
    payload = _final_payload(text_answer=answer)
    image_dir = tmp_path / "image"
    image_dir.mkdir()

    result = _run(_context(matched=matched, final_record=payload), image_dir=image_dir)

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.IMAGE_MISSING


def test_existing_answer_image_reference_passes(tmp_path: Path) -> None:
    answer = '<img src="image/answer.png">'
    matched = _matched(answer_text=answer)
    payload = _final_payload(text_answer=answer)
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "answer.png").write_bytes(b"png")

    result = _run(_context(matched=matched, final_record=payload), image_dir=image_dir)

    assert result.passed is True
    assert result.rejection is None


@pytest.mark.parametrize(
    ("image_class", "reason"),
    [
        (ImageClass.QUESTION_SCREENSHOT, "question_screenshot_multimodal_verified"),
        (ImageClass.MIXED, "multimodal_fallback_verified"),
        (ImageClass.UNKNOWN, "multimodal_fallback_verified"),
    ],
)
def test_verified_fallback_then_multimodal_route_is_accepted(
    image_class: ImageClass,
    reason: str,
) -> None:
    evidence = _multimodal_evidence(image_class, reason)

    result = _run(_context(recognition=(evidence,)))

    assert result.passed is True
    assert result.rejection is None


def test_multimodal_route_rejects_missing_or_non_independent_fallback() -> None:
    missing = _multimodal_evidence(
        ImageClass.MIXED,
        "multimodal_fallback_verified",
        include_fallback=False,
    )
    non_independent = _multimodal_evidence(
        ImageClass.UNKNOWN,
        "multimodal_fallback_verified",
        fallback_provider="primary-provider",
    )

    for evidence in (missing, non_independent):
        result = _run(_context(recognition=(evidence,)))
        assert result.passed is False
        assert result.rejection is not None
        assert result.rejection.reason_code is RejectReason.OCR_LOW_CONFIDENCE


def test_multimodal_route_rejects_low_final_multimodal_score() -> None:
    evidence = _multimodal_evidence(
        ImageClass.QUESTION_SCREENSHOT,
        "question_screenshot_multimodal_verified",
        multimodal_score=0.97,
    )

    result = _run(_context(recognition=(evidence,)))

    assert result.passed is False
    assert result.rejection is not None
    assert result.rejection.reason_code is RejectReason.OCR_LOW_CONFIDENCE
