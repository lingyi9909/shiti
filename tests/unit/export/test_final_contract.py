from __future__ import annotations

import json

import pytest

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.final import FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion, MatchEvidence
from question_builder.domain.question import QuestionCandidate
from question_builder.metadata.normalizer import (
    FinalBuildError,
    MetadataCandidate,
    MetadataSource,
    build_final_question,
    normalize_metadata,
    slim_question_md5,
)

FINAL_FIELDS = {
    "text_question",
    "is_pic_included",
    "text_answer",
    "answer_analysis",
    "text_course",
    "text_grade_level",
    "text_grade",
    "knowledge_points",
    "exam_points",
    "publisher",
    "text_paper",
    "textbook_version",
    "static_info",
    "language",
    "text_year",
    "entrance_exam_type",
    "text_city",
    "question_type",
    "competition_event",
}


def _block(
    block_id: str,
    order: int,
    block_type: str = "paragraph",
    raw_text: str = "",
    *,
    metadata: dict[str, object] | None = None,
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type=block_type,
        raw_text=raw_text,
        metadata=metadata or {},
    )


def _question_document(*, document_id: str = "qdoc", with_image: bool = True) -> DocumentIR:
    blocks = [_block("qb1", 1, raw_text="1. 计算 $1+1$")]
    if with_image:
        blocks.append(
            _block(
                "qb2",
                2,
                "image",
                metadata={"asset_filename": "abc123.png"},
            )
        )
    return DocumentIR(
        document_id=document_id,
        source_file="2024初一数学试卷.docx",
        source_sha256="a" * 64,
        blocks=tuple(blocks),
    )


def _answer_document(*, document_id: str = "adoc") -> DocumentIR:
    return DocumentIR(
        document_id=document_id,
        source_file="2024初一数学答案.docx",
        source_sha256="b" * 64,
        blocks=(_block("ab1", 1, raw_text="1. 2"),),
    )


def _matched(*, question_document_id: str = "qdoc", answer_document_id: str = "adoc", with_image: bool = True) -> MatchedQuestion:
    question_blocks = ("qb1", "qb2") if with_image else ("qb1",)
    question = QuestionCandidate(
        question_candidate_id="q1",
        document_id=question_document_id,
        content_blocks=question_blocks,
        question_number="1",
        question_type_candidate="填空题",
        split_score=1.0,
    )
    answer = AnswerCandidate(
        answer_candidate_id="a1",
        document_id=answer_document_id,
        question_number="1",
        answer="2",
        analysis="略",
        source_blocks=("ab1",),
        extract_score=1.0,
    )
    evidence = MatchEvidence(
        question_candidate_id="q1",
        answer_candidate_id="a1",
        match_score=0.999,
        second_best_score=0.5,
        verifier_score=0.999,
        question_source_blocks=question_blocks,
        answer_source_blocks=("ab1",),
        evidence={
            "verifier_provider": "must-not-leak",
            "verifier_model": "must-not-leak",
        },
    )
    return MatchedQuestion(question=question, answer=answer, evidence=evidence)


def _metadata():
    return normalize_metadata(
        (
            MetadataCandidate(field="text_course", value="数学", source=MetadataSource.FILENAME, score=0.95),
            MetadataCandidate(field="text_grade_level", value="初一", source=MetadataSource.FILENAME, score=0.95),
            MetadataCandidate(field="text_paper", value="2024年期末数学试卷", source=MetadataSource.TITLE_HEADER, score=0.98),
            MetadataCandidate(field="language", value="zh", source=MetadataSource.EXPLICIT_DOCUMENT, score=1.0),
            MetadataCandidate(field="text_year", value="2024年", source=MetadataSource.FILENAME, score=0.95),
            MetadataCandidate(field="question_type", value="填空题", source=MetadataSource.EXPLICIT_DOCUMENT, score=1.0),
        )
    )


def test_final_builder_produces_exact_19_fields_from_matched_source_evidence() -> None:
    record = build_final_question(
        _matched(),
        question_document=_question_document(),
        answer_document=_answer_document(),
        metadata=_metadata(),
        copyright="0",
        pipeline_version="question-builder-v1",
    )

    assert isinstance(record, FinalQuestionRecord)
    assert set(record.model_dump()) == FINAL_FIELDS
    assert len(record.model_dump()) == 19
    assert record.text_question == '1. 计算 $1+1$\n\n<img src="image/abc123.png">'
    assert record.is_pic_included == 1
    assert record.text_answer == "2"
    assert record.answer_analysis == "略"
    assert record.text_course == "数学"
    assert record.text_grade_level == "初中一年级"
    assert record.text_grade == "初中"
    assert record.text_year == "2024"
    assert record.question_type == "填空题"


def test_picture_flag_is_computed_from_final_question_not_caller_input() -> None:
    record = build_final_question(
        _matched(with_image=False),
        question_document=_question_document(with_image=False),
        answer_document=_answer_document(),
        metadata=_metadata(),
        copyright="0",
        pipeline_version="question-builder-v1",
    )

    assert record.is_pic_included == 0


def test_static_info_is_sorted_json_with_only_approved_provenance() -> None:
    record = build_final_question(
        _matched(),
        question_document=_question_document(),
        answer_document=_answer_document(),
        metadata=_metadata(),
        copyright="1",
        pipeline_version="question-builder-v1",
    )

    parsed = json.loads(record.static_info)
    assert parsed == {
        "copyright": "1",
        "md5_version": "slim_md5_v1",
        "pipeline_version": "question-builder-v1",
        "slim_question_md5": slim_question_md5(record.text_question),
        "source_answer_blocks": ["ab1"],
        "source_files": ["2024初一数学试卷.docx", "2024初一数学答案.docx"],
        "source_question_blocks": ["qb1", "qb2"],
    }
    assert record.static_info == json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    assert "verifier_provider" not in record.static_info
    assert "verifier_model" not in record.static_info


def test_slim_md5_v1_normalizes_unicode_line_endings_outer_trim_and_blank_lines() -> None:
    left = '  Ａ\r\n\r\n\r\n$x^2$\r\n<img src="image/a.png">  '
    right = 'A\n\n$x^2$\n<img src="image/a.png">'

    assert slim_question_md5(left) == slim_question_md5(right)
    assert len(slim_question_md5(left)) == 32
    assert slim_question_md5(left) == slim_question_md5(left).lower()


def test_slim_md5_v1_preserves_actual_formula_and_image_reference_content() -> None:
    baseline = '$x^2$\n\n<img src="image/a.png">'

    assert slim_question_md5(baseline) != slim_question_md5('$x^3$\n\n<img src="image/a.png">')
    assert slim_question_md5(baseline) != slim_question_md5('$x^2$\n\n<img src="image/b.png">')


def test_final_builder_fails_closed_when_document_identity_does_not_match_matched_question() -> None:
    with pytest.raises(FinalBuildError, match="question document"):
        build_final_question(
            _matched(),
            question_document=_question_document(document_id="wrong-qdoc"),
            answer_document=_answer_document(),
            metadata=_metadata(),
            copyright="0",
            pipeline_version="question-builder-v1",
        )

    with pytest.raises(FinalBuildError, match="answer document"):
        build_final_question(
            _matched(),
            question_document=_question_document(),
            answer_document=_answer_document(document_id="wrong-adoc"),
            metadata=_metadata(),
            copyright="0",
            pipeline_version="question-builder-v1",
        )


def test_final_builder_requires_binary_string_copyright_and_nonempty_pipeline_version() -> None:
    with pytest.raises(FinalBuildError, match="copyright"):
        build_final_question(
            _matched(),
            question_document=_question_document(),
            answer_document=_answer_document(),
            metadata=_metadata(),
            copyright="2",
            pipeline_version="question-builder-v1",
        )

    with pytest.raises(FinalBuildError, match="pipeline_version"):
        build_final_question(
            _matched(),
            question_document=_question_document(),
            answer_document=_answer_document(),
            metadata=_metadata(),
            copyright="0",
            pipeline_version="",
        )
