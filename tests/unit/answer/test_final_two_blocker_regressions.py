from __future__ import annotations

import pytest

from question_builder.answer.extractor import AnswerExtractionError, extract_answer_candidates
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


def _document(*blocks: ContentBlock) -> DocumentIR:
    return DocumentIR(
        document_id="doc_answer_final_blockers",
        source_file="answer-final-blockers.docx",
        source_sha256="c" * 64,
        blocks=blocks,
    )


def _paragraph(block_id: str, order: int, text: str) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type="paragraph",
        raw_text=text,
    )


def _critical(block_type: str, block_id: str, order: int) -> ContentBlock:
    if block_type == "image":
        return ContentBlock(
            block_id=block_id,
            order=order,
            type="image",
            raw_text="",
            metadata={"asset_filename": "leading-answer.png"},
        )
    if block_type == "formula":
        return ContentBlock(
            block_id=block_id,
            order=order,
            type="formula",
            raw_text="x+1",
            normalized_text="x+1",
        )
    if block_type == "table":
        return ContentBlock(
            block_id=block_id,
            order=order,
            type="table",
            raw_text="A B",
            metadata={
                "render_format": "markdown",
                "rendered": "| A | B |\n| --- | --- |",
            },
        )
    raise AssertionError(f"unsupported critical type: {block_type}")


def test_decimal_answer_does_not_create_false_next_question_number() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 答案为 2.5"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("1", "答案为 2.5"),
    ]


def test_decimal_result_does_not_create_false_following_question_number() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "2. 结果为 3.14"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("2", "结果为 3.14"),
    ]


def test_single_space_compact_mixed_answers_still_extract_all_entries() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. A 2. 15 3. B"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("1", "A"),
        ("2", "15"),
        ("3", "B"),
    ]


@pytest.mark.parametrize("block_type", ["image", "formula", "table"])
def test_unassigned_leading_critical_answer_block_is_rejected(block_type: str) -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _critical(block_type, "b2", 2),
        _paragraph("b3", 3, "2. B"),
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND
