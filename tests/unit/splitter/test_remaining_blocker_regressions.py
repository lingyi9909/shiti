from __future__ import annotations

import json

import pytest

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.splitter.builder import (
    QuestionSplitError,
    build_question_candidates,
    parse_llm_split,
)


def _document(blocks: list[ContentBlock]) -> DocumentIR:
    return DocumentIR(
        document_id="doc_task5_remaining_blockers",
        source_file="试卷.docx",
        source_sha256="a" * 64,
        blocks=tuple(blocks),
    )


def _paragraph(
    block_id: str,
    order: int,
    text: str,
    *,
    block_type: str = "paragraph",
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type=block_type,
        raw_text=text,
        normalized_text=text,
    )


def _selection(document: DocumentIR, ranges: list[list[str]]):
    return parse_llm_split(
        json.dumps(
            {
                "ranges": [
                    {"content_blocks": block_ids, "confidence": 0.99}
                    for block_ids in ranges
                ]
            }
        ),
        document,
    )


def _critical_block(block_id: str, order: int, critical_type: str) -> ContentBlock:
    if critical_type == "formula":
        return ContentBlock(
            block_id=block_id,
            order=order,
            type="formula",
            raw_text="x^2",
            normalized_text="x^2",
        )
    if critical_type == "image":
        return ContentBlock(
            block_id=block_id,
            order=order,
            type="image",
            raw_text="",
            metadata={"asset_filename": "q.png"},
        )
    return ContentBlock(
        block_id=block_id,
        order=order,
        type="table",
        raw_text="A",
        metadata={"rendered": "| A |\n| --- |"},
    )


def test_candidate_set_rejects_unassigned_ordinary_body_gap() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "题干第一段"),
            _paragraph("b2", 2, "题干续文"),
            _paragraph("b3", 3, "下一题"),
        ]
    )
    selection = _selection(document, [["b1"], ["b3"]])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_candidate_set_allows_deterministically_excluded_header_gap() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "题干第一段"),
            _paragraph("b2", 2, "页眉", block_type="header"),
            _paragraph("b3", 3, "下一题"),
        ]
    )
    selection = _selection(document, [["b1"], ["b3"]])

    candidates = build_question_candidates(document, llm_selection=selection)

    assert [candidate.content_blocks for candidate in candidates] == [("b1",), ("b3",)]


@pytest.mark.parametrize("critical_type", ["formula", "image", "table"])
def test_global_assignment_rejects_leading_unassigned_critical_block(
    critical_type: str,
) -> None:
    document = _document(
        [
            _critical_block("b1", 1, critical_type),
            _paragraph("b2", 2, "题干"),
        ]
    )
    selection = _selection(document, [["b2"]])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


@pytest.mark.parametrize("critical_type", ["formula", "image", "table"])
def test_global_assignment_rejects_trailing_unassigned_critical_before_answer(
    critical_type: str,
) -> None:
    document = _document(
        [
            _paragraph("b1", 1, "题干"),
            _critical_block("b2", 2, critical_type),
            _paragraph("b3", 3, "参考答案"),
        ]
    )
    selection = _selection(document, [["b1"]])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE
