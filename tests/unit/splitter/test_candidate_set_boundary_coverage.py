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
        document_id="doc_task5_boundary_coverage",
        source_file="试卷.docx",
        source_sha256="b" * 64,
        blocks=tuple(blocks),
    )


def _paragraph(
    block_id: str,
    order: int,
    text: str,
    *,
    numbering: str | None = None,
    block_type: str = "paragraph",
) -> ContentBlock:
    numbering_data = {"resolved_label": numbering} if numbering is not None else None
    return ContentBlock(
        block_id=block_id,
        order=order,
        type=block_type,
        raw_text=text,
        normalized_text=text,
        numbering=numbering_data,
    )


def _selection(document: DocumentIR, block_ids: list[str]):
    return parse_llm_split(
        json.dumps(
            {
                "ranges": [
                    {
                        "content_blocks": block_ids,
                        "confidence": 0.99,
                    }
                ]
            }
        ),
        document,
    )


def test_shared_material_before_first_numbered_question_is_preserved() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "阅读下面材料："),
            _paragraph("b2", 2, "材料正文……"),
            _paragraph("b3", 3, "根据材料回答问题", numbering="1."),
            _paragraph("b4", 4, "第一个小问", numbering="（1）"),
            _paragraph("b5", 5, "第二个小问", numbering="（2）"),
        ]
    )

    candidates = build_question_candidates(document)

    assert [candidate.content_blocks for candidate in candidates] == [
        ("b1", "b2", "b3", "b4", "b5")
    ]


def test_trailing_ordinary_question_content_before_answer_cannot_be_unassigned() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "question"),
            _paragraph("b2", 2, "candidate selected content"),
            _paragraph("b3", 3, "题干续文"),
            _paragraph("b4", 4, "参考答案"),
        ]
    )
    selection = _selection(document, ["b1", "b2"])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_external_deterministic_exclusions_remain_allowed() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "页眉", block_type="header"),
            _paragraph("b2", 2, "一、选择题"),
            _paragraph("b3", 3, "题干"),
            _paragraph("b4", 4, "页脚", block_type="footer"),
            _paragraph("b5", 5, "扫描噪声", block_type="noise_candidate"),
            _paragraph("b6", 6, "参考答案"),
        ]
    )
    selection = _selection(document, ["b3"])

    candidates = build_question_candidates(document, llm_selection=selection)

    assert [candidate.content_blocks for candidate in candidates] == [("b3",)]
