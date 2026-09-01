from __future__ import annotations

import json

import pytest

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.splitter.builder import (
    LLMSplitRange,
    LLMSplitSelection,
    QuestionSplitContractError,
    QuestionSplitError,
    build_question_candidates,
    parse_llm_split,
)


def _document(blocks: list[ContentBlock]) -> DocumentIR:
    return DocumentIR(
        document_id="doc_task5_blockers",
        source_file="试卷.docx",
        source_sha256="f" * 64,
        blocks=tuple(blocks),
    )


def _paragraph(
    block_id: str,
    order: int,
    text: str,
    *,
    numbering: dict[str, object] | None = None,
    block_type: str = "paragraph",
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type=block_type,
        raw_text=text,
        normalized_text=text,
        numbering=numbering,
    )


def _selection(document: DocumentIR, ranges: list[tuple[list[str], float]]) -> LLMSplitSelection:
    return parse_llm_split(
        json.dumps(
            {
                "ranges": [
                    {"content_blocks": block_ids, "confidence": confidence}
                    for block_ids, confidence in ranges
                ]
            }
        ),
        document,
    )


def test_strong_structural_rules_cannot_be_overridden_by_llm_merge() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "第一题", numbering={"resolved_label": "1."}),
            _paragraph("b2", 2, "第二题", numbering={"resolved_label": "2."}),
        ]
    )
    llm_selection = _selection(document, [(["b1", "b2"], 0.999)])

    candidates = build_question_candidates(document, llm_selection=llm_selection)

    assert [candidate.content_blocks for candidate in candidates] == [("b1",), ("b2",)]
    assert [candidate.question_number for candidate in candidates] == ["1", "2"]


def test_llm_range_cannot_skip_ordinary_body_block() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "题干第一段"),
            _paragraph("b2", 2, "题干第二段"),
            _paragraph("b3", 3, "题干第三段"),
        ]
    )

    with pytest.raises(QuestionSplitContractError, match="continuous"):
        _selection(document, [(["b1", "b3"], 0.99)])


def test_llm_range_may_skip_deterministically_excluded_header() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "题干第一段"),
            _paragraph("b2", 2, "页眉", block_type="header"),
            _paragraph("b3", 3, "题干第二段"),
        ]
    )

    selection = _selection(document, [(["b1", "b3"], 0.99)])

    assert selection.ranges[0].content_blocks == ("b1", "b3")


@pytest.mark.parametrize(
    ("critical_type", "raw_text", "metadata"),
    [
        ("formula", "x^2", {}),
        ("image", "", {"asset_filename": "q.png"}),
        ("table", "", {"rendered": "| A |\n|---|\n| 1 |"}),
    ],
)
def test_candidate_set_rejects_unassigned_critical_block(
    critical_type: str,
    raw_text: str,
    metadata: dict[str, object],
) -> None:
    document = _document(
        [
            _paragraph("b1", 1, "第一段问题"),
            ContentBlock(
                block_id="b2",
                order=2,
                type=critical_type,
                raw_text=raw_text,
                normalized_text=raw_text,
                metadata=metadata,
            ),
            _paragraph("b3", 3, "下一段问题"),
        ]
    )
    llm_selection = _selection(document, [(["b1"], 0.99), (["b3"], 0.99)])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=llm_selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_llm_range_cannot_cross_into_explicit_answer_section() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "1. 第一题"),
            _paragraph("b2", 2, "参考答案"),
            _paragraph("b3", 3, "1. A"),
        ]
    )

    with pytest.raises(QuestionSplitContractError, match="answer section"):
        _selection(document, [(["b1", "b2", "b3"], 0.99)])

    manual = LLMSplitSelection(ranges=(LLMSplitRange(("b1", "b2", "b3"), 0.99),))
    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=manual)
    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_valid_abcd_option_structure_passes() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "请选择正确答案"),
            _paragraph("b2", 2, "A. 甲"),
            _paragraph("b3", 3, "B. 乙"),
            _paragraph("b4", 4, "C. 丙"),
            _paragraph("b5", 5, "D. 丁"),
        ]
    )
    selection = _selection(document, [(["b1", "b2", "b3", "b4", "b5"], 0.99)])

    candidates = build_question_candidates(document, llm_selection=selection)

    assert len(candidates) == 1
    assert candidates[0].content_blocks == ("b1", "b2", "b3", "b4", "b5")


def test_option_structure_rejects_range_starting_at_b_without_a() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "A. 甲"),
            _paragraph("b2", 2, "B. 乙"),
            _paragraph("b3", 3, "C. 丙"),
            _paragraph("b4", 4, "D. 丁"),
        ]
    )
    selection = _selection(document, [(["b2", "b3", "b4"], 0.99)])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE


def test_option_structure_rejects_duplicate_same_level_label() -> None:
    document = _document(
        [
            _paragraph("b1", 1, "A. 甲"),
            _paragraph("b2", 2, "B. 乙"),
            _paragraph("b3", 3, "B. 丙"),
            _paragraph("b4", 4, "C. 丁"),
        ]
    )
    selection = _selection(document, [(["b1", "b2", "b3", "b4"], 0.99)])

    with pytest.raises(QuestionSplitError) as exc_info:
        build_question_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE
