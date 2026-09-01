from __future__ import annotations

import json

import pytest

import question_builder.splitter.builder as builder
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


def _document(blocks: list[ContentBlock]) -> DocumentIR:
    return DocumentIR(
        document_id="doc_builder",
        source_file="试卷.docx",
        source_sha256="e" * 64,
        blocks=tuple(blocks),
    )


def _paragraph(
    block_id: str,
    order: int,
    text: str,
    *,
    numbering: dict[str, object] | None = None,
) -> ContentBlock:
    return ContentBlock(
        block_id=block_id,
        order=order,
        type="paragraph",
        raw_text=text,
        normalized_text=text,
        numbering=numbering,
    )


def test_build_question_candidates_uses_strong_rule_ranges() -> None:
    build = getattr(builder, "build_question_candidates", None)
    assert callable(build)
    document = _document(
        [
            _paragraph("b1", 1, "第一题", numbering={"resolved_label": "1."}),
            _paragraph("b2", 2, "A. 甲  B. 乙  C. 丙  D. 丁"),
            _paragraph("b3", 3, "第二题", numbering={"resolved_label": "2."}),
        ]
    )

    candidates = build(document)

    assert [item.question_number for item in candidates] == ["1", "2"]
    assert [item.content_blocks for item in candidates] == [("b1", "b2"), ("b3",)]
    assert all(item.split_score >= 0.98 for item in candidates)


def test_weak_rule_only_document_rejects_low_confidence() -> None:
    build = getattr(builder, "build_question_candidates", None)
    error_type = getattr(builder, "QuestionSplitError", None)
    assert callable(build)
    assert isinstance(error_type, type)
    document = _document(
        [
            _paragraph("b1", 1, "阅读下面材料"),
            _paragraph("b2", 2, "A. 甲  B. 乙  C. 丙  D. 丁"),
        ]
    )

    with pytest.raises(error_type) as exc_info:
        build(document)

    assert exc_info.value.reason_code is RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE


def test_llm_split_contract_allows_only_source_block_ranges_and_confidence() -> None:
    parse = getattr(builder, "parse_llm_split", None)
    contract_error = getattr(builder, "QuestionSplitContractError", None)
    assert callable(parse)
    assert isinstance(contract_error, type)
    document = _document(
        [
            _paragraph("b1", 1, "材料"),
            _paragraph("b2", 2, "问题"),
            _paragraph("b3", 3, "下一题"),
        ]
    )

    selection = parse(
        json.dumps(
            {
                "ranges": [
                    {"content_blocks": ["b1", "b2"], "confidence": 0.99},
                    {"content_blocks": ["b3"], "confidence": 0.995},
                ]
            }
        ),
        document,
    )
    assert [item.content_blocks for item in selection.ranges] == [("b1", "b2"), ("b3",)]

    with pytest.raises(contract_error, match="only"):
        parse(
            json.dumps(
                {
                    "ranges": [
                        {
                            "content_blocks": ["b1", "b2"],
                            "confidence": 0.99,
                            "question_text": "模型改写后的题目",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            document,
        )

    with pytest.raises(contract_error, match="unknown block"):
        parse(
            json.dumps({"ranges": [{"content_blocks": ["missing"], "confidence": 0.99}]}),
            document,
        )

    with pytest.raises(contract_error, match="source order"):
        parse(
            json.dumps({"ranges": [{"content_blocks": ["b2", "b1"], "confidence": 0.99}]}),
            document,
        )


def test_llm_disambiguation_builds_candidate_from_cited_source_blocks() -> None:
    parse = getattr(builder, "parse_llm_split", None)
    build = getattr(builder, "build_question_candidates", None)
    assert callable(parse)
    assert callable(build)
    document = _document(
        [
            _paragraph("b1", 1, "共享材料"),
            _paragraph("b2", 2, "问题 A"),
            _paragraph("b3", 3, "问题 B"),
        ]
    )
    selection = parse(
        json.dumps({"ranges": [{"content_blocks": ["b1", "b2", "b3"], "confidence": 0.99}]}),
        document,
    )

    candidates = build(document, llm_selection=selection)

    assert len(candidates) == 1
    assert candidates[0].content_blocks == ("b1", "b2", "b3")
    assert candidates[0].split_score == 0.99


def test_low_confidence_llm_range_rejects_instead_of_auto_accepting() -> None:
    parse = getattr(builder, "parse_llm_split", None)
    build = getattr(builder, "build_question_candidates", None)
    error_type = getattr(builder, "QuestionSplitError", None)
    assert callable(parse)
    assert callable(build)
    assert isinstance(error_type, type)
    document = _document([_paragraph("b1", 1, "问题")])
    selection = parse(
        json.dumps({"ranges": [{"content_blocks": ["b1"], "confidence": 0.97}]}),
        document,
    )

    with pytest.raises(error_type) as exc_info:
        build(document, llm_selection=selection)

    assert exc_info.value.reason_code is RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE


def test_missing_or_unresolved_critical_block_rejects_incomplete_content() -> None:
    parse = getattr(builder, "parse_llm_split", None)
    build = getattr(builder, "build_question_candidates", None)
    error_type = getattr(builder, "QuestionSplitError", None)
    assert callable(parse)
    assert callable(build)
    assert isinstance(error_type, type)
    document = _document(
        [
            _paragraph("b1", 1, "题干"),
            ContentBlock(
                block_id="b2",
                order=2,
                type="formula",
                raw_text=r"x^2",
                normalized_text=r"x^2",
            ),
            _paragraph("b3", 3, "问题"),
        ]
    )
    selection = parse(
        json.dumps({"ranges": [{"content_blocks": ["b1", "b3"], "confidence": 0.99}]}),
        document,
    )

    with pytest.raises(error_type) as exc_info:
        build(document, llm_selection=selection)
    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE

    unresolved = _document(
        [
            _paragraph("b1", 1, "题干", numbering={"resolved_label": "1."}),
            ContentBlock(
                block_id="b2",
                order=2,
                type="unresolved",
                raw_text="",
                metadata={"reason": "FORMULA_UNRESOLVED"},
            ),
        ]
    )
    with pytest.raises(error_type) as unresolved_exc:
        build(unresolved)
    assert unresolved_exc.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE
