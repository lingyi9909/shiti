from __future__ import annotations

import pytest

from question_builder.answer.extractor import (
    AnswerExtractionError,
    extract_answer_candidates,
    parse_llm_answer_extract,
)
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


def _document(*blocks: ContentBlock) -> DocumentIR:
    return DocumentIR(
        document_id="doc_answer_blockers",
        source_file="answer-blockers.docx",
        source_sha256="b" * 64,
        blocks=blocks,
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
        numbering=numbering,
    )


def test_compact_choice_matches_do_not_hide_following_numeric_answer() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. A  2. C"),
        _paragraph("b3", 3, "3. 15"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("1", "A"),
        ("2", "C"),
        ("3", "15"),
    ]


def test_mixed_compact_choice_and_numeric_answers_are_all_extracted() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. A  2. 15  3. B"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("1", "A"),
        ("2", "15"),
        ("3", "B"),
    ]


def test_llm_question_number_must_match_structured_source_numbering() -> None:
    document = _document(
        _paragraph("b2", 1, "A", numbering={"resolved_label": "2."}),
    )
    selection = parse_llm_answer_extract(
        '{"answers":[{"question_number":"1","answer":"A","analysis":"略",'
        '"source_blocks":["b2"],"confidence":0.99}]}',
        document,
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_llm_question_number_matching_structured_source_numbering_passes() -> None:
    document = _document(
        _paragraph("b2", 1, "A", numbering={"resolved_label": "2."}),
    )
    selection = parse_llm_answer_extract(
        '{"answers":[{"question_number":"2","answer":"A","analysis":"略",'
        '"source_blocks":["b2"],"confidence":0.99}]}',
        document,
    )

    answers = extract_answer_candidates(document, llm_selection=selection)

    assert len(answers) == 1
    assert answers[0].question_number == "2"
    assert answers[0].answer == "A"


def test_llm_sources_before_explicit_answer_section_are_rejected() -> None:
    document = _document(
        _paragraph("b1", 1, "1. 已知数字42，求……"),
        _paragraph("b2", 2, "参考答案"),
        _paragraph("b3", 3, "答案内容格式复杂，deterministic无法抽取"),
    )
    selection = parse_llm_answer_extract(
        '{"answers":[{"question_number":"1","answer":"42","analysis":"略",'
        '"source_blocks":["b1"],"confidence":0.99}]}',
        document,
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_answer_span_preserves_image_block_and_real_asset_reference() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 答案如下："),
        ContentBlock(
            block_id="b3",
            order=3,
            type="image",
            raw_text="",
            metadata={"asset_filename": "answer.png"},
        ),
    )

    answers = extract_answer_candidates(document)

    assert len(answers) == 1
    assert answers[0].source_blocks == ("b2", "b3")
    assert '<img src="image/answer.png">' in answers[0].answer


def test_answer_span_renders_formula_in_source_order() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. x ="),
        ContentBlock(
            block_id="b3",
            order=3,
            type="formula",
            raw_text="2",
            normalized_text="2",
        ),
    )

    answers = extract_answer_candidates(document)

    assert len(answers) == 1
    assert answers[0].source_blocks == ("b2", "b3")
    assert answers[0].answer == "x =\n\n$2$"


def test_answer_span_reuses_reliable_table_reconstruction() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 见表"),
        ContentBlock(
            block_id="b3",
            order=3,
            type="table",
            raw_text="A B",
            metadata={
                "render_format": "markdown",
                "rendered": "| A | B |\n| --- | --- |",
            },
        ),
    )

    answers = extract_answer_candidates(document)

    assert answers[0].source_blocks == ("b2", "b3")
    assert "| A | B |" in answers[0].answer
    assert "| --- | --- |" in answers[0].answer


def test_answer_span_with_unresolved_critical_content_is_rejected() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 答案如下："),
        ContentBlock(
            block_id="b3",
            order=3,
            type="unresolved",
            raw_text="",
            metadata={"reason": "FORMULA_UNRESOLVED"},
        ),
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_answer_span_with_image_missing_asset_mapping_is_rejected() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 答案如下："),
        ContentBlock(block_id="b3", order=3, type="image", raw_text=""),
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND
