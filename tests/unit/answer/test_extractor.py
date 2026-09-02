from __future__ import annotations

from pathlib import Path

import pytest

from question_builder.answer.extractor import (
    AnswerExtractContractError,
    AnswerExtractionError,
    extract_answer_candidates,
    parse_llm_answer_extract,
)
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


def _document(*blocks: ContentBlock) -> DocumentIR:
    return DocumentIR(
        document_id="doc_answer",
        source_file="answers.docx",
        source_sha256="a" * 64,
        blocks=blocks,
    )


def _paragraph(block_id: str, order: int, text: str) -> ContentBlock:
    return ContentBlock(block_id=block_id, order=order, type="paragraph", raw_text=text)


def test_compact_answer_list_extracts_each_source_backed_answer() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. A  2. C  3. B"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer, item.source_blocks) for item in answers] == [
        ("1", "A", ("b2",)),
        ("2", "C", ("b2",)),
        ("3", "B", ("b2",)),
    ]
    assert all(item.analysis == "略" for item in answers)
    assert all(item.extract_score == 1.0 for item in answers)


def test_per_line_answers_preserve_question_numbers_and_source_blocks() -> None:
    document = _document(
        _paragraph("b1", 1, "答案"),
        _paragraph("b2", 2, "1. A"),
        _paragraph("b3", 3, "2. 15"),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer, item.source_blocks) for item in answers] == [
        ("1", "A", ("b2",)),
        ("2", "15", ("b3",)),
    ]


def test_answer_and_analysis_are_split_only_when_explicitly_separable() -> None:
    document = _document(
        _paragraph("b1", 1, "答案解析"),
        _paragraph("b2", 2, "1. 答案：C 解析：因为原文明确给出C。"),
    )

    answers = extract_answer_candidates(document)

    assert len(answers) == 1
    assert answers[0].question_number == "1"
    assert answers[0].answer == "C"
    assert answers[0].analysis == "因为原文明确给出C。"
    assert answers[0].source_blocks == ("b2",)


def test_inseparable_answer_and_reason_are_kept_together_with_default_analysis() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. C，因为题干中的条件直接对应C"),
    )

    answers = extract_answer_candidates(document)

    assert answers[0].answer == "C，因为题干中的条件直接对应C"
    assert answers[0].analysis == "略"


def test_long_form_solution_extracts_only_explicit_original_final_answer() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 解：由条件可得2x=4，继续化简。最终答案为 x=2"),
    )

    answers = extract_answer_candidates(document)

    assert len(answers) == 1
    assert answers[0].question_number == "1"
    assert answers[0].answer == "x=2"
    assert answers[0].analysis == "由条件可得2x=4，继续化简。"
    assert answers[0].source_blocks == ("b2",)


def test_long_form_solution_without_reliable_original_final_answer_is_rejected() -> None:
    document = _document(
        _paragraph("b1", 1, "参考答案"),
        _paragraph("b2", 2, "1. 解：由条件可得2x=4，再移项并化简。"),
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_missing_answer_is_rejected() -> None:
    document = _document(_paragraph("b1", 1, "本页没有任何答案内容"))

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_llm_fallback_accepts_only_cited_source_backed_spans() -> None:
    document = _document(
        _paragraph("b1", 1, "1. 解题过程。答案：42 解析：原文中的计算过程。"),
    )
    selection = parse_llm_answer_extract(
        '{"answers":[{"question_number":"1","answer":"42","analysis":"原文中的计算过程。",'
        '"source_blocks":["b1"],"confidence":0.99}]}',
        document,
    )

    answers = extract_answer_candidates(document, llm_selection=selection)

    assert answers[0].answer == "42"
    assert answers[0].analysis == "原文中的计算过程。"
    assert answers[0].source_blocks == ("b1",)
    assert answers[0].extract_score == 0.99


def test_llm_plausible_answer_absent_from_source_is_rejected() -> None:
    document = _document(_paragraph("b1", 1, "1. 只有解题过程，没有写出最终答案。"))
    selection = parse_llm_answer_extract(
        '{"answers":[{"question_number":"1","answer":"42","analysis":"略",'
        '"source_blocks":["b1"],"confidence":0.999}]}',
        document,
    )

    with pytest.raises(AnswerExtractionError) as exc_info:
        extract_answer_candidates(document, llm_selection=selection)

    assert exc_info.value.reason_code == RejectReason.ANSWER_NOT_FOUND


def test_llm_contract_rejects_unknown_source_blocks_and_extra_fields() -> None:
    document = _document(_paragraph("b1", 1, "1. 答案：A"))

    with pytest.raises(AnswerExtractContractError):
        parse_llm_answer_extract(
            '{"answers":[{"question_number":"1","answer":"A","analysis":"略",'
            '"source_blocks":["missing"],"confidence":0.99,"replacement_answer":"B"}]}',
            document,
        )


def test_answer_extract_prompt_forbids_solving_correcting_and_missing_answer_fill() -> None:
    prompt = Path("prompts/answer_extract/v1.txt").read_text(encoding="utf-8")

    assert "不得解题" in prompt
    assert "不得纠正" in prompt
    assert "不得补答案" in prompt
    assert "source_blocks" in prompt
    assert "只允许引用输入中存在的 block_id" in prompt
