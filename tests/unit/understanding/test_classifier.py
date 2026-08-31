from __future__ import annotations

import json

import pytest

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.understanding.classifier import (
    ClassificationContractError,
    DocumentClass,
    classify_document,
    extract_document_features,
    parse_llm_classification,
)


def _document(
    source_file: str,
    texts: list[str],
    *,
    block_types: list[str] | None = None,
) -> DocumentIR:
    types = block_types or ["paragraph"] * len(texts)
    return DocumentIR(
        document_id=f"doc_{abs(hash(source_file))}",
        source_file=source_file,
        source_sha256="a" * 64,
        blocks=tuple(
            ContentBlock(
                block_id=f"b{index}",
                order=index,
                type=block_type,
                raw_text=text,
                normalized_text=text,
            )
            for index, (text, block_type) in enumerate(zip(texts, types, strict=True), start=1)
        ),
    )


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (
            _document(
                "2025北京市七年级数学期中试卷.docx",
                [
                    "2025北京市七年级数学期中考试",
                    "1. 下列计算正确的是",
                    "A. 1  B. 2  C. 3  D. 4",
                    "2. 计算 2+3 的值",
                ],
            ),
            DocumentClass.QUESTION,
        ),
        (
            _document(
                "2025北京市七年级数学期中答案.docx",
                ["2025北京市七年级数学期中考试参考答案", "参考答案", "1. C", "2. 5"],
            ),
            DocumentClass.ANSWER,
        ),
        (
            _document(
                "2025北京市七年级数学期中试卷含答案.docx",
                [
                    "2025北京市七年级数学期中考试",
                    "1. 下列计算正确的是",
                    "A. 1  B. 2  C. 3  D. 4",
                    "参考答案",
                    "1. C",
                ],
            ),
            DocumentClass.QUESTION_AND_ANSWER,
        ),
        (
            _document(
                "数学资料.docx",
                ["1. 下列计算正确的是", "A. 1  B. 2  C. 3  D. 4", "1. C", "2. 5"],
            ),
            DocumentClass.MIXED,
        ),
        (
            _document("教学资料汇编.docx", ["课堂材料", "请阅读以下内容"]),
            DocumentClass.UNKNOWN,
        ),
    ],
)
def test_rule_first_document_classification_is_deterministic(
    document: DocumentIR,
    expected: DocumentClass,
) -> None:
    understanding = classify_document(document)

    assert understanding.document_class is expected
    assert understanding.rule_classification is expected
    assert understanding.source_evidence


def test_feature_extraction_preserves_document_metadata_and_number_sequences() -> None:
    document = _document(
        "2025北京市七年级数学期中试卷.docx",
        [
            "北京市2025年七年级数学期中考试",
            "1. 第一题",
            "2. 第二题",
            "参考答案",
            "1. A",
            "2. B",
        ],
        block_types=["header", "paragraph", "paragraph", "paragraph", "paragraph", "footer"],
    )

    features = extract_document_features(document)

    assert features.title == "北京市2025年七年级数学期中考试"
    assert features.subject == "数学"
    assert features.grade == "七年级"
    assert features.year == "2025"
    assert features.city == "北京市"
    assert features.exam_type == "期中"
    assert features.question_number_sequence == ("1", "2")
    assert features.answer_number_sequence == ("1", "2")
    assert "b1" in features.metadata_source_blocks


def test_llm_fallback_contract_requires_allowed_enum_and_existing_cited_blocks() -> None:
    document = _document("unknown.docx", ["普通材料", "答案可能在后文"])

    evidence = parse_llm_classification(
        json.dumps(
            {
                "document_class": "ANSWER",
                "cited_block_ids": ["b2"],
                "reason": "第二个内容块明确描述答案材料",
            },
            ensure_ascii=False,
        ),
        document,
    )

    assert evidence.document_class is DocumentClass.ANSWER
    assert evidence.cited_block_ids == ("b2",)

    with pytest.raises(ClassificationContractError, match="allowed document_class"):
        parse_llm_classification(
            json.dumps({"document_class": "SOLVED", "cited_block_ids": ["b1"]}),
            document,
        )

    with pytest.raises(ClassificationContractError, match="unknown cited block"):
        parse_llm_classification(
            json.dumps({"document_class": "QUESTION", "cited_block_ids": ["missing"]}),
            document,
        )


def test_llm_fallback_fuses_classification_without_overwriting_source_evidence() -> None:
    document = _document("unknown.docx", ["课堂材料", "参考答案见下"])
    rule_only = classify_document(document)
    assert rule_only.document_class is DocumentClass.UNKNOWN

    llm_evidence = parse_llm_classification(
        json.dumps(
            {
                "document_class": "ANSWER",
                "cited_block_ids": ["b2"],
                "reason": "引用原始块中的答案提示",
            },
            ensure_ascii=False,
        ),
        document,
    )
    fused = classify_document(document, llm_evidence=llm_evidence)

    assert fused.document_class is DocumentClass.ANSWER
    assert fused.rule_classification is DocumentClass.UNKNOWN
    assert fused.llm_evidence == llm_evidence
    assert fused.source_evidence == rule_only.source_evidence


def _numbered_document(source_file: str, rows: list[tuple[str, str]]) -> DocumentIR:
    return DocumentIR(
        document_id=f"doc_numbered_{abs(hash(source_file))}",
        source_file=source_file,
        source_sha256="b" * 64,
        blocks=tuple(
            ContentBlock(
                block_id=f"n{index}",
                order=index,
                type="paragraph",
                raw_text=text,
                normalized_text=text,
                numbering={"resolved_label": label},
            )
            for index, (label, text) in enumerate(rows, start=1)
        ),
    )


def test_structured_word_numbering_is_preferred_question_evidence() -> None:
    document = _numbered_document("普通试题.docx", [("1.", "下列说法正确的是")])

    understanding = classify_document(document)

    assert understanding.features.question_number_sequence == ("1",)
    assert understanding.document_class is DocumentClass.QUESTION
    assert understanding.features.question_evidence_blocks == ("n1",)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("（1）", "1"),
        ("①", "1"),
        ("A.", "A"),
        ("一、", "一"),
        ("1.1.", "1.1"),
    ],
)
def test_structured_word_numbering_styles_are_sequence_evidence(
    label: str,
    expected: str,
) -> None:
    document = _numbered_document("编号样式.docx", [(label, "这是一道题目")])

    features = extract_document_features(document)

    assert features.question_number_sequence == (expected,)
    assert features.question_evidence_blocks == ("n1",)


def test_continuous_word_auto_numbering_drives_question_classification_and_sequence() -> None:
    document = _numbered_document(
        "连续自动编号.docx",
        [
            ("1.", "第一道题"),
            ("2.", "第二道题"),
            ("3.", "第三道题"),
        ],
    )

    understanding = classify_document(document)

    assert understanding.features.question_number_sequence == ("1", "2", "3")
    assert understanding.document_class is DocumentClass.QUESTION


def test_single_weak_answer_like_line_does_not_force_answer_classification() -> None:
    document = _document("普通资料.docx", ["课堂材料", "1. A"])

    understanding = classify_document(document)

    assert understanding.rule_classification is DocumentClass.UNKNOWN
    assert understanding.document_class is DocumentClass.UNKNOWN


def test_answer_heading_and_continuous_answer_sequence_is_strong_answer_structure() -> None:
    document = _document("普通文件.docx", ["参考答案", "1. A", "2. C", "3. B"])

    understanding = classify_document(document)

    assert understanding.features.answer_number_sequence == ("1", "2", "3")
    assert understanding.document_class is DocumentClass.ANSWER


def test_title_subject_has_priority_over_incidental_body_keyword() -> None:
    document = _document(
        "考试资料.docx",
        ["2025年七年级历史期中考试", "古代数学的发展影响了社会生活"],
    )

    features = extract_document_features(document)

    assert features.subject == "历史"
    assert "b1" in features.metadata_source_blocks


def test_title_grade_has_priority_over_incidental_body_grade_keyword() -> None:
    document = _document(
        "考试资料.docx",
        ["2025年七年级历史期中考试", "一年级学生也可以阅读这段材料"],
    )

    features = extract_document_features(document)

    assert features.grade == "七年级"
    assert "b1" in features.metadata_source_blocks
