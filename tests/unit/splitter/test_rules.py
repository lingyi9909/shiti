from __future__ import annotations

import question_builder.splitter.rules as rules
from question_builder.domain.document import ContentBlock, DocumentIR


def _document(rows: list[dict[str, object]]) -> DocumentIR:
    blocks = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", ""))
        blocks.append(
            ContentBlock(
                block_id=f"b{index}",
                order=index,
                type=str(row.get("type", "paragraph")),
                raw_text=text,
                normalized_text=text,
                numbering=row.get("numbering"),
                metadata=row.get("metadata", {}),
            )
        )
    return DocumentIR(
        document_id="doc_split_rules",
        source_file="试卷.docx",
        source_sha256="d" * 64,
        blocks=tuple(blocks),
    )


def test_structural_rules_split_top_level_questions_and_stop_before_answer_section() -> None:
    generate = getattr(rules, "generate_rule_ranges", None)
    assert callable(generate)
    document = _document(
        [
            {"text": "一、选择题"},
            {"text": "第一题题干", "numbering": {"resolved_label": "1."}},
            {"text": "A. 甲  B. 乙  C. 丙  D. 丁"},
            {"text": "第二题题干", "numbering": {"resolved_label": "二、"}},
            {"text": "（1）说明理由"},
            {"text": "参考答案"},
            {"text": "1. A"},
        ]
    )

    ranges = generate(document)

    assert [item.content_blocks for item in ranges] == [("b2", "b3"), ("b4", "b5")]
    assert [item.question_number for item in ranges] == ["1", "二"]
    assert all(item.score >= 0.98 for item in ranges)
    assert all("strong_top_level_number" in item.evidence for item in ranges)


def test_compound_question_keeps_shared_material_and_subquestions_together() -> None:
    generate = getattr(rules, "generate_rule_ranges", None)
    assert callable(generate)
    document = _document(
        [
            {"text": "阅读下面材料", "numbering": {"resolved_label": "1."}},
            {"text": "共享材料内容"},
            {"text": "第一个小问", "numbering": {"resolved_label": "（1）"}},
            {"text": "第二个小问", "numbering": {"resolved_label": "（2）"}},
            {"text": "下一道独立题", "numbering": {"resolved_label": "2."}},
        ]
    )

    ranges = generate(document)

    assert [item.content_blocks for item in ranges] == [
        ("b1", "b2", "b3", "b4"),
        ("b5",),
    ]


def test_section_heading_is_not_emitted_as_a_question_candidate_range() -> None:
    generate = getattr(rules, "generate_rule_ranges", None)
    assert callable(generate)
    document = _document(
        [
            {"text": "二、填空题"},
            {"text": "请填空", "numbering": {"resolved_label": "3."}},
        ]
    )

    ranges = generate(document)

    assert len(ranges) == 1
    assert ranges[0].content_blocks == ("b2",)
