from __future__ import annotations

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.understanding.classifier import extract_document_features


def test_header_metadata_outranks_ordinary_first_body_paragraph() -> None:
    document = DocumentIR(
        document_id="doc_header_precedence",
        source_file="考试资料.docx",
        source_sha256="c" * 64,
        blocks=(
            ContentBlock(
                block_id="b1",
                order=1,
                type="paragraph",
                raw_text="一年级数学课堂练习",
                normalized_text="一年级数学课堂练习",
            ),
            ContentBlock(
                block_id="b2",
                order=2,
                type="header",
                raw_text="2025年七年级历史期中考试",
                normalized_text="2025年七年级历史期中考试",
            ),
        ),
    )

    features = extract_document_features(document)

    assert features.subject == "历史"
    assert features.grade == "七年级"
    assert features.year == "2025"
    assert features.exam_type == "期中"
    assert "b2" in features.metadata_source_blocks
