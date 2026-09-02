from __future__ import annotations

from question_builder.answer.extractor import extract_answer_candidates
from question_builder.domain.document import ContentBlock, DocumentIR


def test_single_space_compact_mixed_answers_are_not_collapsed_into_first_answer() -> None:
    document = DocumentIR(
        document_id="doc_compact_single_space",
        source_file="answers.docx",
        source_sha256="c" * 64,
        blocks=(
            ContentBlock(block_id="b1", order=1, type="paragraph", raw_text="参考答案"),
            ContentBlock(
                block_id="b2",
                order=2,
                type="paragraph",
                raw_text="1. A 2. 15 3. B",
            ),
        ),
    )

    answers = extract_answer_candidates(document)

    assert [(item.question_number, item.answer) for item in answers] == [
        ("1", "A"),
        ("2", "15"),
        ("3", "B"),
    ]
