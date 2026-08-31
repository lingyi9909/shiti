from __future__ import annotations

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.understanding.classifier import classify_document
from question_builder.understanding.clustering import build_exam_clusters


def _document(source_file: str, texts: list[str]) -> DocumentIR:
    return DocumentIR(
        document_id=f"doc_{source_file}",
        source_file=source_file,
        source_sha256=(source_file.encode("utf-8").hex() + "0" * 64)[:64],
        blocks=tuple(
            ContentBlock(
                block_id=f"b{index}",
                order=index,
                type="paragraph",
                raw_text=text,
                normalized_text=text,
            )
            for index, text in enumerate(texts, start=1)
        ),
    )


def test_obvious_question_and_answer_pair_join_same_exam_cluster() -> None:
    question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            [
                "北京市2025年七年级数学期中考试",
                "1. 第一题",
                "A. 1 B. 2 C. 3 D. 4",
                "2. 第二题",
            ],
        )
    )
    answer = classify_document(
        _document(
            "2025北京市七年级数学期中答案.docx",
            ["北京市2025年七年级数学期中考试参考答案", "参考答案", "1. A", "2. B"],
        )
    )

    clusters = build_exam_clusters((question, answer))

    assert len(clusters) == 1
    assert set(clusters[0].document_ids) == {question.document_id, answer.document_id}
    assert clusters[0].accepted is True
    assert "number_sequence_compatible" in clusters[0].evidence


def test_subject_year_or_grade_conflict_prevents_merge() -> None:
    math_question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            ["2025北京市七年级数学期中考试", "1. 第一题", "A. 1 B. 2 C. 3 D. 4"],
        )
    )
    chinese_answer = classify_document(
        _document(
            "2025北京市七年级语文期中答案.docx",
            ["2025北京市七年级语文期中考试参考答案", "参考答案", "1. A"],
        )
    )
    wrong_year_answer = classify_document(
        _document(
            "2024北京市七年级数学期中答案.docx",
            ["2024北京市七年级数学期中考试参考答案", "参考答案", "1. A"],
        )
    )
    wrong_grade_answer = classify_document(
        _document(
            "2025北京市八年级数学期中答案.docx",
            ["2025北京市八年级数学期中考试参考答案", "参考答案", "1. A"],
        )
    )

    clusters = build_exam_clusters(
        (math_question, chinese_answer, wrong_year_answer, wrong_grade_answer)
    )

    assert len(clusters) == 4
    assert all(len(cluster.document_ids) == 1 for cluster in clusters)


def test_ambiguous_document_remains_singleton_instead_of_force_merge() -> None:
    question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            ["2025北京市七年级数学期中考试", "1. 第一题", "A. 1 B. 2 C. 3 D. 4"],
        )
    )
    ambiguous = classify_document(
        _document("资料.docx", ["课堂材料", "内容整理"])
    )

    clusters = build_exam_clusters((question, ambiguous))

    assert len(clusters) == 2
    singleton = next(
        cluster for cluster in clusters if cluster.document_ids == (ambiguous.document_id,)
    )
    assert singleton.accepted is True
    assert "singleton_conservative" in singleton.evidence


def test_number_sequence_conflict_prevents_tempting_filename_merge() -> None:
    question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            [
                "2025北京市七年级数学期中考试",
                "1. 第一题",
                "A. 1 B. 2 C. 3 D. 4",
                "2. 第二题",
                "3. 第三题",
            ],
        )
    )
    answer = classify_document(
        _document(
            "2025北京市七年级数学期中答案.docx",
            ["2025北京市七年级数学期中考试参考答案", "参考答案", "7. A", "8. B"],
        )
    )

    clusters = build_exam_clusters((question, answer))

    assert len(clusters) == 2
    assert all(len(cluster.document_ids) == 1 for cluster in clusters)


def test_explicit_city_conflict_prevents_merge() -> None:
    question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            ["2025北京市七年级数学期中考试", "1. 第一题", "A. 1 B. 2 C. 3 D. 4"],
        )
    )
    answer = classify_document(
        _document(
            "2025上海市七年级数学期中答案.docx",
            ["2025上海市七年级数学期中考试参考答案", "参考答案", "1. A"],
        )
    )

    clusters = build_exam_clusters((question, answer))

    assert len(clusters) == 2
    assert all(len(cluster.document_ids) == 1 for cluster in clusters)


def test_explicit_exam_type_conflict_prevents_merge() -> None:
    question = classify_document(
        _document(
            "2025北京市七年级数学期中试卷.docx",
            ["2025北京市七年级数学期中考试", "1. 第一题", "A. 1 B. 2 C. 3 D. 4"],
        )
    )
    answer = classify_document(
        _document(
            "2025北京市七年级数学期末答案.docx",
            ["2025北京市七年级数学期末考试参考答案", "参考答案", "1. A"],
        )
    )

    clusters = build_exam_clusters((question, answer))

    assert len(clusters) == 2
    assert all(len(cluster.document_ids) == 1 for cluster in clusters)
