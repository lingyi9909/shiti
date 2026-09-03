from __future__ import annotations

import pytest

from question_builder.metadata.normalizer import (
    MetadataCandidate,
    MetadataNormalizationError,
    MetadataSource,
    normalize_metadata,
)


def _candidate(
    field: str,
    value: str,
    source: MetadataSource,
    score: float = 0.99,
) -> MetadataCandidate:
    return MetadataCandidate(field=field, value=value, source=source, score=score)


def test_normalizes_approved_course_grade_exam_and_question_enums() -> None:
    normalized = normalize_metadata(
        (
            _candidate("text_course", "道德与法治", MetadataSource.EXPLICIT_DOCUMENT),
            _candidate("text_grade_level", "初一", MetadataSource.EXPLICIT_DOCUMENT),
            _candidate("entrance_exam_type", "中考", MetadataSource.EXPLICIT_DOCUMENT),
            _candidate("question_type", "单选题", MetadataSource.EXPLICIT_DOCUMENT),
            _candidate("language", "中文", MetadataSource.EXPLICIT_DOCUMENT),
        )
    )

    assert normalized.text_course.value == "政治"
    assert normalized.text_grade_level.value == "初中一年级"
    assert normalized.text_grade.value == "初中"
    assert normalized.entrance_exam_type.value == "中考"
    assert normalized.question_type.value == "选择题"
    assert normalized.language.value == "zh"


def test_source_priority_beats_higher_score_from_lower_priority_source() -> None:
    normalized = normalize_metadata(
        (
            _candidate("text_course", "数学", MetadataSource.SAME_CLUSTER, 1.0),
            _candidate("text_course", "英语", MetadataSource.TITLE_HEADER, 0.999),
            _candidate("text_course", "语文", MetadataSource.FILENAME, 0.70),
            _candidate("text_course", "化学", MetadataSource.EXPLICIT_DOCUMENT, 0.60),
            _candidate("language", "zh", MetadataSource.EXPLICIT_DOCUMENT),
        )
    )

    assert normalized.text_course.value == "化学"
    assert normalized.text_course.source is MetadataSource.EXPLICIT_DOCUMENT
    assert normalized.text_course.score == 0.60


def test_same_source_uses_highest_scored_valid_candidate_and_preserves_provenance() -> None:
    normalized = normalize_metadata(
        (
            _candidate("text_grade_level", "八年级", MetadataSource.TITLE_HEADER, 0.82),
            _candidate("text_grade_level", "初三", MetadataSource.TITLE_HEADER, 0.91),
            _candidate("language", "zh", MetadataSource.EXPLICIT_DOCUMENT),
        )
    )

    assert normalized.text_grade_level.value == "初中三年级"
    assert normalized.text_grade_level.source is MetadataSource.TITLE_HEADER
    assert normalized.text_grade_level.score == 0.91
    assert normalized.text_grade.value == "初中"
    assert normalized.text_grade.source is MetadataSource.TITLE_HEADER
    assert normalized.text_grade.score == 0.91


def test_invalid_or_unknown_higher_priority_candidate_does_not_mask_valid_lower_source() -> None:
    normalized = normalize_metadata(
        (
            _candidate("entrance_exam_type", "期中考试", MetadataSource.EXPLICIT_DOCUMENT, 1.0),
            _candidate("entrance_exam_type", "高考", MetadataSource.SAME_CLUSTER, 0.88),
            _candidate("language", "zh", MetadataSource.EXPLICIT_DOCUMENT),
        )
    )

    assert normalized.entrance_exam_type.value == "高考"
    assert normalized.entrance_exam_type.source is MetadataSource.SAME_CLUSTER


def test_optional_free_text_defaults_to_empty_and_enum_fields_use_unknown() -> None:
    normalized = normalize_metadata(
        (_candidate("language", "en", MetadataSource.FILENAME, 0.95),)
    )

    assert normalized.text_course.value == "未知"
    assert normalized.text_grade_level.value == "未知"
    assert normalized.text_grade.value == "未知"
    assert normalized.entrance_exam_type.value == "未知"
    assert normalized.question_type.value == "未知"
    assert normalized.text_paper.value == "未知"
    assert normalized.knowledge_points.value == ""
    assert normalized.exam_points.value == ""
    assert normalized.publisher.value == ""
    assert normalized.textbook_version.value == ""
    assert normalized.text_year.value == ""
    assert normalized.text_city.value == ""
    assert normalized.competition_event.value == ""


def test_year_is_deterministically_normalized_to_four_digit_string() -> None:
    normalized = normalize_metadata(
        (
            _candidate("text_year", "2024年秋季", MetadataSource.FILENAME, 0.9),
            _candidate("language", "zh", MetadataSource.EXPLICIT_DOCUMENT),
        )
    )

    assert normalized.text_year.value == "2024"
    assert isinstance(normalized.text_year.value, str)


def test_language_must_resolve_to_lowercase_iso_639_1() -> None:
    with pytest.raises(MetadataNormalizationError, match="language"):
        normalize_metadata(
            (_candidate("language", "not-a-language", MetadataSource.LLM, 0.99),)
        )


def test_metadata_candidate_rejects_unknown_external_field() -> None:
    with pytest.raises(ValueError):
        _candidate("provider_payload", "must-not-enter-final", MetadataSource.LLM)
