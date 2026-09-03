from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from question_builder.domain.document import DocumentIR
from question_builder.domain.final import _ISO_639_1_CODES, FinalQuestionRecord
from question_builder.domain.matching import MatchedQuestion
from question_builder.export.markdown import render_question_markdown


class MetadataSource(StrEnum):
    EXPLICIT_DOCUMENT = "explicit_document"
    FILENAME = "filename"
    TITLE_HEADER = "title_header"
    SAME_CLUSTER = "same_cluster"
    LLM = "llm"
    DEFAULT = "default"
    DERIVED = "derived"


_INCOMING_SOURCES = {
    MetadataSource.EXPLICIT_DOCUMENT,
    MetadataSource.FILENAME,
    MetadataSource.TITLE_HEADER,
    MetadataSource.SAME_CLUSTER,
    MetadataSource.LLM,
}
_SOURCE_PRIORITY = {
    MetadataSource.EXPLICIT_DOCUMENT: 0,
    MetadataSource.FILENAME: 1,
    MetadataSource.TITLE_HEADER: 2,
    MetadataSource.SAME_CLUSTER: 3,
    MetadataSource.LLM: 4,
}
_METADATA_FIELDS = {
    "text_course",
    "text_grade_level",
    "knowledge_points",
    "exam_points",
    "publisher",
    "text_paper",
    "textbook_version",
    "language",
    "text_year",
    "entrance_exam_type",
    "text_city",
    "question_type",
    "competition_event",
}
_FREE_TEXT_FIELDS = {
    "knowledge_points",
    "exam_points",
    "publisher",
    "text_paper",
    "textbook_version",
    "text_city",
    "competition_event",
}
_COURSES = {
    "语文",
    "数学",
    "英语",
    "科学",
    "政治",
    "历史",
    "地理",
    "物理",
    "化学",
    "生物",
}
_COURSE_ALIASES = {
    "道德与法治": "政治",
    "思想政治": "政治",
}
_GRADE_LEVEL_ALIASES = {
    "一年级": "小学一年级",
    "二年级": "小学二年级",
    "三年级": "小学三年级",
    "四年级": "小学四年级",
    "五年级": "小学五年级",
    "六年级": "小学六年级",
    "七年级": "初中一年级",
    "八年级": "初中二年级",
    "九年级": "初中三年级",
    "初一": "初中一年级",
    "初二": "初中二年级",
    "初三": "初中三年级",
    "高一": "高中一年级",
    "高二": "高中二年级",
    "高三": "高中三年级",
}
_GRADE_LEVELS = {
    *(f"小学{value}年级" for value in "一二三四五六"),
    *(f"初中{value}年级" for value in "一二三"),
    *(f"高中{value}年级" for value in "一二三"),
}
_ENTRANCE_EXAM_TYPES = {"小升初", "中考", "高考"}
_QUESTION_TYPE_ALIASES = {
    "判断": "判断题",
    "判断题": "判断题",
    "单选": "选择题",
    "单选题": "选择题",
    "多选": "选择题",
    "多选题": "选择题",
    "选择": "选择题",
    "选择题": "选择题",
    "填空": "填空题",
    "填空题": "填空题",
    "问答": "问答题",
    "问答题": "问答题",
    "简答题": "问答题",
    "解答题": "问答题",
    "计算题": "问答题",
    "证明题": "问答题",
    "其他题型": "其他题型",
}
_LANGUAGE_ALIASES = {
    "中文": "zh",
    "汉语": "zh",
    "chinese": "zh",
    "英语": "en",
    "英文": "en",
    "english": "en",
}
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_IMAGE_REF = re.compile(r'<img\s+[^>]*src=["\']image/[^"\']+["\'][^>]*>', re.IGNORECASE)
MD5_VERSION = "slim_md5_v1"


@dataclass(frozen=True, slots=True)
class MetadataCandidate:
    field: str
    value: str
    source: MetadataSource
    score: float

    def __post_init__(self) -> None:
        if self.field not in _METADATA_FIELDS:
            raise ValueError(f"unsupported metadata field: {self.field}")
        if self.source not in _INCOMING_SOURCES:
            raise ValueError("metadata candidates must use an authoritative input source")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("metadata score must be numeric")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("metadata score must be between 0 and 1")
        if not isinstance(self.value, str):
            raise ValueError("metadata candidate value must be a string")


@dataclass(frozen=True, slots=True)
class MetadataValue:
    value: str
    source: MetadataSource
    score: float


@dataclass(frozen=True, slots=True)
class NormalizedMetadata:
    text_course: MetadataValue
    text_grade_level: MetadataValue
    text_grade: MetadataValue
    knowledge_points: MetadataValue
    exam_points: MetadataValue
    publisher: MetadataValue
    text_paper: MetadataValue
    textbook_version: MetadataValue
    language: MetadataValue
    text_year: MetadataValue
    entrance_exam_type: MetadataValue
    text_city: MetadataValue
    question_type: MetadataValue
    competition_event: MetadataValue


class MetadataNormalizationError(ValueError):
    pass


class FinalBuildError(ValueError):
    pass


def _clean(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _normalize_course(value: str) -> str | None:
    cleaned = _clean(value)
    if cleaned in _COURSES:
        return cleaned
    return _COURSE_ALIASES.get(cleaned)


def _normalize_grade_level(value: str) -> str | None:
    cleaned = _clean(value)
    if cleaned in _GRADE_LEVELS:
        return cleaned
    return _GRADE_LEVEL_ALIASES.get(cleaned)


def _normalize_entrance_exam_type(value: str) -> str | None:
    cleaned = _clean(value)
    return cleaned if cleaned in _ENTRANCE_EXAM_TYPES else None


def _normalize_question_type(value: str) -> str | None:
    return _QUESTION_TYPE_ALIASES.get(_clean(value))


def _normalize_language(value: str) -> str | None:
    cleaned = _clean(value)
    alias = _LANGUAGE_ALIASES.get(cleaned.lower())
    if alias is not None:
        return alias
    if cleaned == cleaned.lower() and cleaned in _ISO_639_1_CODES:
        return cleaned
    return None


def _normalize_year(value: str) -> str | None:
    match = _YEAR.search(_clean(value))
    return match.group(1) if match is not None else None


def _normalize_free_text(value: str) -> str | None:
    cleaned = _clean(value)
    if not cleaned or cleaned == "未知":
        return None
    return cleaned


def _normalize_candidate(candidate: MetadataCandidate) -> str | None:
    if candidate.field == "text_course":
        return _normalize_course(candidate.value)
    if candidate.field == "text_grade_level":
        return _normalize_grade_level(candidate.value)
    if candidate.field == "entrance_exam_type":
        return _normalize_entrance_exam_type(candidate.value)
    if candidate.field == "question_type":
        return _normalize_question_type(candidate.value)
    if candidate.field == "language":
        return _normalize_language(candidate.value)
    if candidate.field == "text_year":
        return _normalize_year(candidate.value)
    if candidate.field in _FREE_TEXT_FIELDS:
        return _normalize_free_text(candidate.value)
    raise MetadataNormalizationError(f"unsupported metadata field: {candidate.field}")


def _default(value: str) -> MetadataValue:
    return MetadataValue(value=value, source=MetadataSource.DEFAULT, score=0.0)


def _select_field(
    field: str,
    candidates: tuple[MetadataCandidate, ...],
    *,
    default: str,
) -> MetadataValue:
    valid: list[tuple[int, float, int, MetadataCandidate, str]] = []
    for index, candidate in enumerate(candidates):
        if candidate.field != field:
            continue
        normalized = _normalize_candidate(candidate)
        if normalized is None:
            continue
        valid.append(
            (
                _SOURCE_PRIORITY[candidate.source],
                -float(candidate.score),
                index,
                candidate,
                normalized,
            )
        )
    if not valid:
        return _default(default)
    _, _, _, candidate, normalized = min(valid, key=lambda item: item[:3])
    return MetadataValue(
        value=normalized,
        source=candidate.source,
        score=float(candidate.score),
    )


def _grade_from_level(grade_level: MetadataValue) -> MetadataValue:
    if grade_level.value == "未知":
        return _default("未知")
    for prefix in ("小学", "初中", "高中"):
        if grade_level.value.startswith(prefix):
            return MetadataValue(
                value=prefix,
                source=grade_level.source,
                score=grade_level.score,
            )
    raise MetadataNormalizationError("text_grade_level cannot be mapped to text_grade")


def normalize_metadata(candidates: tuple[MetadataCandidate, ...]) -> NormalizedMetadata:
    if not isinstance(candidates, tuple):
        raise MetadataNormalizationError("metadata candidates must be a tuple")

    text_grade_level = _select_field(
        "text_grade_level", candidates, default="未知"
    )
    language = _select_field("language", candidates, default="")
    if not language.value:
        raise MetadataNormalizationError("language could not be resolved to ISO 639-1")

    return NormalizedMetadata(
        text_course=_select_field("text_course", candidates, default="未知"),
        text_grade_level=text_grade_level,
        text_grade=_grade_from_level(text_grade_level),
        knowledge_points=_select_field("knowledge_points", candidates, default=""),
        exam_points=_select_field("exam_points", candidates, default=""),
        publisher=_select_field("publisher", candidates, default=""),
        text_paper=_select_field("text_paper", candidates, default="未知"),
        textbook_version=_select_field("textbook_version", candidates, default=""),
        language=language,
        text_year=_select_field("text_year", candidates, default=""),
        entrance_exam_type=_select_field(
            "entrance_exam_type", candidates, default="未知"
        ),
        text_city=_select_field("text_city", candidates, default=""),
        question_type=_select_field("question_type", candidates, default="未知"),
        competition_event=_select_field("competition_event", candidates, default=""),
    )


def _canonicalize_question(text_question: str) -> str:
    if not isinstance(text_question, str) or not text_question.strip():
        raise FinalBuildError("text_question must be non-empty")
    text = unicodedata.normalize("NFKC", text_question)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n(?:[ \t]*\n)+", "\n\n", text)
    return text


def slim_question_md5(text_question: str) -> str:
    canonical = _canonicalize_question(text_question)
    return hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()


def _source_files(question_document: DocumentIR, answer_document: DocumentIR) -> list[str]:
    files: list[str] = []
    for source_file in (question_document.source_file, answer_document.source_file):
        if source_file not in files:
            files.append(source_file)
    return files


def _validate_answer_sources(matched: MatchedQuestion, answer_document: DocumentIR) -> None:
    known_blocks = {block.block_id for block in answer_document.blocks}
    missing = [
        block_id
        for block_id in matched.answer.source_blocks
        if block_id not in known_blocks
    ]
    if missing:
        raise FinalBuildError(f"answer source blocks are missing: {missing}")


def build_final_question(
    matched: MatchedQuestion,
    *,
    question_document: DocumentIR,
    answer_document: DocumentIR,
    metadata: NormalizedMetadata,
    copyright: str,
    pipeline_version: str,
) -> FinalQuestionRecord:
    if question_document.document_id != matched.question.document_id:
        raise FinalBuildError("question document does not match the matched question")
    if answer_document.document_id != matched.answer.document_id:
        raise FinalBuildError("answer document does not match the matched answer")
    if copyright not in {"0", "1"}:
        raise FinalBuildError('copyright must be string "0" or "1"')
    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise FinalBuildError("pipeline_version must be non-empty")

    _validate_answer_sources(matched, answer_document)
    text_question = render_question_markdown(
        question_document,
        matched.question.content_blocks,
    )
    static_info = json.dumps(
        {
            "copyright": copyright,
            "md5_version": MD5_VERSION,
            "pipeline_version": pipeline_version.strip(),
            "slim_question_md5": slim_question_md5(text_question),
            "source_answer_blocks": list(matched.answer.source_blocks),
            "source_files": _source_files(question_document, answer_document),
            "source_question_blocks": list(matched.question.content_blocks),
        },
        ensure_ascii=False,
        sort_keys=True,
    )

    return FinalQuestionRecord(
        text_question=text_question,
        is_pic_included=1 if _IMAGE_REF.search(text_question) is not None else 0,
        text_answer=matched.answer.answer,
        answer_analysis=matched.answer.analysis.strip() or "略",
        text_course=metadata.text_course.value,  # type: ignore[arg-type]
        text_grade_level=metadata.text_grade_level.value,  # type: ignore[arg-type]
        text_grade=metadata.text_grade.value,  # type: ignore[arg-type]
        knowledge_points=metadata.knowledge_points.value,
        exam_points=metadata.exam_points.value,
        publisher=metadata.publisher.value,
        text_paper=metadata.text_paper.value,
        textbook_version=metadata.textbook_version.value,
        static_info=static_info,
        language=metadata.language.value,
        text_year=metadata.text_year.value,
        entrance_exam_type=metadata.entrance_exam_type.value,  # type: ignore[arg-type]
        text_city=metadata.text_city.value,
        question_type=metadata.question_type.value,  # type: ignore[arg-type]
        competition_event=metadata.competition_event.value,
    )
