from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from question_builder.domain.document import ContentBlock, DocumentIR


class DocumentClass(StrEnum):
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    QUESTION_AND_ANSWER = "QUESTION_AND_ANSWER"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class ClassificationContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentFeatures:
    title: str | None
    subject: str | None
    grade: str | None
    year: str | None
    city: str | None
    exam_type: str | None
    question_number_sequence: tuple[str, ...]
    answer_number_sequence: tuple[str, ...]
    metadata_source_blocks: tuple[str, ...]
    question_evidence_blocks: tuple[str, ...]
    answer_evidence_blocks: tuple[str, ...]
    option_structure_blocks: tuple[str, ...]
    answer_heading_blocks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LLMClassificationEvidence:
    document_class: DocumentClass
    cited_block_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class DocumentUnderstanding:
    document_id: str
    source_file: str
    document_class: DocumentClass
    rule_classification: DocumentClass
    features: DocumentFeatures
    source_evidence: tuple[str, ...]
    llm_evidence: LLMClassificationEvidence | None = None


_SUBJECTS = (
    "数学",
    "语文",
    "英语",
    "物理",
    "化学",
    "生物",
    "历史",
    "地理",
    "政治",
    "道德与法治",
)
_GRADES = (
    "一年级",
    "二年级",
    "三年级",
    "四年级",
    "五年级",
    "六年级",
    "七年级",
    "八年级",
    "九年级",
    "初一",
    "初二",
    "初三",
    "高一",
    "高二",
    "高三",
)
_EXAM_TYPES = ("期中", "期末", "中考", "高考", "月考", "模拟", "联考")
_RAW_NUMBERED_LINE = re.compile(
    r"^\s*(?P<label>"
    r"(?:[（(]\s*\d{1,4}\s*[）)]|[①-⑳]|"
    r"\d{1,4}(?:[.．]\d{1,4})+[.．、)）]?|"
    r"\d{1,4}[.、．)）]|"
    r"[一二三四五六七八九十百]+[、.．])"
    r")\s*(?P<payload>.+?)\s*$"
)
_OPTION_STRUCTURE = re.compile(r"(?:^|\s)A[.、．)]\s*.+?(?:\s+)B[.、．)]\s*", re.IGNORECASE)
_SIMPLE_ANSWER = re.compile(
    r"^(?:[A-H](?:\s*[,，/]\s*[A-H])*|[-+]?\d+(?:\.\d+)?|[√×对错]|[TF])$",
    re.IGNORECASE,
)
_CITY = re.compile(r"([\u4e00-\u9fff]{2,8}市)")
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_MUNICIPALITIES = ("北京", "上海", "天津", "重庆")
_CIRCLED_NUMBER_START = ord("①")
_CIRCLED_NUMBER_END = ord("⑳")

type _MetadataSource = tuple[str, str | None]


def _text(block: ContentBlock) -> str:
    return (block.normalized_text if block.normalized_text is not None else block.raw_text).strip()


def _is_answer_heading(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return normalized in {
        "答案",
        "参考答案",
        "答案解析",
        "答案与解析",
        "参考答案及解析",
        "解析",
    }


def _normalize_number_label(label: str) -> str | None:
    normalized = re.sub(r"\s+", "", label).replace("．", ".")
    if not normalized:
        return None

    if len(normalized) == 1:
        codepoint = ord(normalized)
        if _CIRCLED_NUMBER_START <= codepoint <= _CIRCLED_NUMBER_END:
            return str(codepoint - _CIRCLED_NUMBER_START + 1)

    if normalized[0] in "(（" and normalized[-1] in ")）":
        normalized = normalized[1:-1]

    normalized = normalized.rstrip(".、)）")
    if re.fullmatch(r"\d{1,4}(?:\.\d{1,4})*", normalized):
        return normalized
    if re.fullmatch(r"[A-Za-z]", normalized):
        return normalized.upper()
    if re.fullmatch(r"[一二三四五六七八九十百]+", normalized):
        return normalized
    return None


def _structured_number(block: ContentBlock) -> str | None:
    if block.numbering is None:
        return None
    resolved_label = block.numbering.get("resolved_label")
    if not isinstance(resolved_label, str):
        return None
    return _normalize_number_label(resolved_label)


def _raw_numbered_payload(text: str) -> tuple[str, str] | None:
    match = _RAW_NUMBERED_LINE.match(text)
    if match is None:
        return None
    number = _normalize_number_label(match.group("label"))
    if number is None:
        return None
    return number, match.group("payload").strip()


def _numbered_payload(block: ContentBlock, text: str) -> tuple[str, str] | None:
    structured_number = _structured_number(block)
    if structured_number is not None:
        return structured_number, text
    return _raw_numbered_payload(text)


def _first_keyword(haystack: str, candidates: tuple[str, ...]) -> str | None:
    matches = (
        (index, -len(candidate), candidate)
        for candidate in candidates
        if (index := haystack.find(candidate)) >= 0
    )
    first = min(matches, default=None)
    return first[2] if first is not None else None


def _year_from_text(text: str) -> str | None:
    match = _YEAR.search(text)
    return match.group(1) if match is not None else None


def _city_from_text(text: str) -> str | None:
    for municipality in _MUNICIPALITIES:
        if f"{municipality}市" in text or municipality in text:
            return f"{municipality}市"
    match = _CITY.search(text)
    return match.group(1) if match is not None else None


def _first_title(blocks: tuple[ContentBlock, ...]) -> tuple[str | None, tuple[str, ...]]:
    for preferred_type in ("header", "paragraph", "textbox"):
        for block in blocks:
            text = _text(block)
            if block.type == preferred_type and text:
                return text, (block.block_id,)
    return None, ()


def _first_explicit_title(blocks: tuple[ContentBlock, ...]) -> tuple[str | None, str | None]:
    for preferred_type in ("paragraph", "textbox"):
        for block in blocks:
            text = _text(block)
            if block.type == preferred_type and text:
                return text, block.block_id
    return None, None


def _metadata_sources(
    document: DocumentIR,
) -> tuple[tuple[_MetadataSource, ...], tuple[_MetadataSource, ...], tuple[str, ...]]:
    source_stem = Path(document.source_file).stem
    explicit_title, explicit_title_block = _first_explicit_title(document.blocks)

    strong_sources: list[_MetadataSource] = [(source_stem, None)]
    strong_block_ids: list[str] = []

    for block in document.blocks:
        text = _text(block)
        if block.type == "header" and text:
            strong_sources.append((text, block.block_id))
            strong_block_ids.append(block.block_id)

    if explicit_title is not None:
        strong_sources.append((explicit_title, explicit_title_block))
        if explicit_title_block is not None:
            strong_block_ids.append(explicit_title_block)

    strong_id_set = set(strong_block_ids)
    weak_sources = tuple(
        (text, block.block_id)
        for block in document.blocks
        if block.block_id not in strong_id_set and (text := _text(block))
    )
    return tuple(strong_sources), weak_sources, tuple(dict.fromkeys(strong_block_ids))


def _metadata_value(
    strong_sources: tuple[_MetadataSource, ...],
    weak_sources: tuple[_MetadataSource, ...],
    extractor: Callable[[str], str | None],
) -> tuple[str | None, str | None]:
    for text, block_id in strong_sources:
        value = extractor(text)
        if value is not None:
            return value, block_id
    for text, block_id in weak_sources:
        value = extractor(text)
        if value is not None:
            return value, block_id
    return None, None


def _is_consecutive_numeric_sequence(sequence: tuple[str, ...]) -> bool:
    if len(sequence) < 2 or any(re.fullmatch(r"\d+", item) is None for item in sequence):
        return False
    values = [int(item) for item in sequence]
    return all(
        current == previous + 1
        for previous, current in zip(values, values[1:], strict=False)
    )


def _has_high_answer_density(document: DocumentIR, features: DocumentFeatures) -> bool:
    answer_block_ids = set(features.answer_evidence_blocks) - set(features.answer_heading_blocks)
    if len(answer_block_ids) < 3:
        return False
    content_block_count = sum(
        1 for block in document.blocks if (text := _text(block)) and not _is_answer_heading(text)
    )
    return content_block_count > 0 and len(answer_block_ids) / content_block_count >= 0.5


def extract_document_features(document: DocumentIR) -> DocumentFeatures:
    blocks = document.blocks
    title, title_blocks = _first_title(blocks)
    strong_sources, weak_sources, strong_metadata_blocks = _metadata_sources(document)

    selected_metadata_blocks: list[str] = []

    subject, source_block = _metadata_value(
        strong_sources, weak_sources, lambda text: _first_keyword(text, _SUBJECTS)
    )
    if source_block is not None:
        selected_metadata_blocks.append(source_block)

    grade, source_block = _metadata_value(
        strong_sources, weak_sources, lambda text: _first_keyword(text, _GRADES)
    )
    if source_block is not None:
        selected_metadata_blocks.append(source_block)

    year, source_block = _metadata_value(strong_sources, weak_sources, _year_from_text)
    if source_block is not None:
        selected_metadata_blocks.append(source_block)

    city, source_block = _metadata_value(strong_sources, weak_sources, _city_from_text)
    if source_block is not None:
        selected_metadata_blocks.append(source_block)

    exam_type, source_block = _metadata_value(
        strong_sources, weak_sources, lambda text: _first_keyword(text, _EXAM_TYPES)
    )
    if source_block is not None:
        selected_metadata_blocks.append(source_block)

    question_numbers: list[str] = []
    answer_numbers: list[str] = []
    question_blocks: list[str] = []
    answer_blocks: list[str] = []
    option_blocks: list[str] = []
    heading_blocks: list[str] = []
    in_answer_section = False

    for block in blocks:
        text = _text(block)
        if not text:
            continue
        if _is_answer_heading(text):
            in_answer_section = True
            heading_blocks.append(block.block_id)
            answer_blocks.append(block.block_id)
            continue

        numbered = _numbered_payload(block, text)
        if numbered is not None:
            number, payload = numbered
            if in_answer_section:
                answer_numbers.append(number)
                answer_blocks.append(block.block_id)
            elif _SIMPLE_ANSWER.fullmatch(payload.strip()) is not None:
                answer_numbers.append(number)
                answer_blocks.append(block.block_id)
            else:
                question_numbers.append(number)
                question_blocks.append(block.block_id)

        if _OPTION_STRUCTURE.search(text) is not None:
            option_blocks.append(block.block_id)
            if block.block_id not in question_blocks:
                question_blocks.append(block.block_id)

    metadata_source_blocks = tuple(
        dict.fromkeys(
            [
                *title_blocks,
                *strong_metadata_blocks,
                *(
                    block.block_id
                    for block in blocks
                    if block.type == "footer" and _text(block)
                ),
                *selected_metadata_blocks,
            ]
        )
    )

    return DocumentFeatures(
        title=title,
        subject=subject,
        grade=grade,
        year=year,
        city=city,
        exam_type=exam_type,
        question_number_sequence=tuple(question_numbers),
        answer_number_sequence=tuple(answer_numbers),
        metadata_source_blocks=metadata_source_blocks,
        question_evidence_blocks=tuple(dict.fromkeys(question_blocks)),
        answer_evidence_blocks=tuple(dict.fromkeys(answer_blocks)),
        option_structure_blocks=tuple(dict.fromkeys(option_blocks)),
        answer_heading_blocks=tuple(heading_blocks),
    )


def _rule_classification(document: DocumentIR, features: DocumentFeatures) -> DocumentClass:
    filename = Path(document.source_file).stem
    has_answer_name = "答案" in filename or "解析" in filename
    has_question = bool(features.question_number_sequence or features.option_structure_blocks)
    has_answer_evidence = bool(features.answer_number_sequence or features.answer_heading_blocks)
    has_explicit_answer_section = bool(features.answer_heading_blocks)
    has_strong_answer = (
        has_explicit_answer_section
        or has_answer_name
        or _is_consecutive_numeric_sequence(features.answer_number_sequence)
        or _has_high_answer_density(document, features)
    )

    if has_question and has_answer_evidence:
        if has_explicit_answer_section:
            return DocumentClass.QUESTION_AND_ANSWER
        return DocumentClass.MIXED
    if has_question:
        return DocumentClass.QUESTION
    if has_answer_evidence and has_strong_answer:
        return DocumentClass.ANSWER
    if has_answer_evidence:
        return DocumentClass.UNKNOWN
    return DocumentClass.UNKNOWN


def classify_document(
    document: DocumentIR,
    *,
    llm_evidence: LLMClassificationEvidence | None = None,
) -> DocumentUnderstanding:
    features = extract_document_features(document)
    rule_classification = _rule_classification(document, features)
    source_evidence = tuple(
        dict.fromkeys(
            [
                *features.metadata_source_blocks,
                *features.question_evidence_blocks,
                *features.answer_evidence_blocks,
                *features.option_structure_blocks,
                *features.answer_heading_blocks,
            ]
        )
    )

    final_classification = rule_classification
    if llm_evidence is not None and rule_classification in {
        DocumentClass.MIXED,
        DocumentClass.UNKNOWN,
    }:
        final_classification = llm_evidence.document_class

    return DocumentUnderstanding(
        document_id=document.document_id,
        source_file=document.source_file,
        document_class=final_classification,
        rule_classification=rule_classification,
        features=features,
        source_evidence=source_evidence,
        llm_evidence=llm_evidence,
    )


def parse_llm_classification(payload: str, document: DocumentIR) -> LLMClassificationEvidence:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ClassificationContractError("LLM classification must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ClassificationContractError("LLM classification must be a JSON object")

    allowed = ", ".join(item.value for item in DocumentClass)
    class_value: object = parsed.get("document_class")
    if not isinstance(class_value, str):
        raise ClassificationContractError(
            f"document_class must be one allowed document_class: {allowed}"
        )
    try:
        document_class = DocumentClass(class_value)
    except ValueError as exc:
        raise ClassificationContractError(
            f"document_class must be one allowed document_class: {allowed}"
        ) from exc

    cited_value: object = parsed.get("cited_block_ids")
    if not isinstance(cited_value, list) or not cited_value:
        raise ClassificationContractError("cited_block_ids must be a non-empty string list")
    cited: list[str] = []
    for item in cited_value:
        if not isinstance(item, str) or not item:
            raise ClassificationContractError(
                "cited_block_ids must be a non-empty string list"
            )
        cited.append(item)

    known_ids = {block.block_id for block in document.blocks}
    unknown_ids = [block_id for block_id in cited if block_id not in known_ids]
    if unknown_ids:
        raise ClassificationContractError(f"unknown cited block: {unknown_ids[0]}")

    reason_value: object = parsed.get("reason", "")
    if not isinstance(reason_value, str):
        raise ClassificationContractError("reason must be a string when present")

    return LLMClassificationEvidence(
        document_class=document_class,
        cited_block_ids=tuple(dict.fromkeys(cited)),
        reason=reason_value,
    )
