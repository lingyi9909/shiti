from __future__ import annotations

import json
import re
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
_NUMBERED_LINE = re.compile(r"^\s*(\d{1,4})\s*[.、．)）]\s*(.+?)\s*$")
_OPTION_STRUCTURE = re.compile(r"(?:^|\s)A[.、．)]\s*.+?(?:\s+)B[.、．)]\s*", re.IGNORECASE)
_SIMPLE_ANSWER = re.compile(
    r"^(?:[A-H](?:\s*[,，/]\s*[A-H])*|[-+]?\d+(?:\.\d+)?|[√×对错]|[TF])$",
    re.IGNORECASE,
)
_CITY = re.compile(r"([\u4e00-\u9fff]{2,8}市)")
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


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


def _simple_answer_number(text: str) -> str | None:
    match = _NUMBERED_LINE.match(text)
    if match is None:
        return None
    number, payload = match.groups()
    if _SIMPLE_ANSWER.fullmatch(payload.strip()) is None:
        return None
    return number


def _question_number(text: str) -> str | None:
    match = _NUMBERED_LINE.match(text)
    if match is None:
        return None
    number, payload = match.groups()
    if _SIMPLE_ANSWER.fullmatch(payload.strip()) is not None:
        return None
    return number


def _first_keyword(haystack: str, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in haystack:
            return candidate
    return None


def _first_title(blocks: tuple[ContentBlock, ...]) -> tuple[str | None, tuple[str, ...]]:
    for preferred_type in ("header", "paragraph", "textbox"):
        for block in blocks:
            text = _text(block)
            if block.type == preferred_type and text:
                return text, (block.block_id,)
    return None, ()


def extract_document_features(document: DocumentIR) -> DocumentFeatures:
    blocks = document.blocks
    title, title_blocks = _first_title(blocks)
    source_stem = Path(document.source_file).stem
    searchable = "\n".join([source_stem, *(text for block in blocks if (text := _text(block)))])

    subject = _first_keyword(searchable, _SUBJECTS)
    grade = _first_keyword(searchable, _GRADES)
    year_match = _YEAR.search(searchable)
    year = year_match.group(1) if year_match is not None else None
    city_match = _CITY.search(searchable)
    city = city_match.group(1) if city_match is not None else None
    exam_type = _first_keyword(searchable, _EXAM_TYPES)

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

        simple_answer = _simple_answer_number(text)
        numbered_question = _question_number(text)

        if in_answer_section:
            match = _NUMBERED_LINE.match(text)
            if match is not None:
                answer_numbers.append(match.group(1))
                answer_blocks.append(block.block_id)
        elif simple_answer is not None:
            answer_numbers.append(simple_answer)
            answer_blocks.append(block.block_id)
        elif numbered_question is not None:
            question_numbers.append(numbered_question)
            question_blocks.append(block.block_id)

        if _OPTION_STRUCTURE.search(text) is not None:
            option_blocks.append(block.block_id)
            if block.block_id not in question_blocks:
                question_blocks.append(block.block_id)

    metadata_source_blocks = tuple(
        dict.fromkeys(
            [
                *title_blocks,
                *(
                    block.block_id
                    for block in blocks
                    if block.type in {"header", "footer"} and _text(block)
                ),
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
    has_answer = bool(features.answer_number_sequence or features.answer_heading_blocks)
    has_explicit_answer_section = bool(features.answer_heading_blocks)

    if has_question and has_answer:
        if has_explicit_answer_section:
            return DocumentClass.QUESTION_AND_ANSWER
        return DocumentClass.MIXED
    if has_answer and (has_answer_name or has_explicit_answer_section):
        return DocumentClass.ANSWER
    if has_question:
        return DocumentClass.QUESTION
    if has_answer:
        return DocumentClass.ANSWER
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
