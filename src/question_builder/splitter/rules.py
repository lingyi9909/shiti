from __future__ import annotations

import re
from dataclasses import dataclass

from question_builder.domain.document import ContentBlock, DocumentIR


@dataclass(frozen=True, slots=True)
class RuleRange:
    content_blocks: tuple[str, ...]
    question_number: str | None
    score: float
    evidence: tuple[str, ...]


_ANSWER_HEADINGS = {
    "答案",
    "参考答案",
    "答案解析",
    "答案与解析",
    "参考答案及解析",
    "解析",
}
_SECTION_HEADING = re.compile(
    r"^\s*[一二三四五六七八九十百]+[、.．]\s*"
    r"(?:单项选择题|多项选择题|选择题|填空题|判断题|计算题|解答题|综合题|阅读题|作文题)\s*$"
)
_RAW_ARABIC_TOP = re.compile(r"^\s*(\d{1,4})\s*[.、．)）]\s*.+$")
_RAW_CHINESE_TOP = re.compile(r"^\s*([一二三四五六七八九十百]+)\s*[、.．]\s*.+$")
_RAW_SUBQUESTION = re.compile(
    r"^\s*(?:[（(]\s*\d{1,4}\s*[）)]|[①-⑳]|\d{1,4}[.．]\d{1,4})"
)
_OPTION_LINE = re.compile(r"(?:^|\s)A[.、．)]\s*.+?(?:\s+)B[.、．)]\s*", re.IGNORECASE)
_CIRCLED_NUMBER_START = ord("①")
_CIRCLED_NUMBER_END = ord("⑳")


def _text(block: ContentBlock) -> str:
    return (block.normalized_text if block.normalized_text is not None else block.raw_text).strip()


def _is_answer_heading(text: str) -> bool:
    return re.sub(r"\s+", "", text) in _ANSWER_HEADINGS


def _is_section_heading(text: str) -> bool:
    return _SECTION_HEADING.fullmatch(text) is not None


def is_answer_heading_block(block: ContentBlock) -> bool:
    return _is_answer_heading(_text(block))


def is_section_heading_block(block: ContentBlock) -> bool:
    return _is_section_heading(_text(block))


def is_deterministically_excluded_question_block(block: ContentBlock) -> bool:
    return block.type in {"header", "footer", "noise_candidate"} or is_section_heading_block(block)


def _normalize_label(label: str) -> str | None:
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
    if re.fullmatch(r"[一二三四五六七八九十百]+", normalized):
        return normalized
    if re.fullmatch(r"[A-Za-z]", normalized):
        return normalized.upper()
    return None


def _structured_label(block: ContentBlock) -> str | None:
    if block.numbering is None:
        return None
    value = block.numbering.get("resolved_label")
    return value if isinstance(value, str) else None


def _structured_marker(block: ContentBlock, text: str) -> tuple[str, str] | None:
    label = _structured_label(block)
    if label is None:
        return None
    normalized = _normalize_label(label)
    if normalized is None:
        return None
    compact = re.sub(r"\s+", "", label).replace("．", ".")
    if _is_section_heading(text):
        return None
    if compact.startswith(("(", "（")) or compact[:1] in "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳":
        return ("sub", normalized)
    if "." in normalized and normalized.count(".") >= 1:
        return ("sub", normalized)
    if re.fullmatch(r"[A-Za-z]", normalized):
        return ("sub", normalized)
    return ("top", normalized)


def _raw_marker(text: str) -> tuple[str, str] | None:
    if _is_section_heading(text) or _RAW_SUBQUESTION.match(text) is not None:
        return None
    match = _RAW_ARABIC_TOP.match(text)
    if match is not None:
        return ("top", match.group(1))
    match = _RAW_CHINESE_TOP.match(text)
    if match is not None:
        return ("top", match.group(1))
    return None


def question_marker(block: ContentBlock) -> tuple[str, str, str] | None:
    text = _text(block)
    if not text or _is_answer_heading(text):
        return None
    structured = _structured_marker(block, text)
    if structured is not None:
        kind, number = structured
        return kind, number, "word_numbering"
    raw = _raw_marker(text)
    if raw is not None:
        kind, number = raw
        return kind, number, "raw_numbering"
    return None


def question_number_for_block(block: ContentBlock) -> str | None:
    marker = question_marker(block)
    if marker is None or marker[0] != "top":
        return None
    return marker[1]


def generate_rule_ranges(document: DocumentIR) -> tuple[RuleRange, ...]:
    blocks = document.blocks
    answer_start = len(blocks)
    top_level_starts: list[tuple[int, str, str]] = []

    for index, block in enumerate(blocks):
        text = _text(block)
        if _is_answer_heading(text):
            answer_start = index
            break
        marker = question_marker(block)
        if marker is None or marker[0] != "top":
            continue
        _, number, source = marker
        top_level_starts.append((index, number, source))

    if not top_level_starts:
        return ()

    ranges: list[RuleRange] = []
    for position, (start, number, source) in enumerate(top_level_starts):
        end = (
            top_level_starts[position + 1][0]
            if position + 1 < len(top_level_starts)
            else answer_start
        )
        selected = tuple(
            block.block_id
            for block in blocks[start:end]
            if not is_deterministically_excluded_question_block(block)
            and not _is_answer_heading(_text(block))
        )
        if not selected:
            continue
        evidence = ["strong_top_level_number", source]
        if any(_OPTION_LINE.search(_text(block)) is not None for block in blocks[start:end]):
            evidence.append("option_structure")
        ranges.append(
            RuleRange(
                content_blocks=selected,
                question_number=number,
                score=1.0,
                evidence=tuple(evidence),
            )
        )

    return tuple(ranges)
