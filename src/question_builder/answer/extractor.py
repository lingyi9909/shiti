from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

from question_builder.domain.answer import AnswerCandidate
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.export.markdown import QuestionContentError, render_question_markdown


class AnswerExtractContractError(ValueError):
    pass


class AnswerExtractionError(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class LLMAnswerItem:
    question_number: str | None
    answer: str
    analysis: str
    source_blocks: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class LLMAnswerSelection:
    answers: tuple[LLMAnswerItem, ...]


_ANSWER_HEADINGS = {
    "答案",
    "参考答案",
    "答案解析",
    "答案与解析",
    "参考答案及解析",
    "解析",
}
_EXCLUDED_TYPES = {"header", "footer", "noise_candidate"}
_CRITICAL_ANSWER_TYPES = {"formula", "image", "table", "unresolved"}
_RAW_NUMBERED = re.compile(
    r"^\s*(?P<number>\d{1,4}|[一二三四五六七八九十百]+)\s*[.、．)）]\s*(?P<body>.+?)\s*$"
)
_COMPACT_SIMPLE_ANSWER = re.compile(
    r"(?<!\S)(?P<number>\d{1,4})\s*[.、．)）]\s*"
    r"(?P<answer>[A-D]|正确|错误|对|错|√|×)(?=\s|$)",
    re.IGNORECASE,
)
_COMPACT_NUMBER_MARKER = re.compile(
    r"(?P<prefix>^|\s+)(?P<number>\d{1,4})\s*[.、．)）]\s*"
)
_EXPLICIT_QUESTION_NUMBER = re.compile(
    r"^\s*第\s*(?P<number>\d{1,4}|[一二三四五六七八九十百]+)\s*题"
)
_ANALYSIS_MARKER = re.compile(r"\s*(?:解析|分析)\s*[：:]\s*")
_ANSWER_PREFIX = re.compile(r"^\s*(?:答案|答)\s*[：:]\s*")
_SOLUTION_PREFIX = re.compile(r"^\s*解\s*[：:]\s*")
_FINAL_ANSWER_MARKER = re.compile(
    r"(?:最终答案|最后答案|答案)\s*(?:为|是|[：:])\s*",
)


def _text(block: ContentBlock) -> str:
    value = block.normalized_text if block.normalized_text is not None else block.raw_text
    return value.strip()


def _compact_heading(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _answer_region(document: DocumentIR) -> tuple[tuple[ContentBlock, ...], bool]:
    for index, block in enumerate(document.blocks):
        if _compact_heading(_text(block)) in _ANSWER_HEADINGS:
            return document.blocks[index + 1 :], True
    return document.blocks, False


def _normalize_numbering_label(label: str) -> str | None:
    value = re.sub(r"\s+", "", label).replace("．", ".")
    if value.startswith(("(", "（")) and value.endswith((")", "）")):
        value = value[1:-1]
    value = value.rstrip(".、)）")
    if re.fullmatch(r"\d{1,4}|[一二三四五六七八九十百]+", value):
        return value
    return None


def _numbered_body(block: ContentBlock) -> tuple[str | None, str] | None:
    if block.numbering is not None:
        raw_label = block.numbering.get("resolved_label")
        if isinstance(raw_label, str):
            number = _normalize_numbering_label(raw_label)
            if number is not None and _text(block):
                return number, _text(block)

    match = _RAW_NUMBERED.match(_text(block))
    if match is None:
        return None
    return match.group("number"), match.group("body").strip()


def _require_exact_keys(value: dict[object, object], allowed: set[str], *, where: str) -> None:
    if set(value) != allowed:
        raise AnswerExtractContractError(
            f"{where} may contain only: {', '.join(sorted(allowed))}"
        )


def parse_llm_answer_extract(payload: str, document: DocumentIR) -> LLMAnswerSelection:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AnswerExtractContractError("LLM answer output must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise AnswerExtractContractError("LLM answer output must be a JSON object")
    _require_exact_keys(parsed, {"answers"}, where="LLM answer output")

    raw_answers = parsed.get("answers")
    if not isinstance(raw_answers, list) or not raw_answers:
        raise AnswerExtractContractError("answers must be a non-empty list")

    positions = {block.block_id: index for index, block in enumerate(document.blocks)}
    answers: list[LLMAnswerItem] = []
    for index, raw_item in enumerate(raw_answers):
        if not isinstance(raw_item, dict):
            raise AnswerExtractContractError("each answer must be an object")
        _require_exact_keys(
            raw_item,
            {"question_number", "answer", "analysis", "source_blocks", "confidence"},
            where=f"answer {index}",
        )

        question_number = raw_item.get("question_number")
        if question_number is not None and (
            not isinstance(question_number, str) or not question_number.strip()
        ):
            raise AnswerExtractContractError("question_number must be a non-empty string or null")

        answer = raw_item.get("answer")
        analysis = raw_item.get("analysis")
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerExtractContractError("answer must be a non-empty string")
        if not isinstance(analysis, str) or not analysis.strip():
            raise AnswerExtractContractError("analysis must be a non-empty string")

        raw_source_blocks = raw_item.get("source_blocks")
        if not isinstance(raw_source_blocks, list) or not raw_source_blocks:
            raise AnswerExtractContractError("source_blocks must be a non-empty list")
        source_blocks: list[str] = []
        for block_id in raw_source_blocks:
            if not isinstance(block_id, str) or not block_id:
                raise AnswerExtractContractError("source_blocks must contain non-empty strings")
            if block_id not in positions:
                raise AnswerExtractContractError(f"unknown source block: {block_id}")
            source_blocks.append(block_id)
        if len(source_blocks) != len(set(source_blocks)):
            raise AnswerExtractContractError("source_blocks must not contain duplicates")
        source_positions = [positions[block_id] for block_id in source_blocks]
        if source_positions != sorted(source_positions):
            raise AnswerExtractContractError("source_blocks must preserve source order")

        confidence = raw_item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise AnswerExtractContractError("confidence must be a number between 0 and 1")
        normalized_confidence = float(confidence)
        if not 0.0 <= normalized_confidence <= 1.0:
            raise AnswerExtractContractError("confidence must be a number between 0 and 1")

        answers.append(
            LLMAnswerItem(
                question_number=question_number.strip() if question_number is not None else None,
                answer=answer.strip(),
                analysis=analysis.strip(),
                source_blocks=tuple(source_blocks),
                confidence=normalized_confidence,
            )
        )

    return LLMAnswerSelection(answers=tuple(answers))


def _candidate_id(
    document_id: str,
    question_number: str | None,
    source_blocks: tuple[str, ...],
    answer: str,
) -> str:
    material = (
        f"{document_id}\0{question_number or ''}\0{'\0'.join(source_blocks)}\0{answer}"
    ).encode()
    return "ac_" + hashlib.sha256(material).hexdigest()[:16]


def _build_candidate(
    document: DocumentIR,
    *,
    question_number: str | None,
    answer: str,
    analysis: str,
    source_blocks: tuple[str, ...],
    score: float,
) -> AnswerCandidate:
    return AnswerCandidate(
        answer_candidate_id=_candidate_id(
            document.document_id,
            question_number,
            source_blocks,
            answer,
        ),
        document_id=document.document_id,
        question_number=question_number,
        answer=answer,
        analysis=analysis,
        source_blocks=source_blocks,
        extract_score=score,
    )


def _compact_answer_candidates(
    document: DocumentIR,
    block: ContentBlock,
) -> list[AnswerCandidate]:
    matches = list(_COMPACT_SIMPLE_ANSWER.finditer(_text(block)))
    if len(matches) < 2:
        return []
    return [
        _build_candidate(
            document,
            question_number=match.group("number"),
            answer=match.group("answer").upper(),
            analysis="略",
            source_blocks=(block.block_id,),
            score=1.0,
        )
        for match in matches
    ]


def _compact_numbered_bodies(block: ContentBlock) -> list[tuple[str, str]] | None:
    text = _text(block)
    matches = list(_COMPACT_NUMBER_MARKER.finditer(text))
    if len(matches) < 2:
        return None
    if text[: matches[0].start()].strip():
        raise AnswerExtractionError(
            RejectReason.ANSWER_NOT_FOUND,
            "compact answer source cannot be completely parsed",
        )

    numbers = [int(match.group("number")) for match in matches]
    if any(current != previous + 1 for previous, current in zip(numbers, numbers[1:])):
        raise AnswerExtractionError(
            RejectReason.ANSWER_NOT_FOUND,
            "compact answer numbering is ambiguous",
        )

    entries: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            raise AnswerExtractionError(
                RejectReason.ANSWER_NOT_FOUND,
                "compact answer source contains an empty numbered answer",
            )
        entries.append((match.group("number"), body))
    return entries


def _split_answer_and_analysis(body: str) -> tuple[str, str] | None:
    marker = _ANALYSIS_MARKER.search(body)
    if marker is None:
        return None
    answer_part = _ANSWER_PREFIX.sub("", body[: marker.start()]).strip()
    analysis = body[marker.end() :].strip()
    if not answer_part or not analysis:
        return None
    return answer_part, analysis


def _solution_answer(body: str) -> tuple[str, str] | None:
    prefix = _SOLUTION_PREFIX.match(body)
    if prefix is None:
        return None
    solution = body[prefix.end() :].strip()
    marker = _FINAL_ANSWER_MARKER.search(solution)
    if marker is None:
        raise AnswerExtractionError(
            RejectReason.ANSWER_NOT_FOUND,
            "long-form solution has no reliable original final answer",
        )

    process = solution[: marker.start()].strip()
    final_region = solution[marker.end() :].strip()
    trailing_analysis = ""
    analysis_marker = _ANALYSIS_MARKER.search(final_region)
    if analysis_marker is not None:
        answer = final_region[: analysis_marker.start()].strip()
        trailing_analysis = final_region[analysis_marker.end() :].strip()
    else:
        answer = final_region

    if not answer:
        raise AnswerExtractionError(
            RejectReason.ANSWER_NOT_FOUND,
            "long-form solution final answer marker has no source answer",
        )

    analysis_parts = [part for part in (process, trailing_analysis) if part]
    return answer, "\n".join(analysis_parts) or "略"


def _render_answer_block(document: DocumentIR, block: ContentBlock) -> str | None:
    if block.type in _CRITICAL_ANSWER_TYPES:
        try:
            return render_question_markdown(document, (block.block_id,))
        except QuestionContentError as exc:
            raise AnswerExtractionError(
                RejectReason.ANSWER_NOT_FOUND,
                f"critical answer source cannot be reconstructed: {block.block_id}",
            ) from exc
    return _text(block) or None


def _numbered_entries(
    document: DocumentIR,
    blocks: tuple[ContentBlock, ...],
) -> list[tuple[str, tuple[str, ...], str]]:
    entries: list[tuple[str, list[str], list[str]]] = []
    for block in blocks:
        if block.type in _EXCLUDED_TYPES:
            continue

        compact = _compact_numbered_bodies(block)
        if compact is not None:
            for compact_number, compact_body in compact:
                entries.append((compact_number, [block.block_id], [compact_body]))
            continue

        numbered = _numbered_body(block)
        if numbered is not None:
            numbered_number, numbered_body = numbered
            if numbered_number is not None:
                entries.append((numbered_number, [block.block_id], [numbered_body]))
            continue

        if entries:
            rendered = _render_answer_block(document, block)
            if rendered:
                entries[-1][1].append(block.block_id)
                entries[-1][2].append(rendered)

    return [
        (number, tuple(block_ids), "\n\n".join(parts).strip())
        for number, block_ids, parts in entries
    ]


def _deterministic_candidates(document: DocumentIR) -> list[AnswerCandidate]:
    region, explicit_section = _answer_region(document)
    blocks = tuple(block for block in region if block.type not in _EXCLUDED_TYPES)

    candidates: list[AnswerCandidate] = []
    for number, source_blocks, body in _numbered_entries(document, blocks):
        solution = _solution_answer(body)
        if solution is not None:
            answer, analysis = solution
        else:
            split = _split_answer_and_analysis(body)
            if split is not None:
                answer, analysis = split
            elif explicit_section:
                answer = _ANSWER_PREFIX.sub("", body).strip()
                analysis = "略"
            elif _ANSWER_PREFIX.match(body) is not None:
                answer = _ANSWER_PREFIX.sub("", body).strip()
                analysis = "略"
            else:
                continue

        if answer:
            candidates.append(
                _build_candidate(
                    document,
                    question_number=number,
                    answer=answer,
                    analysis=analysis,
                    source_blocks=source_blocks,
                    score=1.0,
                )
            )

    if candidates:
        return candidates

    # A source can contain a strong explicit answer marker without numbering.
    for block in blocks:
        text = _text(block)
        if _ANSWER_PREFIX.match(text) is None:
            continue
        split = _split_answer_and_analysis(text)
        if split is not None:
            answer, analysis = split
        else:
            answer = _ANSWER_PREFIX.sub("", text).strip()
            analysis = "略"
        if answer:
            return [
                _build_candidate(
                    document,
                    question_number=None,
                    answer=answer,
                    analysis=analysis,
                    source_blocks=(block.block_id,),
                    score=1.0,
                )
            ]

    return []


def _normalize_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized)


def _structured_number_evidence(block: ContentBlock) -> str | None:
    if block.numbering is None:
        return None
    raw_label = block.numbering.get("resolved_label")
    if not isinstance(raw_label, str):
        return None
    return _normalize_numbering_label(raw_label)


def _text_number_evidence(block: ContentBlock) -> str | None:
    text = _text(block)
    raw_numbered = _RAW_NUMBERED.match(text)
    if raw_numbered is not None:
        return raw_numbered.group("number")
    explicit = _EXPLICIT_QUESTION_NUMBER.match(text)
    if explicit is not None:
        return explicit.group("number")
    return None


def _llm_question_number_is_source_backed(
    item: LLMAnswerItem,
    blocks_by_id: dict[str, ContentBlock],
) -> bool:
    if item.question_number is None:
        return True
    expected = _normalize_numbering_label(item.question_number)
    if expected is None:
        return False

    structured = [
        number
        for block_id in item.source_blocks
        if (number := _structured_number_evidence(blocks_by_id[block_id])) is not None
    ]
    if structured:
        return len(set(structured)) == 1 and structured[0] == expected

    textual = [
        number
        for block_id in item.source_blocks
        if (number := _text_number_evidence(blocks_by_id[block_id])) is not None
    ]
    return bool(textual) and len(set(textual)) == 1 and textual[0] == expected


def _render_llm_source_evidence(
    document: DocumentIR,
    source_blocks: tuple[str, ...],
) -> str:
    try:
        return render_question_markdown(document, source_blocks)
    except QuestionContentError as exc:
        raise AnswerExtractionError(
            RejectReason.ANSWER_NOT_FOUND,
            "LLM cited answer source cannot be reliably reconstructed",
        ) from exc


def _llm_item_is_source_backed(
    item: LLMAnswerItem,
    document: DocumentIR,
    blocks_by_id: dict[str, ContentBlock],
) -> bool:
    if not _llm_question_number_is_source_backed(item, blocks_by_id):
        return False
    source = _render_llm_source_evidence(document, item.source_blocks)
    normalized_source = _normalize_evidence(source)
    if _normalize_evidence(item.answer) not in normalized_source:
        return False
    if item.analysis != "略" and _normalize_evidence(item.analysis) not in normalized_source:
        return False
    return True


def _llm_candidates(
    document: DocumentIR,
    selection: LLMAnswerSelection,
) -> list[AnswerCandidate]:
    blocks_by_id = {block.block_id: block for block in document.blocks}
    answer_region, explicit_section = _answer_region(document)
    allowed_source_ids = {block.block_id for block in answer_region}
    candidates: list[AnswerCandidate] = []
    for item in selection.answers:
        if explicit_section and any(
            block_id not in allowed_source_ids for block_id in item.source_blocks
        ):
            raise AnswerExtractionError(
                RejectReason.ANSWER_NOT_FOUND,
                "LLM answer source is outside the explicit answer section",
            )
        if not _llm_item_is_source_backed(item, document, blocks_by_id):
            raise AnswerExtractionError(
                RejectReason.ANSWER_NOT_FOUND,
                "LLM answer is not recoverable from its cited original source blocks",
            )
        candidates.append(
            _build_candidate(
                document,
                question_number=item.question_number,
                answer=item.answer,
                analysis=item.analysis,
                source_blocks=item.source_blocks,
                score=item.confidence,
            )
        )
    return candidates


def extract_answer_candidates(
    document: DocumentIR,
    *,
    llm_selection: LLMAnswerSelection | None = None,
) -> list[AnswerCandidate]:
    deterministic = _deterministic_candidates(document)
    if deterministic:
        return deterministic

    if llm_selection is not None:
        llm_candidates = _llm_candidates(document, llm_selection)
        if llm_candidates:
            return llm_candidates

    raise AnswerExtractionError(
        RejectReason.ANSWER_NOT_FOUND,
        "no reliable original answer could be extracted from source content",
    )
