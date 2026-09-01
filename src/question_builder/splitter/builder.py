from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.splitter.rules import (
    RuleRange,
    generate_rule_ranges,
    is_answer_heading_block,
    is_deterministically_excluded_question_block,
    question_number_for_block,
)


class QuestionSplitContractError(ValueError):
    pass


class QuestionSplitError(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class LLMSplitRange:
    content_blocks: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class LLMSplitSelection:
    ranges: tuple[LLMSplitRange, ...]


_CRITICAL_TYPES = {"formula", "image", "table", "unresolved"}
_OPTION_LABEL = re.compile(r"(?:^|\s)([A-D])[.、．)]\s*", re.IGNORECASE)


def _positions(document: DocumentIR) -> dict[str, int]:
    return {block.block_id: index for index, block in enumerate(document.blocks)}


def _text(block: ContentBlock) -> str:
    return (block.normalized_text if block.normalized_text is not None else block.raw_text).strip()


def _answer_start_index(document: DocumentIR) -> int:
    for index, block in enumerate(document.blocks):
        if is_answer_heading_block(block):
            return index
    return len(document.blocks)


def _require_exact_keys(value: dict[object, object], allowed: set[str], *, where: str) -> None:
    keys = set(value)
    if keys != allowed:
        raise QuestionSplitContractError(f"{where} may contain only: {', '.join(sorted(allowed))}")


def _validate_llm_range_boundary(
    document: DocumentIR,
    content_blocks: tuple[str, ...],
    *,
    positions: dict[str, int],
    answer_start: int,
) -> None:
    block_positions = [positions[block_id] for block_id in content_blocks]
    if any(position >= answer_start for position in block_positions):
        raise QuestionSplitContractError("LLM range must not include the answer section")

    selected = set(content_blocks)
    for block in document.blocks[block_positions[0] : block_positions[-1] + 1]:
        if block.block_id in selected:
            continue
        if block.type in _CRITICAL_TYPES:
            continue
        if is_deterministically_excluded_question_block(block):
            continue
        raise QuestionSplitContractError(
            "LLM range must select a continuous question boundary without skipping body blocks"
        )


def parse_llm_split(payload: str, document: DocumentIR) -> LLMSplitSelection:
    try:
        parsed: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise QuestionSplitContractError("LLM split output must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise QuestionSplitContractError("LLM split output must be a JSON object")
    _require_exact_keys(parsed, {"ranges"}, where="LLM split output")

    raw_ranges = parsed.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise QuestionSplitContractError("ranges must be a non-empty list")

    positions = _positions(document)
    answer_start = _answer_start_index(document)
    previous_last = -1
    ranges: list[LLMSplitRange] = []

    for index, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, dict):
            raise QuestionSplitContractError("each range must be an object")
        _require_exact_keys(
            raw_range,
            {"content_blocks", "confidence"},
            where=f"range {index}",
        )

        raw_blocks = raw_range.get("content_blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise QuestionSplitContractError("content_blocks must be a non-empty list")
        content_blocks: list[str] = []
        for block_id in raw_blocks:
            if not isinstance(block_id, str) or not block_id:
                raise QuestionSplitContractError("content_blocks must contain non-empty strings")
            if block_id not in positions:
                raise QuestionSplitContractError(f"unknown block: {block_id}")
            content_blocks.append(block_id)
        if len(content_blocks) != len(set(content_blocks)):
            raise QuestionSplitContractError("content_blocks must not contain duplicates")

        block_positions = [positions[block_id] for block_id in content_blocks]
        if block_positions != sorted(block_positions):
            raise QuestionSplitContractError("content_blocks must preserve source order")
        if block_positions[0] <= previous_last:
            raise QuestionSplitContractError("ranges must preserve source order and not overlap")
        previous_last = block_positions[-1]

        normalized_blocks = tuple(content_blocks)
        _validate_llm_range_boundary(
            document,
            normalized_blocks,
            positions=positions,
            answer_start=answer_start,
        )

        confidence = raw_range.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise QuestionSplitContractError("confidence must be a number between 0 and 1")
        normalized_confidence = float(confidence)
        if not 0.0 <= normalized_confidence <= 1.0:
            raise QuestionSplitContractError("confidence must be a number between 0 and 1")

        ranges.append(
            LLMSplitRange(
                content_blocks=normalized_blocks,
                confidence=normalized_confidence,
            )
        )

    return LLMSplitSelection(ranges=tuple(ranges))


def _critical_content_complete(document: DocumentIR, content_blocks: tuple[str, ...]) -> bool:
    positions = _positions(document)
    selected = set(content_blocks)
    selected_positions = [positions[block_id] for block_id in content_blocks]
    start, end = selected_positions[0], selected_positions[-1]

    for block in document.blocks[start : end + 1]:
        if block.type in _CRITICAL_TYPES and block.block_id not in selected:
            return False
        if block.block_id not in selected:
            continue
        if block.type == "unresolved":
            return False
        if block.type == "image":
            asset_filename = block.metadata.get("asset_filename")
            if not isinstance(asset_filename, str) or not asset_filename:
                return False
        if block.type == "formula":
            if not _text(block):
                return False
        if block.type == "table":
            rendered = block.metadata.get("rendered")
            if not isinstance(rendered, str) or not rendered.strip():
                return False
    return True


def _option_structure_consistent(document: DocumentIR, content_blocks: tuple[str, ...]) -> bool:
    blocks_by_id = {block.block_id: block for block in document.blocks}
    labels: list[str] = []
    for block_id in content_blocks:
        labels.extend(
            label.upper()
            for label in _OPTION_LABEL.findall(_text(blocks_by_id[block_id]))
        )

    if not labels:
        return True
    if labels[0] != "A":
        return False

    previous = labels[0]
    for label in labels[1:]:
        if label == "A":
            if previous == "A":
                return False
            previous = label
            continue
        if previous == "D":
            return False
        expected = chr(ord(previous) + 1)
        if label != expected:
            return False
        previous = label
    return True


def _candidate_set_assigns_all_critical_blocks(
    document: DocumentIR,
    ranges: tuple[LLMSplitRange | RuleRange, ...],
) -> bool:
    positions = _positions(document)
    first = min(positions[split_range.content_blocks[0]] for split_range in ranges)
    last = max(positions[split_range.content_blocks[-1]] for split_range in ranges)
    answer_start = _answer_start_index(document)
    last = min(last, answer_start - 1)

    assignments: dict[str, int] = {}
    for split_range in ranges:
        for block_id in split_range.content_blocks:
            assignments[block_id] = assignments.get(block_id, 0) + 1

    for block in document.blocks[first : last + 1]:
        if block.type in _CRITICAL_TYPES and assignments.get(block.block_id, 0) != 1:
            return False
    return True


def _llm_selection_crosses_answer_section(
    document: DocumentIR,
    selection: LLMSplitSelection,
) -> bool:
    positions = _positions(document)
    answer_start = _answer_start_index(document)
    for split_range in selection.ranges:
        for block_id in split_range.content_blocks:
            position = positions.get(block_id)
            if position is not None and position >= answer_start:
                return True
    return False


def _candidate_id(document_id: str, content_blocks: tuple[str, ...]) -> str:
    material = f"{document_id}\0{'\0'.join(content_blocks)}".encode()
    return "qc_" + hashlib.sha256(material).hexdigest()[:16]


def _candidate_from_rule(document: DocumentIR, split_range: RuleRange) -> QuestionCandidate:
    return QuestionCandidate(
        question_candidate_id=_candidate_id(document.document_id, split_range.content_blocks),
        document_id=document.document_id,
        content_blocks=split_range.content_blocks,
        question_number=split_range.question_number,
        question_type_candidate=None,
        split_score=split_range.score,
    )


def _candidate_from_llm(document: DocumentIR, split_range: LLMSplitRange) -> QuestionCandidate:
    blocks_by_id = {block.block_id: block for block in document.blocks}
    first_block = blocks_by_id[split_range.content_blocks[0]]
    return QuestionCandidate(
        question_candidate_id=_candidate_id(document.document_id, split_range.content_blocks),
        document_id=document.document_id,
        content_blocks=split_range.content_blocks,
        question_number=question_number_for_block(first_block),
        question_type_candidate=None,
        split_score=split_range.confidence,
    )


def build_question_candidates(
    document: DocumentIR,
    *,
    llm_selection: LLMSplitSelection | None = None,
    split_threshold: float = 0.98,
) -> list[QuestionCandidate]:
    if not 0.0 <= split_threshold <= 1.0:
        raise ValueError("split_threshold must be between 0 and 1")

    if llm_selection is not None and _llm_selection_crosses_answer_section(document, llm_selection):
        raise QuestionSplitError(
            RejectReason.QUESTION_CONTENT_INCOMPLETE,
            "LLM split selection crosses into the explicit answer section",
        )

    rule_ranges = generate_rule_ranges(document)
    if rule_ranges and all(split_range.score >= split_threshold for split_range in rule_ranges):
        ranges: tuple[LLMSplitRange | RuleRange, ...] = rule_ranges
    elif llm_selection is not None:
        ranges = llm_selection.ranges
    else:
        ranges = rule_ranges

    if not ranges:
        raise QuestionSplitError(
            RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE,
            "no high-confidence structural question boundary was found",
        )

    candidates: list[QuestionCandidate] = []
    for split_range in ranges:
        score = (
            split_range.confidence
            if isinstance(split_range, LLMSplitRange)
            else split_range.score
        )
        if score < split_threshold:
            raise QuestionSplitError(
                RejectReason.QUESTION_SPLIT_LOW_CONFIDENCE,
                f"split score {score:.3f} is below threshold {split_threshold:.3f}",
            )
        if not _critical_content_complete(document, split_range.content_blocks):
            raise QuestionSplitError(
                RejectReason.QUESTION_CONTENT_INCOMPLETE,
                "candidate omits or contains unresolved critical source content",
            )
        if not _option_structure_consistent(document, split_range.content_blocks):
            raise QuestionSplitError(
                RejectReason.QUESTION_CONTENT_INCOMPLETE,
                "candidate contains inconsistent choice option structure",
            )
        if isinstance(split_range, LLMSplitRange):
            candidates.append(_candidate_from_llm(document, split_range))
        else:
            candidates.append(_candidate_from_rule(document, split_range))

    if not _candidate_set_assigns_all_critical_blocks(document, ranges):
        raise QuestionSplitError(
            RejectReason.QUESTION_CONTENT_INCOMPLETE,
            "question candidate set leaves critical source content unassigned",
        )

    return candidates
