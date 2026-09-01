from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from question_builder.domain.document import DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.domain.question import QuestionCandidate
from question_builder.splitter.rules import (
    RuleRange,
    generate_rule_ranges,
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


def _positions(document: DocumentIR) -> dict[str, int]:
    return {block.block_id: index for index, block in enumerate(document.blocks)}


def _require_exact_keys(value: dict[object, object], allowed: set[str], *, where: str) -> None:
    keys = set(value)
    if keys != allowed:
        raise QuestionSplitContractError(f"{where} may contain only: {', '.join(sorted(allowed))}")


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

        confidence = raw_range.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise QuestionSplitContractError("confidence must be a number between 0 and 1")
        normalized_confidence = float(confidence)
        if not 0.0 <= normalized_confidence <= 1.0:
            raise QuestionSplitContractError("confidence must be a number between 0 and 1")

        ranges.append(
            LLMSplitRange(
                content_blocks=tuple(content_blocks),
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
            text = (
                block.normalized_text
                if block.normalized_text is not None
                else block.raw_text
            ).strip()
            if not text:
                return False
        if block.type == "table":
            rendered = block.metadata.get("rendered")
            if not isinstance(rendered, str) or not rendered.strip():
                return False
    return True


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

    if llm_selection is not None:
        ranges: tuple[LLMSplitRange | RuleRange, ...] = llm_selection.ranges
    else:
        ranges = generate_rule_ranges(document)

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
        if isinstance(split_range, LLMSplitRange):
            candidates.append(_candidate_from_llm(document, split_range))
        else:
            candidates.append(_candidate_from_rule(document, split_range))

    return candidates
