from __future__ import annotations

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


class QuestionContentError(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


_CRITICAL_TYPES = {"formula", "image", "table", "unresolved"}
_SKIP_TYPES = {"header", "footer", "noise_candidate"}


def _block_text(block: ContentBlock) -> str:
    return (
        block.normalized_text if block.normalized_text is not None else block.raw_text
    ).strip()


def _fail(message: str) -> QuestionContentError:
    return QuestionContentError(RejectReason.QUESTION_CONTENT_INCOMPLETE, message)


def _validate_selection(
    document: DocumentIR,
    content_blocks: tuple[str, ...],
) -> tuple[ContentBlock, ...]:
    if not content_blocks:
        raise _fail("question reconstruction requires at least one source block")

    blocks_by_id = {block.block_id: block for block in document.blocks}
    positions = {block.block_id: index for index, block in enumerate(document.blocks)}
    selected: list[ContentBlock] = []
    selected_positions: list[int] = []

    for block_id in content_blocks:
        block = blocks_by_id.get(block_id)
        if block is None:
            raise _fail(f"unknown source block: {block_id}")
        selected.append(block)
        selected_positions.append(positions[block_id])

    if (
        selected_positions != sorted(selected_positions)
        or len(selected_positions) != len(set(selected_positions))
    ):
        raise _fail("source blocks must be unique and preserve source order")

    selected_ids = set(content_blocks)
    for block in document.blocks[selected_positions[0] : selected_positions[-1] + 1]:
        if block.type in _CRITICAL_TYPES and block.block_id not in selected_ids:
            raise _fail(f"critical source block omitted: {block.block_id}")

    return tuple(selected)


def _render_block(block: ContentBlock) -> str | None:
    if block.type in _SKIP_TYPES:
        return None
    if block.type in {"paragraph", "textbox"}:
        text = _block_text(block)
        return text or None
    if block.type == "formula":
        formula = _block_text(block)
        if not formula:
            raise _fail(f"formula source is empty: {block.block_id}")
        if formula.startswith("$") and formula.endswith("$"):
            return formula
        return f"${formula}$"
    if block.type == "image":
        filename = block.metadata.get("asset_filename")
        if not isinstance(filename, str) or not filename:
            raise _fail(f"image asset is missing: {block.block_id}")
        return f'<img src="image/{filename}">'
    if block.type == "table":
        rendered = block.metadata.get("rendered")
        if not isinstance(rendered, str) or not rendered.strip():
            raise _fail(f"table rendering is missing: {block.block_id}")
        return rendered.strip()
    if block.type == "unresolved":
        reason = block.metadata.get("reason", "unresolved critical content")
        raise _fail(f"{reason}: {block.block_id}")
    return _block_text(block) or None


def render_question_markdown(document: DocumentIR, content_blocks: tuple[str, ...]) -> str:
    selected = _validate_selection(document, content_blocks)
    fragments: list[str] = []
    for block in selected:
        rendered = _render_block(block)
        if rendered:
            fragments.append(rendered)
    if not fragments:
        raise _fail("question reconstruction produced no source content")
    return "\n\n".join(fragments)
