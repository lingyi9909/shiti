from __future__ import annotations

from collections.abc import Mapping
from html import escape

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


class QuestionContentError(RuntimeError):
    def __init__(self, reason_code: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


_CRITICAL_TYPES = {"formula", "image", "table", "unresolved"}
_SKIP_TYPES = {"header", "footer", "noise_candidate"}
_CELL_CRITICAL_KINDS = {"formula", "image", "unresolved_formula", "unresolved"}


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


def _table_cells(block: ContentBlock) -> tuple[Mapping[str, object], ...] | None:
    raw_cells = block.metadata.get("cells")
    raw_assets = block.metadata.get("assets")
    if raw_cells is None:
        if isinstance(raw_assets, tuple) and raw_assets:
            raise _fail(f"table cell asset metadata cannot be reconstructed: {block.block_id}")
        return None
    if not isinstance(raw_cells, tuple):
        raise _fail(f"table cell metadata is invalid: {block.block_id}")

    cells: list[Mapping[str, object]] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping):
            raise _fail(f"table cell metadata is invalid: {block.block_id}")
        cells.append(raw_cell)
    return tuple(cells)


def _table_has_critical_cell_content(
    block: ContentBlock,
    cells: tuple[Mapping[str, object], ...],
) -> bool:
    for cell in cells:
        raw_contents = cell.get("contents")
        if not isinstance(raw_contents, tuple):
            raise _fail(f"table cell contents are invalid: {block.block_id}")
        for raw_content in raw_contents:
            if not isinstance(raw_content, Mapping):
                raise _fail(f"table cell contents are invalid: {block.block_id}")
            kind = raw_content.get("kind")
            if not isinstance(kind, str):
                raise _fail(f"table cell content kind is invalid: {block.block_id}")
            if kind in _CELL_CRITICAL_KINDS:
                return True
    return False


def _table_asset_map(block: ContentBlock) -> dict[str, str]:
    raw_assets = block.metadata.get("assets")
    if raw_assets is None:
        return {}
    if not isinstance(raw_assets, tuple):
        raise _fail(f"table asset metadata is invalid: {block.block_id}")

    assets: dict[str, str] = {}
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise _fail(f"table asset metadata is invalid: {block.block_id}")
        relationship_id = raw_asset.get("relationship_id")
        filename = raw_asset.get("asset_filename")
        if not isinstance(relationship_id, str) or not relationship_id:
            raise _fail(f"table image relationship is missing: {block.block_id}")
        if not isinstance(filename, str) or not filename:
            raise _fail(f"table image asset is missing: {block.block_id}")
        previous = assets.get(relationship_id)
        if previous is not None and previous != filename:
            raise _fail(f"table image relationship is ambiguous: {block.block_id}")
        assets[relationship_id] = filename
    return assets


def _table_cell_fragment(
    block: ContentBlock,
    content: Mapping[str, object],
    *,
    assets: dict[str, str],
    html_mode: bool,
) -> str:
    kind = content.get("kind")
    if kind == "text":
        text = content.get("text")
        if not isinstance(text, str):
            raise _fail(f"table cell text is invalid: {block.block_id}")
        return escape(text) if html_mode else text
    if kind == "formula":
        latex = content.get("latex")
        if not isinstance(latex, str) or not latex:
            raise _fail(f"table cell formula is missing: {block.block_id}")
        formula = latex if latex.startswith("$") and latex.endswith("$") else f"${latex}$"
        return escape(formula) if html_mode else formula
    if kind == "image":
        relationship_id = content.get("relationship_id")
        if not isinstance(relationship_id, str) or not relationship_id:
            raise _fail(f"table cell image relationship is missing: {block.block_id}")
        filename = assets.get(relationship_id)
        if filename is None:
            raise _fail(f"table cell image asset is missing: {block.block_id}")
        return f'<img src="image/{filename}">'
    if kind in {"unresolved_formula", "unresolved"}:
        raise _fail(f"table cell critical content is unresolved: {block.block_id}")
    raise _fail(f"table cell content kind is unsupported: {block.block_id}")


def _render_table_cell(
    block: ContentBlock,
    cell: Mapping[str, object],
    *,
    assets: dict[str, str],
    html_mode: bool,
) -> str:
    raw_contents = cell.get("contents")
    if not isinstance(raw_contents, tuple):
        raise _fail(f"table cell contents are invalid: {block.block_id}")
    fragments: list[str] = []
    for raw_content in raw_contents:
        if not isinstance(raw_content, Mapping):
            raise _fail(f"table cell contents are invalid: {block.block_id}")
        fragments.append(
            _table_cell_fragment(
                block,
                raw_content,
                assets=assets,
                html_mode=html_mode,
            )
        )
    return "".join(fragments)


def _cell_position(block: ContentBlock, cell: Mapping[str, object]) -> tuple[int, int, int, int]:
    row = cell.get("row")
    col = cell.get("col")
    colspan = cell.get("colspan")
    rowspan = cell.get("rowspan")
    values = (row, col, colspan, rowspan)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise _fail(f"table cell position is invalid: {block.block_id}")
    assert isinstance(row, int)
    assert isinstance(col, int)
    assert isinstance(colspan, int)
    assert isinstance(rowspan, int)
    if row < 0 or col < 0 or colspan < 1 or rowspan < 1:
        raise _fail(f"table cell position is invalid: {block.block_id}")
    return row, col, colspan, rowspan


def _render_simple_table_from_cells(
    block: ContentBlock,
    cells: tuple[Mapping[str, object], ...],
    assets: dict[str, str],
) -> str:
    if not cells:
        raise _fail(f"table cell metadata is empty: {block.block_id}")

    positions = [_cell_position(block, cell) for cell in cells]
    if any(colspan != 1 or rowspan != 1 for _, _, colspan, rowspan in positions):
        raise _fail(f"simple table has complex cell spans: {block.block_id}")

    row_count = max(row for row, _, _, _ in positions) + 1
    width = max(col for _, col, _, _ in positions) + 1
    grid = [["" for _ in range(width)] for _ in range(row_count)]
    occupied: set[tuple[int, int]] = set()

    for cell, (row, col, _, _) in zip(cells, positions, strict=True):
        coordinate = (row, col)
        if coordinate in occupied:
            raise _fail(f"table cell position is duplicated: {block.block_id}")
        occupied.add(coordinate)
        grid[row][col] = _render_table_cell(
            block,
            cell,
            assets=assets,
            html_mode=False,
        )

    lines = ["| " + " | ".join(grid[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in grid[1:])
    return "\n".join(lines)


def _render_complex_table_from_cells(
    block: ContentBlock,
    cells: tuple[Mapping[str, object], ...],
    assets: dict[str, str],
) -> str:
    if not cells:
        raise _fail(f"table cell metadata is empty: {block.block_id}")

    entries = [(_cell_position(block, cell), cell) for cell in cells]
    entries.sort(key=lambda item: (item[0][0], item[0][1]))
    row_count = max(position[0] for position, _ in entries) + 1
    by_row: dict[int, list[tuple[tuple[int, int, int, int], Mapping[str, object]]]] = {}
    for position, cell in entries:
        by_row.setdefault(position[0], []).append((position, cell))

    covered: set[tuple[int, int]] = set()
    lines = ["<table>"]
    for row_index in range(row_count):
        lines.append("  <tr>")
        for (row, col, colspan, rowspan), cell in by_row.get(row_index, []):
            if (row, col) in covered:
                continue
            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    coordinate = (row + row_offset, col + col_offset)
                    if coordinate != (row, col):
                        covered.add(coordinate)
            attrs: list[str] = []
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')
            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            cell_text = _render_table_cell(
                block,
                cell,
                assets=assets,
                html_mode=True,
            )
            lines.append(f"    <td{attr_text}>{cell_text}</td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _render_table(block: ContentBlock) -> str:
    rendered = block.metadata.get("rendered")
    if not isinstance(rendered, str) or not rendered.strip():
        raise _fail(f"table rendering is missing: {block.block_id}")

    cells = _table_cells(block)
    if cells is None or not _table_has_critical_cell_content(block, cells):
        return rendered.strip()

    assets = _table_asset_map(block)
    render_format = block.metadata.get("render_format")
    if render_format == "markdown":
        return _render_simple_table_from_cells(block, cells, assets)
    if render_format == "html":
        return _render_complex_table_from_cells(block, cells, assets)
    raise _fail(f"table render format is unsupported: {block.block_id}")


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
        return _render_table(block)
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
