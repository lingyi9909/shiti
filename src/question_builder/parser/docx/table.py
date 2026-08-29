from __future__ import annotations

from dataclasses import dataclass
from html import escape
from xml.etree import ElementTree as ET

from question_builder.parser.docx.formula import FormulaConversionError, omml_to_latex

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
W = f"{{{W_NS}}}"
M = f"{{{M_NS}}}"
A = f"{{{A_NS}}}"
R = f"{{{R_NS}}}"
CELL_WRAPPER_TAGS = {
    f"{W}hyperlink",
    f"{W}smartTag",
    f"{W}customXml",
    f"{W}ins",
    f"{W}del",
    f"{W}moveFrom",
    f"{W}moveTo",
}


@dataclass(frozen=True, slots=True)
class ParsedCell:
    row: int
    col: int
    text: str
    colspan: int = 1
    rowspan: int = 1
    content_kinds: tuple[str, ...] = ()
    relationship_ids: tuple[str, ...] = ()
    contents: tuple[dict[str, str], ...] = ()
    vertical_continuation: bool = False


@dataclass(frozen=True, slots=True)
class ParsedTable:
    rows: tuple[tuple[str, ...], ...]
    cells: tuple[ParsedCell, ...]
    merges: tuple[dict[str, int], ...]
    is_complex: bool
    rendered: str


def parse_table(table: ET.Element) -> ParsedTable:
    rows: list[tuple[str, ...]] = []
    cells: list[ParsedCell] = []
    active_vertical: dict[int, int] = {}
    merge_rows: dict[int, int] = {}

    for row_index, row in enumerate(table.findall(f"{W}tr")):
        row_values: list[str] = []
        col = 0
        for cell_element in row.findall(f"{W}tc"):
            props = cell_element.find(f"{W}tcPr")
            colspan = 1
            vmerge: str | None = None
            if props is not None:
                grid_span = props.find(f"{W}gridSpan")
                if grid_span is not None:
                    colspan = int(grid_span.attrib.get(f"{W}val", "1"))
                vmerge_node = props.find(f"{W}vMerge")
                if vmerge_node is not None:
                    vmerge = vmerge_node.attrib.get(f"{W}val", "continue")

            text, kinds, rel_ids, contents = _cell_content(cell_element)
            continuation = vmerge == "continue"
            parsed = ParsedCell(
                row=row_index,
                col=col,
                text=text,
                colspan=colspan,
                content_kinds=tuple(kinds),
                relationship_ids=tuple(rel_ids),
                contents=tuple(contents),
                vertical_continuation=continuation,
            )
            cell_index = len(cells)
            cells.append(parsed)
            row_values.append(text)

            if vmerge == "restart":
                active_vertical[col] = cell_index
                merge_rows[cell_index] = 1
            elif continuation and col in active_vertical:
                merge_rows[active_vertical[col]] += 1
            else:
                active_vertical.pop(col, None)
            col += colspan
        rows.append(tuple(row_values))

    adjusted: list[ParsedCell] = []
    merges: list[dict[str, int]] = []
    for index, parsed_cell in enumerate(cells):
        rowspan = merge_rows.get(index, 1)
        updated = ParsedCell(
            row=parsed_cell.row,
            col=parsed_cell.col,
            text=parsed_cell.text,
            colspan=parsed_cell.colspan,
            rowspan=rowspan,
            content_kinds=parsed_cell.content_kinds,
            relationship_ids=parsed_cell.relationship_ids,
            contents=parsed_cell.contents,
            vertical_continuation=parsed_cell.vertical_continuation,
        )
        adjusted.append(updated)
        if parsed_cell.colspan > 1 or rowspan > 1:
            merges.append(
                {
                    "row": parsed_cell.row,
                    "col": parsed_cell.col,
                    "colspan": parsed_cell.colspan,
                    "rowspan": rowspan,
                }
            )

    is_complex = bool(merges) or any(
        cell.vertical_continuation for cell in adjusted
    )
    rendered = (
        _render_html(adjusted, len(rows))
        if is_complex
        else _render_markdown(rows)
    )
    return ParsedTable(
        rows=tuple(rows),
        cells=tuple(adjusted),
        merges=tuple(merges),
        is_complex=is_complex,
        rendered=rendered,
    )


def _cell_content(
    cell: ET.Element,
) -> tuple[str, list[str], list[str], list[dict[str, str]]]:
    parts: list[str] = []
    kinds: list[str] = []
    rel_ids: list[str] = []
    contents: list[dict[str, str]] = []

    def append_text(text: str) -> None:
        if not text:
            return
        parts.append(text)
        kinds.append("text")
        contents.append({"kind": "text", "text": text})

    def append_formula(element: ET.Element) -> None:
        source_omml = ET.tostring(element, encoding="unicode")
        try:
            latex = omml_to_latex(element)
        except FormulaConversionError as exc:
            kinds.append("formula")
            contents.append(
                {
                    "kind": "unresolved_formula",
                    "source_omml": source_omml,
                    "error": str(exc),
                }
            )
            return
        parts.append(latex)
        kinds.append("formula")
        contents.append(
            {
                "kind": "formula",
                "latex": latex,
                "source_omml": source_omml,
            }
        )

    def append_unresolved(element: ET.Element) -> None:
        text = "".join(node.text or "" for node in element.iter(f"{W}t"))
        kinds.append("unresolved")
        contents.append(
            {
                "kind": "unresolved",
                "text": text,
                "xml_tag": element.tag,
            }
        )

    def process_drawing(drawing: ET.Element) -> None:
        found_image = False
        for node in drawing.iter():
            if node.tag != f"{A}blip":
                continue
            found_image = True
            rel_id = node.attrib.get(f"{R}embed")
            kinds.append("image")
            image_content = {"kind": "image"}
            if rel_id:
                rel_ids.append(rel_id)
                image_content["relationship_id"] = rel_id
            contents.append(image_content)
        if not found_image:
            append_unresolved(drawing)

    def process_run(run: ET.Element) -> None:
        for child in run:
            if child.tag == f"{W}t":
                append_text(child.text or "")
            elif child.tag == f"{W}tab":
                append_text("\t")
            elif child.tag in {f"{W}br", f"{W}cr"}:
                append_text("\n")
            elif child.tag == f"{W}drawing":
                process_drawing(child)
            elif child.tag in {f"{M}oMath", f"{M}oMathPara"}:
                append_formula(child)
            elif child.tag == f"{W}rPr":
                continue
            else:
                append_unresolved(child)

    def process_wrapper(wrapper: ET.Element) -> None:
        for child in wrapper:
            if child.tag == f"{W}r":
                process_run(child)
            elif child.tag in {f"{M}oMath", f"{M}oMathPara"}:
                append_formula(child)
            elif child.tag in CELL_WRAPPER_TAGS:
                process_wrapper(child)
            else:
                append_unresolved(child)

    for child in cell:
        if child.tag == f"{W}tcPr":
            continue
        if child.tag != f"{W}p":
            append_unresolved(child)
            continue
        for paragraph_child in child:
            if paragraph_child.tag == f"{W}pPr":
                continue
            if paragraph_child.tag == f"{W}r":
                process_run(paragraph_child)
            elif paragraph_child.tag in {f"{M}oMath", f"{M}oMathPara"}:
                append_formula(paragraph_child)
            elif paragraph_child.tag in CELL_WRAPPER_TAGS:
                process_wrapper(paragraph_child)
            else:
                append_unresolved(paragraph_child)

    return "".join(parts), kinds, rel_ids, contents


def _render_markdown(rows: list[tuple[str, ...]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [
        tuple(list(row) + [""] * (width - len(row)))
        for row in rows
    ]
    lines = ["| " + " | ".join(normalized[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _render_html(cells: list[ParsedCell], row_count: int) -> str:
    by_row: dict[int, list[ParsedCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row, []).append(cell)
    lines = ["<table>"]
    for row_index in range(row_count):
        lines.append("  <tr>")
        for cell in by_row.get(row_index, []):
            if cell.vertical_continuation:
                continue
            attrs: list[str] = []
            if cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            if cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            attr_text = (" " + " ".join(attrs)) if attrs else ""
            lines.append(f"    <td{attr_text}>{escape(cell.text)}</td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)
