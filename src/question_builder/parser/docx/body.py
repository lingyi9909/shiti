from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal
from xml.etree import ElementTree as ET

from question_builder.domain.document import BlockType, ContentBlock, DocumentIR
from question_builder.parser.docx.assets import materialize_asset
from question_builder.parser.docx.formula import FormulaConversionError, omml_to_latex
from question_builder.parser.docx.numbering import NumberingResolver
from question_builder.parser.docx.package import DocxPackage
from question_builder.parser.docx.table import parse_table
from question_builder.parser.docx.textbox import find_textboxes

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
OFFICE_NS = "urn:schemas-microsoft-com:office:office"
V_NS = "urn:schemas-microsoft-com:vml"
W = f"{{{W_NS}}}"
R = f"{{{R_NS}}}"
A = f"{{{A_NS}}}"
M = f"{{{M_NS}}}"
OFFICE = f"{{{OFFICE_NS}}}"
V = f"{{{V_NS}}}"
RUN_WRAPPER_TAGS = {
    f"{W}hyperlink",
    f"{W}smartTag",
    f"{W}customXml",
    f"{W}ins",
    f"{W}del",
    f"{W}moveFrom",
    f"{W}moveTo",
}
MATH_TAGS = {f"{M}oMath", f"{M}oMathPara"}


def parse_docx(path: Path, asset_dir: Path) -> DocumentIR:
    package = DocxPackage.open(path)
    numbering = _load_numbering(package)
    relationships = _load_relationships(package)
    root = ET.fromstring(package.read("word/document.xml"))
    body = root.find(f"{W}body")
    blocks: list[ContentBlock] = []
    if body is not None:
        for body_index, child in enumerate(body):
            source_path = f"word/document.xml#/w:document/w:body/*[{body_index + 1}]"
            if child.tag == f"{W}p":
                _append_paragraph_blocks(
                    child,
                    body_index,
                    blocks,
                    package,
                    asset_dir,
                    numbering,
                    relationships,
                )
            elif child.tag == f"{W}tbl":
                _append_table_block(
                    child,
                    source_path,
                    blocks,
                    package,
                    asset_dir,
                    relationships,
                )
            elif child.tag != f"{W}sectPr":
                blocks.append(
                    _block(
                        blocks,
                        "unresolved",
                        "".join(node.text or "" for node in child.iter(f"{W}t")),
                        source_path,
                        metadata={
                            "xml_tag": child.tag,
                            "reason": "unsupported_body_child",
                        },
                    )
                )

    _append_orphan_embedding_blocks(blocks, package)
    _append_header_footer_blocks(root, blocks, package, relationships)
    return DocumentIR(
        document_id=f"doc_{package.source_sha256[:16]}",
        source_file=path.name,
        source_sha256=package.source_sha256,
        blocks=tuple(blocks),
    )


def _append_table_block(
    table: ET.Element,
    source_path: str,
    blocks: list[ContentBlock],
    package: DocxPackage,
    asset_dir: Path,
    relationships: dict[str, str],
) -> None:
    parsed = parse_table(table)
    raw_text = "".join(node.text or "" for node in table.iter(f"{W}t"))
    assets: list[dict[str, str]] = []
    unresolved_items: list[dict[str, object]] = []

    for cell in parsed.cells:
        for content in cell.contents:
            kind = content.get("kind")
            if kind == "image":
                relationship_id = content.get("relationship_id")
                if not relationship_id or relationship_id not in relationships:
                    unresolved_items.append(
                        {
                            "reason": "missing_image_relationship",
                            "relationship_id": relationship_id,
                            "raw_text": "",
                        }
                    )
                    continue
                target = relationships[relationship_id]
                if not target:
                    unresolved_items.append(
                        {
                            "reason": "missing_image_relationship_target",
                            "relationship_id": relationship_id,
                            "raw_text": "",
                        }
                    )
                    continue
                member = _relationship_member(target)
                if member not in package.members:
                    unresolved_items.append(
                        {
                            "reason": "missing_image_package_member",
                            "relationship_id": relationship_id,
                            "target": target,
                            "member": member,
                            "raw_text": "",
                        }
                    )
                    continue
                asset = materialize_asset(package, member, asset_dir)
                assets.append(
                    {
                        "relationship_id": relationship_id,
                        "asset_id": asset.asset_id,
                        "asset_sha256": asset.sha256,
                        "asset_filename": asset.filename,
                        "source_member": asset.source_member,
                    }
                )
            elif kind == "unresolved_formula":
                unresolved_items.append(
                    {
                        "reason": "FORMULA_UNRESOLVED",
                        "source_omml": content.get("source_omml", ""),
                        "error": content.get("error", ""),
                        "raw_text": content.get("source_omml", ""),
                    }
                )
            elif kind == "unresolved":
                unresolved_items.append(
                    {
                        "reason": "unsupported_table_cell_child",
                        "xml_tag": content.get("xml_tag", ""),
                        "raw_text": content.get("text", ""),
                    }
                )

    blocks.append(
        _block(
            blocks,
            "table",
            raw_text,
            source_path,
            metadata={
                "render_format": "html" if parsed.is_complex else "markdown",
                "rendered": parsed.rendered,
                "rows": parsed.rows,
                "merges": parsed.merges,
                "assets": tuple(assets),
                "cells": tuple(
                    {
                        "row": cell.row,
                        "col": cell.col,
                        "colspan": cell.colspan,
                        "rowspan": cell.rowspan,
                        "content_kinds": cell.content_kinds,
                        "relationship_ids": cell.relationship_ids,
                        "contents": cell.contents,
                    }
                    for cell in parsed.cells
                ),
            },
        )
    )
    for unresolved in unresolved_items:
        relationship_id = unresolved.get("relationship_id")
        blocks.append(
            _block(
                blocks,
                "unresolved",
                str(unresolved.get("raw_text", "")),
                source_path,
                relationship_id=(
                    relationship_id if isinstance(relationship_id, str) else None
                ),
                metadata={
                    **unresolved,
                    "container": "table_cell",
                },
            )
        )


def _append_paragraph_blocks(
    paragraph: ET.Element,
    body_index: int,
    blocks: list[ContentBlock],
    package: DocxPackage,
    asset_dir: Path,
    numbering: NumberingResolver,
    relationships: dict[str, str],
) -> None:
    source_path = f"word/document.xml#/w:document/w:body/*[{body_index + 1}]"
    starting_block_count = len(blocks)
    numbering_metadata = _paragraph_numbering(paragraph, numbering)
    style_node = paragraph.find(f"{W}pPr/{W}pStyle")
    style_id = style_node.attrib.get(f"{W}val") if style_node is not None else None
    text_parts: list[str] = []

    def flush_text() -> None:
        if not text_parts:
            return
        text = "".join(text_parts)
        text_parts.clear()
        first_paragraph_block = not any(
            block.source_xml_path == source_path for block in blocks
        )
        blocks.append(
            _block(
                blocks,
                "paragraph",
                text,
                source_path,
                style_id=style_id,
                numbering=numbering_metadata if first_paragraph_block else None,
            )
        )

    def append_unresolved(element: ET.Element, element_path: str, reason: str) -> None:
        flush_text()
        raw_text = "".join(node.text or "" for node in element.iter(f"{W}t"))
        blocks.append(
            _block(
                blocks,
                "unresolved",
                raw_text,
                element_path,
                metadata={"xml_tag": element.tag, "reason": reason},
            )
        )

    def append_formula(element: ET.Element, element_path: str) -> None:
        flush_text()
        source_omml = ET.tostring(element, encoding="unicode")
        try:
            latex = omml_to_latex(element)
        except FormulaConversionError as exc:
            blocks.append(
                _block(
                    blocks,
                    "unresolved",
                    source_omml,
                    element_path,
                    metadata={
                        "reason": "FORMULA_UNRESOLVED",
                        "source_omml": source_omml,
                        "error": str(exc),
                    },
                )
            )
            return
        blocks.append(
            _block(
                blocks,
                "formula",
                latex,
                element_path,
                metadata={"latex": latex, "source_omml": source_omml},
            )
        )

    def append_object(element: ET.Element, element_path: str) -> None:
        flush_text()
        ole = next(iter(element.iter(f"{OFFICE}OLEObject")), None)
        rel_id = ole.attrib.get(f"{R}id") if ole is not None else None
        target = relationships.get(rel_id, "") if rel_id else ""
        embedding_path = _relationship_member(target) if target else None
        preview = next(iter(element.iter(f"{V}imagedata")), None)
        preview_rel = preview.attrib.get(f"{R}id") if preview is not None else None
        blocks.append(
            _block(
                blocks,
                "unresolved",
                "",
                element_path,
                relationship_id=rel_id,
                metadata={
                    "reason": "FORMULA_UNRESOLVED",
                    "embedding_path": embedding_path,
                    "preview_relationship_id": preview_rel,
                },
            )
        )

    def process_run(run: ET.Element, run_path: str) -> None:
        for child_index, child in enumerate(run):
            child_path = f"{run_path}/*[{child_index + 1}]"
            if child.tag == f"{W}t":
                text_parts.append(child.text or "")
            elif child.tag == f"{W}tab":
                text_parts.append("\t")
            elif child.tag in {f"{W}br", f"{W}cr"}:
                text_parts.append("\n")
            elif child.tag == f"{W}drawing":
                _append_drawing(
                    child,
                    child_path,
                    blocks,
                    package,
                    asset_dir,
                    relationships,
                    flush_text,
                )
            elif child.tag in MATH_TAGS:
                append_formula(child, child_path)
            elif child.tag == f"{W}pict":
                textboxes = find_textboxes(child)
                if textboxes:
                    flush_text()
                    for box in textboxes:
                        blocks.append(
                            _block(
                                blocks,
                                "textbox",
                                box.text,
                                child_path,
                                metadata={"anchor_hint": box.anchor_hint},
                            )
                        )
                else:
                    append_unresolved(child, child_path, "unsupported_pict")
            elif child.tag == f"{W}object":
                append_object(child, child_path)
            elif child.tag == f"{W}rPr":
                continue
            else:
                append_unresolved(child, child_path, "unsupported_run_child")

    def process_wrapper(wrapper: ET.Element, wrapper_path: str) -> None:
        for child_index, child in enumerate(wrapper):
            child_path = f"{wrapper_path}/*[{child_index + 1}]"
            if child.tag == f"{W}r":
                process_run(child, child_path)
            elif child.tag in MATH_TAGS:
                append_formula(child, child_path)
            elif child.tag in RUN_WRAPPER_TAGS:
                process_wrapper(child, child_path)
            else:
                append_unresolved(
                    child,
                    child_path,
                    "unsupported_paragraph_wrapper_child",
                )

    for child_index, child in enumerate(paragraph):
        child_path = f"{source_path}/*[{child_index + 1}]"
        if child.tag == f"{W}pPr":
            continue
        if child.tag == f"{W}r":
            process_run(child, child_path)
        elif child.tag in MATH_TAGS:
            append_formula(child, child_path)
        elif child.tag in RUN_WRAPPER_TAGS:
            process_wrapper(child, child_path)
        else:
            append_unresolved(child, child_path, "unsupported_paragraph_child")

    flush_text()
    if len(blocks) == starting_block_count:
        blocks.append(
            _block(
                blocks,
                "paragraph",
                "",
                source_path,
                style_id=style_id,
                numbering=numbering_metadata,
            )
        )


def _append_drawing(
    drawing: ET.Element,
    source_path: str,
    blocks: list[ContentBlock],
    package: DocxPackage,
    asset_dir: Path,
    relationships: dict[str, str],
    flush_text: Callable[[], None],
) -> None:
    flush_text()
    emitted = 0

    def append_image(blip: ET.Element, element_path: str) -> None:
        nonlocal emitted
        rel_id = blip.attrib.get(f"{R}embed")
        if not rel_id or rel_id not in relationships:
            blocks.append(
                _block(
                    blocks,
                    "unresolved",
                    "",
                    element_path,
                    relationship_id=rel_id,
                    metadata={"reason": "missing_image_relationship"},
                )
            )
            emitted += 1
            return
        target = relationships[rel_id]
        if not target:
            blocks.append(
                _block(
                    blocks,
                    "unresolved",
                    "",
                    element_path,
                    relationship_id=rel_id,
                    metadata={"reason": "missing_image_relationship_target"},
                )
            )
            emitted += 1
            return
        member = _relationship_member(target)
        if member not in package.members:
            blocks.append(
                _block(
                    blocks,
                    "unresolved",
                    "",
                    element_path,
                    relationship_id=rel_id,
                    metadata={
                        "reason": "missing_image_package_member",
                        "source_member": member,
                    },
                )
            )
            emitted += 1
            return
        asset = materialize_asset(package, member, asset_dir)
        blocks.append(
            _block(
                blocks,
                "image",
                "",
                element_path,
                relationship_id=rel_id,
                metadata={
                    "asset_id": asset.asset_id,
                    "asset_sha256": asset.sha256,
                    "asset_filename": asset.filename,
                    "source_member": asset.source_member,
                },
            )
        )
        emitted += 1

    def append_textbox(content: ET.Element, element_path: str) -> None:
        nonlocal emitted
        text = "".join(node.text or "" for node in content.iter(f"{W}t"))
        blocks.append(
            _block(
                blocks,
                "textbox",
                text,
                element_path,
                metadata={"anchor_hint": "drawing"},
            )
        )
        emitted += 1

    def walk(element: ET.Element, element_path: str) -> None:
        nonlocal emitted
        for child_index, child in enumerate(element):
            child_path = f"{element_path}/*[{child_index + 1}]"
            if child.tag == f"{W}txbxContent":
                append_textbox(child, child_path)
                continue
            if child.tag == f"{A}blip":
                append_image(child, child_path)
                continue
            before = emitted
            walk(child, child_path)
            if child.tag == f"{A}graphicData" and emitted == before:
                raw_text = "".join(
                    node.text or "" for node in child.iter(f"{W}t")
                )
                blocks.append(
                    _block(
                        blocks,
                        "unresolved",
                        raw_text,
                        child_path,
                        metadata={
                            "reason": "unsupported_drawing_content",
                            "xml_tag": child.tag,
                        },
                    )
                )
                emitted += 1

    walk(drawing, source_path)
    if emitted == 0 and len(drawing):
        blocks.append(
            _block(
                blocks,
                "unresolved",
                "".join(node.text or "" for node in drawing.iter(f"{W}t")),
                source_path,
                metadata={
                    "reason": "unsupported_drawing_content",
                    "xml_tag": drawing.tag,
                },
            )
        )


def _append_orphan_embedding_blocks(
    blocks: list[ContentBlock],
    package: DocxPackage,
) -> None:
    represented = {
        block.metadata.get("embedding_path")
        for block in blocks
        if block.metadata.get("embedding_path")
    }
    for member in sorted(package.members):
        if not member.startswith("word/embeddings/") or member in represented:
            continue
        blocks.append(
            _block(
                blocks,
                "unresolved",
                "",
                member,
                metadata={
                    "reason": "FORMULA_UNRESOLVED",
                    "embedding_path": member,
                    "preview_relationship_id": None,
                    "orphan_embedding": True,
                },
            )
        )


def _append_header_footer_blocks(
    document_root: ET.Element,
    blocks: list[ContentBlock],
    package: DocxPackage,
    relationships: dict[str, str],
) -> None:
    references: list[tuple[Literal["header", "footer"], ET.Element]] = []
    references.extend(
        ("header", node)
        for node in document_root.iter(f"{W}headerReference")
    )
    references.extend(
        ("footer", node)
        for node in document_root.iter(f"{W}footerReference")
    )
    for block_type, reference in references:
        rel_id = reference.attrib.get(f"{R}id")
        target = relationships.get(rel_id or "")
        if not target:
            continue
        member = _relationship_member(target)
        if member not in package.members:
            continue
        root = ET.fromstring(package.read(member))
        paragraphs = [
            "".join(node.text or "" for node in paragraph.iter(f"{W}t"))
            for paragraph in root.iter(f"{W}p")
        ]
        paragraphs = [text for text in paragraphs if text]
        counts = Counter(paragraphs)
        for index, text in enumerate(paragraphs):
            path = f"{member}#/*/w:p[{index + 1}]"
            blocks.append(
                _block(
                    blocks,
                    block_type,
                    text,
                    path,
                    relationship_id=rel_id,
                    metadata={"part": member, "metadata_only": True},
                )
            )
            if counts[text] > 1:
                blocks.append(
                    _block(
                        blocks,
                        "noise_candidate",
                        text,
                        path,
                        relationship_id=rel_id,
                        metadata={
                            "structural_evidence": f"repeated_{block_type}",
                            "repeat_count": counts[text],
                            "source_part": member,
                        },
                    )
                )


def _block(
    blocks: list[ContentBlock],
    block_type: BlockType,
    raw_text: str,
    source_xml_path: str,
    *,
    style_id: str | None = None,
    numbering: dict[str, object] | None = None,
    relationship_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ContentBlock:
    order = len(blocks)
    return ContentBlock(
        block_id=f"b{order + 1:06d}",
        order=order,
        type=block_type,
        raw_text=raw_text,
        normalized_text=raw_text,
        style_id=style_id,
        numbering=numbering,
        source_type="docx_xml",
        source_xml_path=source_xml_path,
        relationship_id=relationship_id,
        metadata=metadata or {},
    )


def _load_numbering(package: DocxPackage) -> NumberingResolver:
    if "word/numbering.xml" not in package.members:
        return NumberingResolver.empty()
    return NumberingResolver.from_xml(package.read("word/numbering.xml"))


def _paragraph_numbering(
    paragraph: ET.Element,
    resolver: NumberingResolver,
) -> dict[str, object] | None:
    num_id_node = paragraph.find(f"{W}pPr/{W}numPr/{W}numId")
    if num_id_node is None:
        return None
    ilvl_node = paragraph.find(f"{W}pPr/{W}numPr/{W}ilvl")
    num_id = int(num_id_node.attrib[f"{W}val"])
    ilvl = (
        int(ilvl_node.attrib.get(f"{W}val", "0"))
        if ilvl_node is not None
        else 0
    )
    try:
        return resolver.next_label(num_id, ilvl).as_metadata()
    except KeyError:
        return {"num_id": num_id, "ilvl": ilvl, "unresolved": True}


def _load_relationships(package: DocxPackage) -> dict[str, str]:
    member = "word/_rels/document.xml.rels"
    if member not in package.members:
        return {}
    root = ET.fromstring(package.read(member))
    return {
        node.attrib["Id"]: node.attrib["Target"]
        for node in root
        if "Id" in node.attrib and "Target" in node.attrib
    }


def _relationship_member(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath("word") / PurePosixPath(target))
