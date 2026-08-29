from __future__ import annotations

from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from question_builder.domain.document import BlockType, ContentBlock, DocumentIR
from question_builder.parser.docx.assets import materialize_asset
from question_builder.parser.docx.numbering import NumberingResolver
from question_builder.parser.docx.package import DocxPackage

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
W = f"{{{W_NS}}}"
R = f"{{{R_NS}}}"
A = f"{{{A_NS}}}"


def parse_docx(path: Path, asset_dir: Path) -> DocumentIR:
    package = DocxPackage.open(path)
    numbering = _load_numbering(package)
    relationships = _load_relationships(package)
    root = ET.fromstring(package.read("word/document.xml"))
    body = root.find(f"{W}body")
    blocks: list[ContentBlock] = []
    if body is not None:
        for body_index, child in enumerate(body):
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
                text = "".join(node.text or "" for node in child.iter(f"{W}t"))
                blocks.append(
                    _block(
                        blocks,
                        "table",
                        text,
                        f"word/document.xml#/w:document/w:body/*[{body_index + 1}]",
                    )
                )
            elif child.tag != f"{W}sectPr":
                blocks.append(
                    _block(
                        blocks,
                        "unresolved",
                        "",
                        f"word/document.xml#/w:document/w:body/*[{body_index + 1}]",
                        metadata={"xml_tag": child.tag},
                    )
                )

    return DocumentIR(
        document_id=f"doc_{package.source_sha256[:16]}",
        source_file=path.name,
        source_sha256=package.source_sha256,
        blocks=tuple(blocks),
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

    for run_index, run in enumerate(paragraph.findall(f"{W}r")):
        for child_index, child in enumerate(run):
            if child.tag == f"{W}t":
                text_parts.append(child.text or "")
            elif child.tag == f"{W}tab":
                text_parts.append("\t")
            elif child.tag in {f"{W}br", f"{W}cr"}:
                text_parts.append("\n")
            elif child.tag == f"{W}drawing":
                flush_text()
                for blip in child.iter(f"{A}blip"):
                    rel_id = blip.attrib.get(f"{R}embed")
                    if not rel_id or rel_id not in relationships:
                        blocks.append(
                            _block(
                                blocks,
                                "unresolved",
                                "",
                                f"{source_path}/w:r[{run_index + 1}]/*[{child_index + 1}]",
                                relationship_id=rel_id,
                                metadata={"reason": "missing_image_relationship"},
                            )
                        )
                        continue
                    member = _relationship_member(relationships[rel_id])
                    asset = materialize_asset(package, member, asset_dir)
                    blocks.append(
                        _block(
                            blocks,
                            "image",
                            "",
                            f"{source_path}/w:r[{run_index + 1}]/*[{child_index + 1}]",
                            relationship_id=rel_id,
                            metadata={
                                "asset_id": asset.asset_id,
                                "asset_sha256": asset.sha256,
                                "asset_filename": asset.filename,
                                "source_member": asset.source_member,
                            },
                        )
                    )
            elif child.tag == f"{W}rPr":
                continue
            else:
                flush_text()
                blocks.append(
                    _block(
                        blocks,
                        "unresolved",
                        child.text or "",
                        f"{source_path}/w:r[{run_index + 1}]/*[{child_index + 1}]",
                        metadata={"xml_tag": child.tag},
                    )
                )
    flush_text()
    if not any(block.source_xml_path == source_path for block in blocks):
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
    ilvl = int(ilvl_node.attrib.get(f"{W}val", "0")) if ilvl_node is not None else 0
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
