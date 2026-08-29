from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from question_builder.parser.docx.body import parse_docx
from question_builder.parser.docx.table import parse_table

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_docx(
    path: Path,
    document_xml: str,
    members: dict[str, bytes] | None = None,
) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml.encode())
        for name, content in (members or {}).items():
            archive.writestr(name, content)


def test_drawingml_textbox_image_order_duplicate_occurrences_and_unresolved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "drawing-textbox.docx"
    document = f"""<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='{W_NS}' xmlns:r='{R_NS}' xmlns:a='{A_NS}'>
  <w:body>
    <w:p><w:r><w:drawing>
      <w:txbxContent><w:p><w:r><w:t>same</w:t></w:r></w:p></w:txbxContent>
      <a:blip r:embed='rIdImage'/>
    </w:drawing></w:r></w:p>
    <w:p><w:r><w:drawing>
      <w:txbxContent><w:p><w:r><w:t>same</w:t></w:r></w:p></w:txbxContent>
    </w:drawing></w:r></w:p>
    <w:p><w:r><w:drawing>
      <a:graphic><a:graphicData><a:unknown/></a:graphicData></a:graphic>
    </w:drawing></w:r></w:p>
  </w:body>
</w:document>"""
    rels = f"""<Relationships xmlns='{PKG_REL_NS}'>
  <Relationship Id='rIdImage' Type='image' Target='media/image1.png'/>
</Relationships>""".encode()
    _write_docx(
        source,
        document,
        {
            "word/_rels/document.xml.rels": rels,
            "word/media/image1.png": b"png",
        },
    )

    parsed = parse_docx(source, tmp_path / "assets")

    relevant = [
        (block.type, block.raw_text)
        for block in parsed.blocks
        if block.type in {"textbox", "image", "unresolved"}
    ]
    assert relevant[:3] == [
        ("textbox", "same"),
        ("image", ""),
        ("textbox", "same"),
    ]
    assert sum(
        block.type == "textbox" and block.raw_text == "same"
        for block in parsed.blocks
    ) == 2
    unresolved = [block for block in parsed.blocks if block.type == "unresolved"]
    assert any(
        block.metadata.get("reason") == "unsupported_drawing_content"
        for block in unresolved
    )


def test_omathpara_routes_through_formula_chain_in_source_order(tmp_path: Path) -> None:
    source = tmp_path / "omathpara.docx"
    document = f"""<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='{W_NS}' xmlns:m='{M_NS}'>
  <w:body><w:p>
    <w:r><w:t>before</w:t></w:r>
    <m:oMathPara><m:oMath><m:r><m:t>x+1</m:t></m:r></m:oMath></m:oMathPara>
    <w:r><w:t>after</w:t></w:r>
  </w:p></w:body>
</w:document>"""
    _write_docx(source, document)

    parsed = parse_docx(source, tmp_path / "assets")

    assert [(block.type, block.raw_text) for block in parsed.blocks] == [
        ("paragraph", "before"),
        ("formula", "x+1"),
        ("paragraph", "after"),
    ]
    assert "oMathPara" in parsed.blocks[1].metadata["source_omml"]


def test_table_cell_recurses_wrappers_preserves_order_and_unresolved() -> None:
    table = ET.fromstring(
        f"""<w:tbl xmlns:w='{W_NS}' xmlns:r='{R_NS}' xmlns:a='{A_NS}' xmlns:m='{M_NS}'>
  <w:tr><w:tc><w:p>
    <w:r><w:t>before</w:t></w:r>
    <w:hyperlink r:id='rIdLink'><w:r><w:t>linked</w:t></w:r></w:hyperlink>
    <m:oMath><m:r><m:t>x</m:t></m:r></m:oMath>
    <w:r><w:drawing><a:blip r:embed='rIdImage'/></w:drawing></w:r>
    <w:custom><w:r><w:t>unsupported</w:t></w:r></w:custom>
    <w:r><w:t>after</w:t></w:r>
  </w:p></w:tc></w:tr>
</w:tbl>"""
    )

    parsed = parse_table(table)
    cell = parsed.cells[0]

    assert cell.content_kinds == (
        "text",
        "text",
        "formula",
        "image",
        "unresolved",
        "text",
    )
    assert [item["kind"] for item in cell.contents] == [
        "text",
        "text",
        "formula",
        "image",
        "unresolved",
        "text",
    ]
    assert cell.contents[1]["text"] == "linked"
    assert cell.contents[4]["text"] == "unsupported"


def test_table_images_missing_relationship_target_or_member_are_explicit_unresolved(
    tmp_path: Path,
) -> None:
    source = tmp_path / "table-images.docx"
    document = f"""<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='{W_NS}' xmlns:r='{R_NS}' xmlns:a='{A_NS}'>
  <w:body><w:tbl><w:tr><w:tc><w:p>
    <w:r><w:drawing><a:blip r:embed='rIdMissingRel'/></w:drawing></w:r>
    <w:r><w:drawing><a:blip r:embed='rIdMissingTarget'/></w:drawing></w:r>
    <w:r><w:drawing><a:blip r:embed='rIdMissingMember'/></w:drawing></w:r>
  </w:p></w:tc></w:tr></w:tbl></w:body>
</w:document>"""
    rels = f"""<Relationships xmlns='{PKG_REL_NS}'>
  <Relationship Id='rIdMissingTarget' Type='image' Target=''/>
  <Relationship Id='rIdMissingMember' Type='image' Target='media/not-there.png'/>
</Relationships>""".encode()
    _write_docx(source, document, {"word/_rels/document.xml.rels": rels})

    parsed = parse_docx(source, tmp_path / "assets")

    table_unresolved = [
        block
        for block in parsed.blocks
        if block.type == "unresolved"
        and block.metadata.get("container") == "table_cell"
    ]
    reasons = {block.metadata.get("reason") for block in table_unresolved}
    assert reasons == {
        "missing_image_relationship",
        "missing_image_relationship_target",
        "missing_image_package_member",
    }
