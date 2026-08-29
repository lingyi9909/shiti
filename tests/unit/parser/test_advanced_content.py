from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from question_builder.parser.docx.body import parse_docx

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
V_NS = "urn:schemas-microsoft-com:vml"
O_NS = "urn:schemas-microsoft-com:office:office"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _write_package(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_textbox_header_footer_noise_and_ole_are_explicit(tmp_path: Path) -> None:
    source = tmp_path / "advanced.docx"
    document_xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='{W_NS}' xmlns:v='{V_NS}' xmlns:o='{O_NS}' xmlns:r='{R_NS}' xmlns:m='{M_NS}'>
  <w:body>
    <w:p><w:r><w:t>Body question</w:t></w:r></w:p>
    <w:p><w:r><w:pict><v:shape><v:textbox><w:txbxContent><w:p><w:r><w:t>Textbox text</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p>
    <w:p><w:r><w:object><o:OLEObject r:id='rIdOle'/></w:object></w:r></w:p>
    <w:sectPr><w:headerReference w:type='default' r:id='rIdHeader'/><w:footerReference w:type='default' r:id='rIdFooter'/></w:sectPr>
  </w:body>
</w:document>""".encode()
    rels = f"""<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns='{PKG_REL_NS}'>
  <Relationship Id='rIdHeader' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/header' Target='header1.xml'/>
  <Relationship Id='rIdFooter' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer' Target='footer1.xml'/>
  <Relationship Id='rIdOle' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject' Target='embeddings/oleObject1.bin'/>
</Relationships>""".encode()
    header = f"<w:hdr xmlns:w='{W_NS}'><w:p><w:r><w:t>2026 Math Exam</w:t></w:r></w:p></w:hdr>".encode()
    footer = f"<w:ftr xmlns:w='{W_NS}'><w:p><w:r><w:t>www.example.com</w:t></w:r></w:p><w:p><w:r><w:t>www.example.com</w:t></w:r></w:p></w:ftr>".encode()
    _write_package(
        source,
        {
            "word/document.xml": document_xml,
            "word/_rels/document.xml.rels": rels,
            "word/header1.xml": header,
            "word/footer1.xml": footer,
            "word/embeddings/oleObject1.bin": b"opaque-ole",
        },
    )

    parsed = parse_docx(source, tmp_path / "assets")

    kinds = [block.type for block in parsed.blocks]
    assert "textbox" in kinds
    assert "header" in kinds
    assert "footer" in kinds
    assert "noise_candidate" in kinds
    assert "unresolved" in kinds

    body_text = "".join(block.raw_text for block in parsed.blocks if block.type == "paragraph")
    assert "2026 Math Exam" not in body_text
    assert "www.example.com" not in body_text

    textbox = next(block for block in parsed.blocks if block.type == "textbox")
    assert textbox.raw_text == "Textbox text"
    assert textbox.metadata["anchor_hint"]

    unresolved = next(
        block for block in parsed.blocks
        if block.type == "unresolved" and block.metadata.get("embedding_path")
    )
    assert unresolved.metadata["reason"] == "FORMULA_UNRESOLVED"
    assert unresolved.metadata["embedding_path"] == "word/embeddings/oleObject1.bin"
