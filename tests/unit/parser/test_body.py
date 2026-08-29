from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from question_builder.parser.docx.body import parse_docx


def _write_docx(path: Path, document_xml: str) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml.encode())


def test_hyperlink_runs_are_preserved_in_original_paragraph_order(tmp_path: Path) -> None:
    source = tmp_path / "hyperlink.docx"
    _write_docx(
        source,
        """<?xml version='1.0' encoding='UTF-8'?>
<w:document
  xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
  xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>
  <w:body>
    <w:p>
      <w:r><w:t>before </w:t></w:r>
      <w:hyperlink r:id='rId1'><w:r><w:t>linked</w:t></w:r></w:hyperlink>
      <w:r><w:t> after</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
""",
    )

    parsed = parse_docx(source, tmp_path / "assets")

    assert [(block.type, block.raw_text) for block in parsed.blocks] == [
        ("paragraph", "before linked after")
    ]


def test_unsupported_paragraph_child_becomes_explicit_unresolved_in_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unsupported-child.docx"
    _write_docx(
        source,
        """<?xml version='1.0' encoding='UTF-8'?>
<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
  <w:body>
    <w:p>
      <w:r><w:t>before</w:t></w:r>
      <w:custom><w:r><w:t>unsupported</w:t></w:r></w:custom>
      <w:r><w:t>after</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
""",
    )

    parsed = parse_docx(source, tmp_path / "assets")

    assert [(block.type, block.raw_text) for block in parsed.blocks] == [
        ("paragraph", "before"),
        ("unresolved", "unsupported"),
        ("paragraph", "after"),
    ]
    assert parsed.blocks[1].metadata["reason"] == "unsupported_paragraph_child"
