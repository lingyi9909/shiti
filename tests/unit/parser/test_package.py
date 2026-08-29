from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from question_builder.parser.docx.package import DocxPackage, DocxPackageError


def write_docx(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_invalid_zip_fails_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "broken.docx"
    source.write_bytes(b"not-a-zip")

    with pytest.raises(DocxPackageError, match="valid DOCX ZIP"):
        DocxPackage.open(source)


def test_document_xml_is_mandatory(tmp_path: Path) -> None:
    source = tmp_path / "missing-document.docx"
    write_docx(source, {"[Content_Types].xml": b"<Types/>"})

    with pytest.raises(DocxPackageError, match="word/document.xml"):
        DocxPackage.open(source)


def test_source_sha256_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "stable.docx"
    write_docx(source, {"word/document.xml": b"<document/>"})

    first = DocxPackage.open(source)
    second = DocxPackage.open(source)

    assert first.source_sha256 == second.source_sha256
    assert len(first.source_sha256) == 64
