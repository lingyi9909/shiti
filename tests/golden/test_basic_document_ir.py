from pathlib import Path
from zipfile import ZipFile

from question_builder.parser.docx.body import parse_docx

FIXTURE = Path("fixtures/synthetic/basic_ordered.docx")


def test_basic_document_ir_preserves_exact_body_and_run_child_order(tmp_path: Path) -> None:
    assets = tmp_path / "assets"

    parsed = parse_docx(FIXTURE, assets)

    assert parsed.source_file == FIXTURE.name
    assert parsed.document_id == f"doc_{parsed.source_sha256[:16]}"
    assert [block.type for block in parsed.blocks] == [
        "paragraph",
        "paragraph",
        "image",
        "paragraph",
        "table",
        "paragraph",
    ]
    assert [block.raw_text for block in parsed.blocks] == [
        "First question",
        "before ",
        "",
        " after",
        "table boundary",
        "Last paragraph",
    ]
    assert parsed.blocks[0].numbering is not None
    assert parsed.blocks[0].numbering["resolved_label"] == "1."
    image = parsed.blocks[2]
    assert image.relationship_id
    assert image.metadata["asset_id"].startswith("img_")
    assert image.metadata["asset_sha256"]
    materialized = assets / image.metadata["asset_filename"]
    assert materialized.is_file()
    with ZipFile(FIXTURE) as archive:
        assert materialized.read_bytes() == archive.read(image.metadata["source_member"])
    assert [block.order for block in parsed.blocks] == list(range(len(parsed.blocks)))
    assert all(block.source_xml_path for block in parsed.blocks)
