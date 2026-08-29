from xml.etree import ElementTree as ET

from question_builder.parser.docx.table import parse_table

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def _xml(body: str) -> ET.Element:
    return ET.fromstring(
        f"<w:tbl xmlns:w='{W_NS}'>{body}</w:tbl>"
    )


def test_plain_table_exposes_markdown_and_cell_order() -> None:
    table = _xml(
        """
        <w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>B</w:t></w:r></w:p></w:tc></w:tr>
        <w:tr><w:tc><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>D</w:t></w:r></w:p></w:tc></w:tr>
        """
    )

    parsed = parse_table(table)

    assert parsed.is_complex is False
    assert parsed.rows == (("A", "B"), ("C", "D"))
    assert parsed.rendered == "| A | B |\n| --- | --- |\n| C | D |"


def test_gridspan_and_vmerge_expose_html_and_merge_evidence() -> None:
    table = _xml(
        """
        <w:tr>
          <w:tc><w:tcPr><w:gridSpan w:val='2'/></w:tcPr><w:p><w:r><w:t>AB</w:t></w:r></w:p></w:tc>
        </w:tr>
        <w:tr>
          <w:tc><w:tcPr><w:vMerge w:val='restart'/></w:tcPr><w:p><w:r><w:t>C</w:t></w:r></w:p></w:tc>
          <w:tc><w:p><w:r><w:t>D</w:t></w:r></w:p></w:tc>
        </w:tr>
        <w:tr>
          <w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p/></w:tc>
          <w:tc><w:p><w:r><w:t>E</w:t></w:r></w:p></w:tc>
        </w:tr>
        """
    )

    parsed = parse_table(table)

    assert parsed.is_complex is True
    assert "colspan=\"2\"" in parsed.rendered
    assert "rowspan=\"2\"" in parsed.rendered
    assert parsed.merges == (
        {"row": 0, "col": 0, "colspan": 2, "rowspan": 1},
        {"row": 1, "col": 0, "colspan": 1, "rowspan": 2},
    )


def test_cell_preserves_text_formula_and_image_child_order() -> None:
    table = ET.fromstring(
        f"""
        <w:tbl xmlns:w='{W_NS}' xmlns:m='http://schemas.openxmlformats.org/officeDocument/2006/math'
          xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'
          xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>
          <w:tr><w:tc><w:p>
            <w:r><w:t>before</w:t></w:r>
            <m:oMath><m:r><m:t>x</m:t></m:r></m:oMath>
            <w:r><w:drawing><a:blip r:embed='rId9'/></w:drawing></w:r>
            <w:r><w:t>after</w:t></w:r>
          </w:p></w:tc></w:tr>
        </w:tbl>
        """
    )

    parsed = parse_table(table)

    assert parsed.cells[0].content_kinds == ("text", "formula", "image", "text")
    assert parsed.cells[0].relationship_ids == ("rId9",)
