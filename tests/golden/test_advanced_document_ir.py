from pathlib import Path

from question_builder.parser.docx.body import parse_docx


def test_advanced_document_ir_preserves_structured_and_unresolved_evidence(
    tmp_path: Path,
) -> None:
    source = Path("fixtures/synthetic/advanced_content.docx")

    parsed = parse_docx(source, tmp_path / "assets")

    assert [(block.type, block.raw_text) for block in parsed.blocks[:7]] == [
        ("paragraph", "Before "),
        ("formula", r"\frac{a}{b}"),
        ("paragraph", " after"),
        ("table", '<table>\n  <tr>\n    <td colspan="2">Merged</td>\n  </tr>\n</table>'),
        ("textbox", "Boxed evidence"),
        ("unresolved", ""),
        ("header", "2026 Math Exam"),
    ]
    table = parsed.blocks[3]
    assert table.metadata["render_format"] == "html"
    assert table.metadata["merges"] == (
        {"row": 0, "col": 0, "colspan": 2, "rowspan": 1},
    )
    formula = parsed.blocks[1]
    assert formula.metadata["source_omml"]
    unresolved = parsed.blocks[5]
    assert unresolved.metadata["reason"] == "FORMULA_UNRESOLVED"
    assert unresolved.metadata["embedding_path"] == "word/embeddings/oleObject1.bin"
    assert any(block.type == "footer" for block in parsed.blocks)
    assert any(block.type == "noise_candidate" for block in parsed.blocks)
