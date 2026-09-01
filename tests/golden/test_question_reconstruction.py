from __future__ import annotations

import pytest

import question_builder.export.markdown as markdown_export
from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason


def _document(blocks: tuple[ContentBlock, ...]) -> DocumentIR:
    return DocumentIR(
        document_id="doc_reconstruction",
        source_file="重建试题.docx",
        source_sha256="f" * 64,
        blocks=blocks,
    )


def test_question_markdown_reconstructs_source_blocks_in_exact_order() -> None:
    render = getattr(markdown_export, "render_question_markdown", None)
    assert callable(render)
    document = _document(
        (
            ContentBlock(
                block_id="b1",
                order=1,
                type="paragraph",
                raw_text="原始题干",
                normalized_text="规范题干",
            ),
            ContentBlock(
                block_id="b2",
                order=2,
                type="formula",
                raw_text=r"\frac{a}{b}",
                normalized_text=r"\frac{a}{b}",
            ),
            ContentBlock(
                block_id="b3",
                order=3,
                type="paragraph",
                raw_text="继续作答",
                normalized_text="继续作答",
            ),
            ContentBlock(
                block_id="b4",
                order=4,
                type="image",
                raw_text="",
                metadata={"asset_filename": "img_deadbeef.png"},
            ),
            ContentBlock(
                block_id="b5",
                order=5,
                type="paragraph",
                raw_text="观察下表",
                normalized_text="观察下表",
            ),
            ContentBlock(
                block_id="b6",
                order=6,
                type="table",
                raw_text="A B 1 2",
                metadata={
                    "render_format": "markdown",
                    "rendered": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                },
            ),
        )
    )

    rendered = render(document, ("b1", "b2", "b3", "b4", "b5", "b6"))

    assert rendered == (
        "规范题干\n\n"
        "$\\frac{a}{b}$\n\n"
        "继续作答\n\n"
        '<img src="image/img_deadbeef.png">\n\n'
        "观察下表\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
    )


def test_complex_table_html_is_preserved_without_model_rewrite() -> None:
    render = getattr(markdown_export, "render_question_markdown", None)
    assert callable(render)
    html = '<table><tr><td colspan="2">Merged</td></tr></table>'
    document = _document(
        (
            ContentBlock(
                block_id="b1",
                order=1,
                type="table",
                raw_text="Merged",
                metadata={"render_format": "html", "rendered": html},
            ),
        )
    )

    assert render(document, ("b1",)) == html


def test_reconstruction_rejects_missing_image_or_unresolved_critical_content() -> None:
    render = getattr(markdown_export, "render_question_markdown", None)
    error_type = getattr(markdown_export, "QuestionContentError", None)
    assert callable(render)
    assert isinstance(error_type, type)

    missing_image = _document(
        (
            ContentBlock(block_id="b1", order=1, type="image", raw_text=""),
        )
    )
    with pytest.raises(error_type) as image_exc:
        render(missing_image, ("b1",))
    assert image_exc.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE

    unresolved = _document(
        (
            ContentBlock(
                block_id="b1",
                order=1,
                type="unresolved",
                raw_text="",
                metadata={"reason": "FORMULA_UNRESOLVED"},
            ),
        )
    )
    with pytest.raises(error_type) as unresolved_exc:
        render(unresolved, ("b1",))
    assert unresolved_exc.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE
