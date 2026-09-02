from __future__ import annotations

import pytest

from question_builder.domain.document import ContentBlock, DocumentIR
from question_builder.domain.quality import RejectReason
from question_builder.export.markdown import QuestionContentError, render_question_markdown


def _document(table_metadata: dict[str, object]) -> DocumentIR:
    return DocumentIR(
        document_id="doc_table_cell_reconstruction",
        source_file="表格试题.docx",
        source_sha256="b" * 64,
        blocks=(
            ContentBlock(
                block_id="b1",
                order=1,
                type="table",
                raw_text="beforeafter",
                metadata=table_metadata,
            ),
        ),
    )


def _cell(contents: tuple[dict[str, str], ...]) -> dict[str, object]:
    return {
        "row": 0,
        "col": 0,
        "colspan": 1,
        "rowspan": 1,
        "content_kinds": tuple(content["kind"] for content in contents),
        "relationship_ids": tuple(
            content["relationship_id"]
            for content in contents
            if content.get("relationship_id")
        ),
        "contents": contents,
    }


def test_table_cell_image_is_reconstructed_in_source_order() -> None:
    contents = (
        {"kind": "text", "text": "before"},
        {"kind": "image", "relationship_id": "rId9"},
        {"kind": "text", "text": "after"},
    )
    document = _document(
        {
            "render_format": "markdown",
            "rendered": "| beforeafter |\n| --- |",
            "assets": (
                {
                    "relationship_id": "rId9",
                    "asset_filename": "cell_image.png",
                },
            ),
            "cells": (_cell(contents),),
        }
    )

    rendered = render_question_markdown(document, ("b1",))

    image = '<img src="image/cell_image.png">'
    assert image in rendered
    assert rendered.index("before") < rendered.index(image) < rendered.index("after")
    assert rendered == f"| before{image}after |\n| --- |"


def test_table_cell_image_without_asset_mapping_is_rejected() -> None:
    contents = (
        {"kind": "text", "text": "before"},
        {"kind": "image", "relationship_id": "rId404"},
        {"kind": "text", "text": "after"},
    )
    document = _document(
        {
            "render_format": "markdown",
            "rendered": "| beforeafter |\n| --- |",
            "assets": (),
            "cells": (_cell(contents),),
        }
    )

    with pytest.raises(QuestionContentError) as exc_info:
        render_question_markdown(document, ("b1",))

    assert exc_info.value.reason_code is RejectReason.QUESTION_CONTENT_INCOMPLETE
