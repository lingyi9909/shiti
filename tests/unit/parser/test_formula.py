from xml.etree import ElementTree as ET

import pytest

from question_builder.parser.docx.formula import FormulaConversionError, omml_to_latex

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _math(body: str) -> ET.Element:
    return ET.fromstring(f"<m:oMath xmlns:m='{M_NS}'>{body}</m:oMath>")


def test_fraction_superscript_subscript_root_and_grouping_convert_to_latex() -> None:
    assert omml_to_latex(
        _math(
            "<m:f><m:num><m:r><m:t>a</m:t></m:r></m:num>"
            "<m:den><m:r><m:t>b</m:t></m:r></m:den></m:f>"
        )
    ) == r"\frac{a}{b}"
    assert omml_to_latex(
        _math(
            "<m:sSup><m:e><m:r><m:t>x</m:t></m:r></m:e>"
            "<m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>"
        )
    ) == "x^{2}"
    assert omml_to_latex(
        _math(
            "<m:sSub><m:e><m:r><m:t>x</m:t></m:r></m:e>"
            "<m:sub><m:r><m:t>1</m:t></m:r></m:sub></m:sSub>"
        )
    ) == "x_{1}"
    assert omml_to_latex(
        _math("<m:rad><m:e><m:r><m:t>x+1</m:t></m:r></m:e></m:rad>")
    ) == r"\sqrt{x+1}"
    assert omml_to_latex(
        _math(
            "<m:d><m:dPr><m:begChr m:val='('/><m:endChr m:val=')'/>"
            "</m:dPr><m:e><m:r><m:t>x+1</m:t></m:r></m:e></m:d>"
        )
    ) == "(x+1)"


def test_unsupported_construct_raises_and_source_is_never_silently_empty() -> None:
    unsupported = _math("<m:matrix><m:r><m:t>x</m:t></m:r></m:matrix>")

    with pytest.raises(FormulaConversionError, match="unsupported OMML"):
        omml_to_latex(unsupported)


def test_empty_or_unbalanced_conversion_is_rejected() -> None:
    with pytest.raises(FormulaConversionError):
        omml_to_latex(_math(""))
