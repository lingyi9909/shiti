from __future__ import annotations

from xml.etree import ElementTree as ET

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
M = f"{{{M_NS}}}"


class FormulaConversionError(ValueError):
    """Raised when OMML cannot be converted without losing critical meaning."""


def omml_to_latex(element: ET.Element) -> str:
    latex = _convert(element).strip()
    if not latex or not _balanced_braces(latex):
        raise FormulaConversionError("invalid or empty OMML conversion")
    return latex


def _convert(node: ET.Element) -> str:
    tag = node.tag
    if tag in {f"{M}oMath", f"{M}oMathPara", f"{M}e", f"{M}num", f"{M}den", f"{M}sup", f"{M}sub"}:
        return "".join(_convert(child) for child in node)
    if tag == f"{M}r":
        return "".join(child.text or "" for child in node if child.tag == f"{M}t")
    if tag == f"{M}t":
        return node.text or ""
    if tag == f"{M}f":
        numerator = node.find(f"{M}num")
        denominator = node.find(f"{M}den")
        if numerator is None or denominator is None:
            raise FormulaConversionError("unsupported OMML fraction shape")
        return rf"\frac{{{_convert(numerator)}}}{{{_convert(denominator)}}}"
    if tag == f"{M}sSup":
        base = node.find(f"{M}e")
        sup = node.find(f"{M}sup")
        if base is None or sup is None:
            raise FormulaConversionError("unsupported OMML superscript shape")
        return f"{_convert(base)}^{{{_convert(sup)}}}"
    if tag == f"{M}sSub":
        base = node.find(f"{M}e")
        sub = node.find(f"{M}sub")
        if base is None or sub is None:
            raise FormulaConversionError("unsupported OMML subscript shape")
        return f"{_convert(base)}_{{{_convert(sub)}}}"
    if tag == f"{M}rad":
        degree = node.find(f"{M}deg")
        base = node.find(f"{M}e")
        if base is None:
            raise FormulaConversionError("unsupported OMML radical shape")
        if degree is not None and _convert(degree).strip():
            return rf"\sqrt[{_convert(degree)}]{{{_convert(base)}}}"
        return rf"\sqrt{{{_convert(base)}}}"
    if tag == f"{M}d":
        base = node.find(f"{M}e")
        if base is None:
            raise FormulaConversionError("unsupported OMML delimiter shape")
        props = node.find(f"{M}dPr")
        begin = "("
        end = ")"
        if props is not None:
            begin_node = props.find(f"{M}begChr")
            end_node = props.find(f"{M}endChr")
            if begin_node is not None:
                begin = begin_node.attrib.get(f"{M}val", begin)
            if end_node is not None:
                end = end_node.attrib.get(f"{M}val", end)
        return f"{begin}{_convert(base)}{end}"
    if tag in {f"{M}dPr", f"{M}radPr", f"{M}fPr", f"{M}sSupPr", f"{M}sSubPr"}:
        return ""
    raise FormulaConversionError(f"unsupported OMML construct: {tag}")


def _balanced_braces(value: str) -> bool:
    depth = 0
    for char in value:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
