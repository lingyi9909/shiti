from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
V_NS = "urn:schemas-microsoft-com:vml"
W = f"{{{W_NS}}}"
V = f"{{{V_NS}}}"


@dataclass(frozen=True, slots=True)
class TextBoxEvidence:
    text: str
    anchor_hint: str


def find_textboxes(element: ET.Element) -> tuple[TextBoxEvidence, ...]:
    evidence: list[TextBoxEvidence] = []
    for textbox in element.iter(f"{V}textbox"):
        content = textbox.find(f"{W}txbxContent")
        if content is None:
            continue
        text = "".join(node.text or "" for node in content.iter(f"{W}t"))
        parent_shape = _nearest_shape_style(textbox, element)
        evidence.append(TextBoxEvidence(text=text, anchor_hint=parent_shape or "paragraph"))
    for content in element.iter(f"{W}txbxContent"):
        if any(item.text == "".join(node.text or "" for node in content.iter(f"{W}t")) for item in evidence):
            continue
        text = "".join(node.text or "" for node in content.iter(f"{W}t"))
        evidence.append(TextBoxEvidence(text=text, anchor_hint="paragraph"))
    return tuple(evidence)


def _nearest_shape_style(textbox: ET.Element, root: ET.Element) -> str | None:
    for shape in root.iter(f"{V}shape"):
        if textbox in tuple(shape.iter()):
            return shape.attrib.get("style") or shape.attrib.get("id")
    return None
