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
    for content in element.iter(f"{W}txbxContent"):
        text = "".join(node.text or "" for node in content.iter(f"{W}t"))
        anchor_hint = _nearest_shape_style(content, element) or "paragraph"
        evidence.append(TextBoxEvidence(text=text, anchor_hint=anchor_hint))
    return tuple(evidence)


def _nearest_shape_style(descendant: ET.Element, root: ET.Element) -> str | None:
    for shape in root.iter(f"{V}shape"):
        if descendant in tuple(shape.iter()):
            return shape.attrib.get("style") or shape.attrib.get("id")
    return None
