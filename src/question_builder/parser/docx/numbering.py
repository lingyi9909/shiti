from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


@dataclass(frozen=True, slots=True)
class NumberingLevel:
    abstract_num_id: int
    ilvl: int
    num_fmt: str
    lvl_text: str
    start: int


@dataclass(frozen=True, slots=True)
class ResolvedNumbering:
    num_id: int
    abstract_num_id: int
    ilvl: int
    lvl_text: str
    start: int
    value: int
    resolved_label: str

    def as_metadata(self) -> dict[str, int | str]:
        return {
            "num_id": self.num_id,
            "abstract_num_id": self.abstract_num_id,
            "ilvl": self.ilvl,
            "lvl_text": self.lvl_text,
            "start": self.start,
            "value": self.value,
            "resolved_label": self.resolved_label,
        }


class NumberingResolver:
    def __init__(
        self,
        num_to_abstract: dict[int, int],
        levels: dict[tuple[int, int], NumberingLevel],
    ) -> None:
        self._num_to_abstract = num_to_abstract
        self._levels = levels
        self._counters: dict[tuple[int, int], int] = {}

    @classmethod
    def empty(cls) -> NumberingResolver:
        return cls({}, {})

    @classmethod
    def from_xml(cls, xml: bytes) -> NumberingResolver:
        root = ET.fromstring(xml)
        levels: dict[tuple[int, int], NumberingLevel] = {}
        for abstract in root.findall(f"{W}abstractNum"):
            abstract_id = _int_attr(abstract, "abstractNumId")
            for level in abstract.findall(f"{W}lvl"):
                ilvl = _int_attr(level, "ilvl")
                start_node = level.find(f"{W}start")
                fmt_node = level.find(f"{W}numFmt")
                text_node = level.find(f"{W}lvlText")
                start = _val_int(start_node, 1)
                num_fmt = _val(fmt_node, "decimal")
                lvl_text = _val(text_node, f"%{ilvl + 1}.")
                levels[(abstract_id, ilvl)] = NumberingLevel(
                    abstract_num_id=abstract_id,
                    ilvl=ilvl,
                    num_fmt=num_fmt,
                    lvl_text=lvl_text,
                    start=start,
                )

        num_to_abstract: dict[int, int] = {}
        for num in root.findall(f"{W}num"):
            num_id = _int_attr(num, "numId")
            abstract_ref = num.find(f"{W}abstractNumId")
            if abstract_ref is not None:
                num_to_abstract[num_id] = _val_int(abstract_ref, 0)
        return cls(num_to_abstract, levels)

    def next_label(self, num_id: int, ilvl: int) -> ResolvedNumbering:
        abstract_id = self._num_to_abstract[num_id]
        level = self._levels[(abstract_id, ilvl)]
        key = (num_id, ilvl)
        value = self._counters.get(key, level.start - 1) + 1
        self._counters[key] = value

        for deeper_key in tuple(self._counters):
            if deeper_key[0] == num_id and deeper_key[1] > ilvl:
                del self._counters[deeper_key]

        label = level.lvl_text
        for level_index in range(ilvl + 1):
            referenced = self._levels.get((abstract_id, level_index))
            if referenced is None:
                continue
            ref_key = (num_id, level_index)
            ref_value = self._counters.get(ref_key, referenced.start)
            label = label.replace(
                f"%{level_index + 1}", _format_number(ref_value, referenced.num_fmt)
            )

        return ResolvedNumbering(
            num_id=num_id,
            abstract_num_id=abstract_id,
            ilvl=ilvl,
            lvl_text=level.lvl_text,
            start=level.start,
            value=value,
            resolved_label=label,
        )


def _val(node: ET.Element | None, default: str) -> str:
    if node is None:
        return default
    return node.attrib.get(f"{W}val", default)


def _val_int(node: ET.Element | None, default: int) -> int:
    return int(_val(node, str(default)))


def _int_attr(node: ET.Element, name: str) -> int:
    return int(node.attrib[f"{W}{name}"])


def _format_number(value: int, num_fmt: str) -> str:
    if num_fmt == "upperLetter":
        return _letters(value, upper=True)
    if num_fmt == "lowerLetter":
        return _letters(value, upper=False)
    if num_fmt in {"chineseCounting", "chineseCountingThousand", "ideographTraditional"}:
        return _chinese_number(value)
    return str(value)


def _letters(value: int, *, upper: bool) -> str:
    chars: list[str] = []
    current = value
    base = ord("A" if upper else "a")
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        chars.append(chr(base + remainder))
    return "".join(reversed(chars))


def _chinese_number(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value == 10:
        return "十"
    if value < 20:
        return "十" + digits[value % 10]
    if value < 100:
        tens, ones = divmod(value, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    return str(value)
