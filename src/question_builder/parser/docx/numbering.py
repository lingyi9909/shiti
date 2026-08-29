from __future__ import annotations

from dataclasses import dataclass, replace
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

    def as_metadata(self) -> dict[str, object]:
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
        overrides: dict[tuple[int, int], NumberingLevel] | None = None,
    ) -> None:
        self._num_to_abstract = num_to_abstract
        self._levels = levels
        self._overrides = overrides or {}
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
            for level_node in abstract.findall(f"{W}lvl"):
                level = _parse_level(level_node, abstract_id)
                levels[(abstract_id, level.ilvl)] = level

        num_to_abstract: dict[int, int] = {}
        overrides: dict[tuple[int, int], NumberingLevel] = {}
        for num in root.findall(f"{W}num"):
            num_id = _int_attr(num, "numId")
            abstract_ref = num.find(f"{W}abstractNumId")
            if abstract_ref is None:
                continue
            abstract_id = _val_int(abstract_ref, 0)
            num_to_abstract[num_id] = abstract_id
            for override_node in num.findall(f"{W}lvlOverride"):
                ilvl = _int_attr(override_node, "ilvl")
                base_level = levels[(abstract_id, ilvl)]
                embedded_level = override_node.find(f"{W}lvl")
                if embedded_level is None:
                    override_level = base_level
                else:
                    override_level = _parse_level(
                        embedded_level,
                        abstract_id,
                        fallback=base_level,
                        forced_ilvl=ilvl,
                    )
                start_override = override_node.find(f"{W}startOverride")
                if start_override is not None:
                    override_level = replace(
                        override_level,
                        start=_val_int(start_override, override_level.start),
                    )
                overrides[(num_id, ilvl)] = override_level
        return cls(num_to_abstract, levels, overrides)

    def next_label(self, num_id: int, ilvl: int) -> ResolvedNumbering:
        abstract_id = self._num_to_abstract[num_id]
        level = self._level_for(num_id, abstract_id, ilvl)
        key = (num_id, ilvl)
        value = self._counters.get(key, level.start - 1) + 1
        self._counters[key] = value

        for deeper_key in tuple(self._counters):
            if deeper_key[0] == num_id and deeper_key[1] > ilvl:
                del self._counters[deeper_key]

        label = level.lvl_text
        for level_index in range(ilvl + 1):
            referenced = self._level_for(num_id, abstract_id, level_index)
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

    def _level_for(self, num_id: int, abstract_id: int, ilvl: int) -> NumberingLevel:
        return self._overrides.get((num_id, ilvl), self._levels[(abstract_id, ilvl)])


def _parse_level(
    node: ET.Element,
    abstract_id: int,
    *,
    fallback: NumberingLevel | None = None,
    forced_ilvl: int | None = None,
) -> NumberingLevel:
    ilvl = forced_ilvl if forced_ilvl is not None else _int_attr(node, "ilvl")
    default_start = fallback.start if fallback is not None else 1
    default_fmt = fallback.num_fmt if fallback is not None else "decimal"
    default_text = fallback.lvl_text if fallback is not None else f"%{ilvl + 1}."
    return NumberingLevel(
        abstract_num_id=abstract_id,
        ilvl=ilvl,
        num_fmt=_val(node.find(f"{W}numFmt"), default_fmt),
        lvl_text=_val(node.find(f"{W}lvlText"), default_text),
        start=_val_int(node.find(f"{W}start"), default_start),
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
    if num_fmt == "decimalEnclosedCircle":
        return _enclosed_circle(value)
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


def _enclosed_circle(value: int) -> str:
    if 1 <= value <= 20:
        return chr(0x2460 + value - 1)
    return str(value)
