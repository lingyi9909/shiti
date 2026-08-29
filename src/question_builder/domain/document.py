from __future__ import annotations

from typing import Any, Literal, Never, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T = TypeVar("T")

BlockType = Literal[
    "paragraph",
    "formula",
    "image",
    "table",
    "textbox",
    "header",
    "footer",
    "noise_candidate",
    "unresolved",
]


class FrozenDict(dict[str, Any]):
    """A dict-shaped value that cannot be mutated after validation."""

    @staticmethod
    def _immutable() -> Never:
        raise TypeError("domain mappings are immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        self._immutable()

    def __delitem__(self, key: str) -> None:
        self._immutable()

    def clear(self) -> None:
        self._immutable()

    def pop(self, key: str, default: Any = None) -> Any:
        self._immutable()

    def popitem(self) -> tuple[str, Any]:
        self._immutable()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._immutable()

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._immutable()

    def __ior__(self, value: Any) -> FrozenDict:
        self._immutable()


def freeze_value(value: T) -> T:
    """Recursively freeze mutable containers while preserving serializable shapes."""

    if isinstance(value, dict):
        return cast(T, FrozenDict({key: freeze_value(item) for key, item in value.items()}))
    if isinstance(value, (list, tuple)):
        return cast(T, tuple(freeze_value(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return cast(T, frozenset(freeze_value(item) for item in value))
    return value


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContentBlock(DomainModel):
    block_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    type: BlockType
    raw_text: str = ""
    normalized_text: str | None = None
    style_id: str | None = None
    numbering: dict[str, Any] | None = None
    source_type: str | None = None
    source_xml_path: str | None = None
    relationship_id: str | None = None
    recognized: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("numbering", "recognized", "metadata", mode="after")
    @classmethod
    def freeze_mapping(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return freeze_value(value)


class DocumentIR(DomainModel):
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocks: tuple[ContentBlock, ...] = ()

    @model_validator(mode="after")
    def validate_block_identity_and_order(self) -> DocumentIR:
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("DocumentIR block_id values must be unique")

        orders = [block.order for block in self.blocks]
        if len(orders) != len(set(orders)):
            raise ValueError("DocumentIR block order values must be unique")
        if orders != sorted(orders):
            raise ValueError("DocumentIR blocks must preserve strictly increasing source order")
        return self
