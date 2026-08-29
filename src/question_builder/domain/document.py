from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class DocumentIR(DomainModel):
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocks: list[ContentBlock] = Field(default_factory=list)

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
