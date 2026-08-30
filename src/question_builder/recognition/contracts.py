from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RecognitionTask(StrEnum):
    TEXT_OCR = "text_ocr"
    FORMULA_OCR = "formula_ocr"
    TABLE_RECOGNITION = "table_recognition"
    VISION = "vision"
    LLM = "llm"


class ImageClass(StrEnum):
    TEXT_IMAGE = "TEXT_IMAGE"
    FORMULA_IMAGE = "FORMULA_IMAGE"
    TABLE_IMAGE = "TABLE_IMAGE"
    QUESTION_SCREENSHOT = "QUESTION_SCREENSHOT"
    DIAGRAM = "DIAGRAM"
    GEOMETRY = "GEOMETRY"
    CHART = "CHART"
    MAP = "MAP"
    CHEMISTRY = "CHEMISTRY"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecognitionRequest(ContractModel):
    task: RecognitionTask
    image_class: ImageClass
    input_ref: str = Field(min_length=1)
    critical: bool = True


class ProviderOutput(ContractModel):
    request_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    raw_score_reference: str = Field(min_length=1)
    content: str


class RecognitionResult(ContractModel):
    task: RecognitionTask
    image_class: ImageClass
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    raw_score: float = Field(ge=0.0, le=1.0)
    raw_score_reference: str = Field(min_length=1)
    normalized_score: float = Field(ge=0.0, le=1.0)
    calibration_id: str = Field(min_length=1)
    content: str


@runtime_checkable
class TextOCRProvider(Protocol):
    provider: str
    model: str

    async def recognize_text(self, request: RecognitionRequest) -> ProviderOutput: ...


@runtime_checkable
class FormulaOCRProvider(Protocol):
    provider: str
    model: str

    async def recognize_formula(self, request: RecognitionRequest) -> ProviderOutput: ...


@runtime_checkable
class TableRecognitionProvider(Protocol):
    provider: str
    model: str

    async def recognize_table(self, request: RecognitionRequest) -> ProviderOutput: ...


@runtime_checkable
class VisionProvider(Protocol):
    provider: str
    model: str

    async def recognize_vision(self, request: RecognitionRequest) -> ProviderOutput: ...


@runtime_checkable
class LLMProvider(Protocol):
    provider: str
    model: str

    async def complete(self, request: RecognitionRequest) -> ProviderOutput: ...
