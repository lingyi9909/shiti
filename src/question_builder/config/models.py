from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model for contracts that reject unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class QualityThresholds(StrictModel):
    precision_first: bool = True
    noncritical_recognition_accept: float = Field(default=0.95, ge=0.0, le=1.0)
    critical_recognition_accept: float = Field(default=0.98, ge=0.0, le=1.0)
    recognition_fallback_floor: float = Field(default=0.90, ge=0.0, le=1.0)
    split_accept: float = Field(default=0.98, ge=0.0, le=1.0)
    answer_match_accept: float = Field(default=0.995, ge=0.0, le=1.0)
    answer_match_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    answer_verify_accept: float = Field(default=0.995, ge=0.0, le=1.0)
    reject_on_missing_answer: bool = True
    reject_on_unresolved_formula: bool = True
    reject_on_ambiguous_match: bool = True

    @model_validator(mode="after")
    def validate_recognition_threshold_order(self) -> QualityThresholds:
        if self.recognition_fallback_floor > self.critical_recognition_accept:
            raise ValueError(
                "recognition_fallback_floor must be <= critical_recognition_accept"
            )
        return self


class ProviderConfig(StrictModel):
    primary: str
    fallback: str | None = None
    verifier: str | None = None


class ProvidersConfig(StrictModel):
    text_ocr: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(primary="provider_a", fallback="provider_b")
    )
    formula_ocr: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(primary="provider_c", fallback="provider_d")
    )
    table: ProviderConfig = Field(default_factory=lambda: ProviderConfig(primary="provider_e"))
    vision: ProviderConfig = Field(default_factory=lambda: ProviderConfig(primary="provider_f"))
    llm: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(primary="provider_g", verifier="provider_h")
    )


class ConcurrencyConfig(StrictModel):
    document_parse: int = Field(default=4, ge=1)
    text_ocr: int = Field(default=8, ge=1)
    formula_ocr: int = Field(default=4, ge=1)
    table: int = Field(default=4, ge=1)
    vision: int = Field(default=4, ge=1)
    llm: int = Field(default=6, ge=1)


class OutputConfig(StrictModel):
    copyright_default: str = "0"
    slim_md5_version: str = "slim_md5_v1"


class AppConfig(StrictModel):
    pipeline_version: str = "0.1.0"
    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return AppConfig.model_validate(raw)
