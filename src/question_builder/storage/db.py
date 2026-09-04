from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Sequence


class StageState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class InvalidStageTransitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    input_hash: str
    config_hash: str
    pipeline_version: str


@dataclass(frozen=True, slots=True)
class StageRecord:
    run_id: str
    stage: str
    subject_id: str
    state: StageState
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderCallRecord:
    run_id: str
    stage: str
    provider: str
    model: str
    task: str
    request_id: str
    prompt_version: str
    content_hash: str
    latency_ms: float
    normalized_score: float | None
    cache_hit: bool
    fallback_reason: str | None


def compute_run_fingerprint(*, input_hash: str, config_hash: str, pipeline_version: str) -> str:
    raise NotImplementedError


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        raise NotImplementedError

    async def ensure_run(
        self,
        *,
        input_hash: str,
        config_hash: str,
        pipeline_version: str,
    ) -> RunRecord:
        raise NotImplementedError

    async def ensure_stage(
        self,
        run_id: str,
        stage: str,
        *,
        subject_id: str = "__run__",
    ) -> StageRecord:
        raise NotImplementedError

    async def transition_stage(
        self,
        run_id: str,
        stage: str,
        new_state: StageState,
        *,
        subject_id: str = "__run__",
        error: str | None = None,
    ) -> StageRecord:
        raise NotImplementedError

    async def first_incomplete_stage(
        self,
        run_id: str,
        stages: Sequence[str],
        *,
        subject_id: str = "__run__",
    ) -> str | None:
        raise NotImplementedError

    async def record_provider_call(self, call: ProviderCallRecord) -> None:
        raise NotImplementedError

    async def list_provider_calls(self, run_id: str) -> tuple[ProviderCallRecord, ...]:
        raise NotImplementedError
