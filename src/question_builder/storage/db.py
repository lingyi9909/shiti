from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import aiosqlite


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
    token_usage: int | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    input_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_states (
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'REJECTED')),
    error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stage, subject_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS provider_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task TEXT NOT NULL,
    request_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    normalized_score REAL,
    cache_hit INTEGER NOT NULL CHECK (cache_hit IN (0, 1)),
    fallback_reason TEXT,
    token_usage INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cache_index (
    cache_key TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    recognition_task TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_ALLOWED_TRANSITIONS: dict[StageState, frozenset[StageState]] = {
    StageState.PENDING: frozenset({StageState.RUNNING}),
    StageState.RUNNING: frozenset(
        {StageState.COMPLETED, StageState.FAILED, StageState.REJECTED}
    ),
    StageState.FAILED: frozenset({StageState.RUNNING}),
    StageState.COMPLETED: frozenset(),
    StageState.REJECTED: frozenset(),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def compute_run_fingerprint(*, input_hash: str, config_hash: str, pipeline_version: str) -> str:
    _require_nonempty(input_hash, "input_hash")
    _require_nonempty(config_hash, "config_hash")
    _require_nonempty(pipeline_version, "pipeline_version")
    canonical = json.dumps(
        {
            "config_hash": config_hash,
            "input_hash": input_hash,
            "pipeline_version": pipeline_version,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "run_" + hashlib.sha256(canonical).hexdigest()


async def ensure_storage_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as connection:
        await connection.execute("PRAGMA foreign_keys = ON")
        await connection.executescript(_SCHEMA)
        await connection.commit()


class RunStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        await ensure_storage_schema(self.db_path)

    async def ensure_run(
        self,
        *,
        input_hash: str,
        config_hash: str,
        pipeline_version: str,
    ) -> RunRecord:
        run_id = compute_run_fingerprint(
            input_hash=input_hash,
            config_hash=config_hash,
            pipeline_version=pipeline_version,
        )
        now = _utc_now()
        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute(
                """
                INSERT OR IGNORE INTO runs (
                    run_id, input_hash, config_hash, pipeline_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, input_hash, config_hash, pipeline_version, now, now),
            )
            await connection.commit()
            cursor = await connection.execute(
                """
                SELECT run_id, input_hash, config_hash, pipeline_version
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            raise RuntimeError("run record was not persisted")
        return RunRecord(
            run_id=str(row[0]),
            input_hash=str(row[1]),
            config_hash=str(row[2]),
            pipeline_version=str(row[3]),
        )

    async def ensure_stage(
        self,
        run_id: str,
        stage: str,
        *,
        subject_id: str = "__run__",
    ) -> StageRecord:
        _require_nonempty(run_id, "run_id")
        _require_nonempty(stage, "stage")
        _require_nonempty(subject_id, "subject_id")
        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute(
                """
                INSERT OR IGNORE INTO stage_states (
                    run_id, stage, subject_id, state, error, updated_at
                ) VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (run_id, stage, subject_id, StageState.PENDING.value, _utc_now()),
            )
            await connection.commit()
            return await self._fetch_stage(connection, run_id, stage, subject_id)

    async def transition_stage(
        self,
        run_id: str,
        stage: str,
        new_state: StageState,
        *,
        subject_id: str = "__run__",
        error: str | None = None,
    ) -> StageRecord:
        _require_nonempty(run_id, "run_id")
        _require_nonempty(stage, "stage")
        _require_nonempty(subject_id, "subject_id")
        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._fetch_stage(connection, run_id, stage, subject_id)
                if current.state is new_state:
                    await connection.commit()
                    return current
                if new_state not in _ALLOWED_TRANSITIONS[current.state]:
                    raise InvalidStageTransitionError(
                        f"invalid stage transition: {current.state.value} -> {new_state.value}"
                    )
                await connection.execute(
                    """
                    UPDATE stage_states
                    SET state = ?, error = ?, updated_at = ?
                    WHERE run_id = ? AND stage = ? AND subject_id = ?
                    """,
                    (
                        new_state.value,
                        error,
                        _utc_now(),
                        run_id,
                        stage,
                        subject_id,
                    ),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
            return StageRecord(
                run_id=run_id,
                stage=stage,
                subject_id=subject_id,
                state=new_state,
                error=error,
            )

    async def first_incomplete_stage(
        self,
        run_id: str,
        stages: Sequence[str],
        *,
        subject_id: str = "__run__",
    ) -> str | None:
        async with aiosqlite.connect(self.db_path) as connection:
            for stage in stages:
                _require_nonempty(stage, "stage")
                cursor = await connection.execute(
                    """
                    SELECT state FROM stage_states
                    WHERE run_id = ? AND stage = ? AND subject_id = ?
                    """,
                    (run_id, stage, subject_id),
                )
                row = await cursor.fetchone()
                if row is None:
                    return stage
                state = StageState(str(row[0]))
                if state is StageState.REJECTED:
                    return None
                if state is not StageState.COMPLETED:
                    return stage
        return None

    async def record_provider_call(self, call: ProviderCallRecord) -> None:
        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute(
                """
                INSERT INTO provider_calls (
                    run_id, stage, provider, model, task, request_id,
                    prompt_version, content_hash, latency_ms, normalized_score,
                    cache_hit, fallback_reason, token_usage, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.run_id,
                    call.stage,
                    call.provider,
                    call.model,
                    call.task,
                    call.request_id,
                    call.prompt_version,
                    call.content_hash,
                    call.latency_ms,
                    call.normalized_score,
                    int(call.cache_hit),
                    call.fallback_reason,
                    call.token_usage,
                    _utc_now(),
                ),
            )
            await connection.commit()

    async def list_provider_calls(self, run_id: str) -> tuple[ProviderCallRecord, ...]:
        async with aiosqlite.connect(self.db_path) as connection:
            cursor = await connection.execute(
                """
                SELECT run_id, stage, provider, model, task, request_id,
                       prompt_version, content_hash, latency_ms, normalized_score,
                       cache_hit, fallback_reason, token_usage
                FROM provider_calls
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            )
            rows = await cursor.fetchall()
        return tuple(
            ProviderCallRecord(
                run_id=str(row[0]),
                stage=str(row[1]),
                provider=str(row[2]),
                model=str(row[3]),
                task=str(row[4]),
                request_id=str(row[5]),
                prompt_version=str(row[6]),
                content_hash=str(row[7]),
                latency_ms=float(row[8]),
                normalized_score=None if row[9] is None else float(row[9]),
                cache_hit=bool(row[10]),
                fallback_reason=None if row[11] is None else str(row[11]),
                token_usage=None if row[12] is None else int(row[12]),
            )
            for row in rows
        )

    @staticmethod
    async def _fetch_stage(
        connection: aiosqlite.Connection,
        run_id: str,
        stage: str,
        subject_id: str,
    ) -> StageRecord:
        cursor = await connection.execute(
            """
            SELECT run_id, stage, subject_id, state, error
            FROM stage_states
            WHERE run_id = ? AND stage = ? AND subject_id = ?
            """,
            (run_id, stage, subject_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"stage state not found: {run_id}/{stage}/{subject_id}")
        return StageRecord(
            run_id=str(row[0]),
            stage=str(row[1]),
            subject_id=str(row[2]),
            state=StageState(str(row[3])),
            error=None if row[4] is None else str(row[4]),
        )
