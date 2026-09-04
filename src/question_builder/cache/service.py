from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from question_builder.storage.db import ensure_storage_schema


class CacheConflictError(RuntimeError):
    pass


class SensitiveCachePayloadError(ValueError):
    pass


class CacheCorruptionError(RuntimeError):
    pass


_SENSITIVE_KEYS = {
    "headers",
    "authorization",
    "apikey",
    "accesskey",
    "accesstoken",
    "refreshtoken",
    "password",
    "secret",
    "clientsecret",
    "cookie",
    "setcookie",
}


@dataclass(frozen=True, slots=True)
class CacheKey:
    content_hash: str
    provider: str
    model_version: str
    prompt_version: str
    recognition_task: str

    def digest(self) -> str:
        for field_name, value in (
            ("content_hash", self.content_hash),
            ("provider", self.provider),
            ("model_version", self.model_version),
            ("prompt_version", self.prompt_version),
            ("recognition_task", self.recognition_task),
        ):
            if not value:
                raise ValueError(f"{field_name} must be non-empty")
        canonical = json.dumps(
            {
                "content_hash": self.content_hash,
                "model_version": self.model_version,
                "prompt_version": self.prompt_version,
                "provider": self.provider,
                "recognition_task": self.recognition_task,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    payload_sha256: str
    relative_path: str


class CacheService:
    def __init__(self, *, db_path: Path, cache_dir: Path) -> None:
        self.db_path = db_path
        self.cache_dir = cache_dir

    async def initialize(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        await ensure_storage_schema(self.db_path)

    async def put(self, key: CacheKey, payload: dict[str, Any]) -> CacheEntry:
        self._reject_sensitive_payload(payload)
        payload_bytes = self._canonical_payload(payload)
        payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        cache_key = key.digest()
        relative_path = Path("payloads") / payload_sha256[:2] / f"{payload_sha256}.json"
        relative_path_text = relative_path.as_posix()

        async with aiosqlite.connect(self.db_path) as connection:
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT payload_sha256, relative_path
                    FROM cache_index WHERE cache_key = ?
                    """,
                    (cache_key,),
                )
                existing = await cursor.fetchone()
                if existing is not None:
                    existing_sha = str(existing[0])
                    if existing_sha != payload_sha256:
                        raise CacheConflictError(
                            "cache key already maps to a different immutable payload"
                        )
                    await connection.commit()
                    return CacheEntry(
                        cache_key=cache_key,
                        payload_sha256=existing_sha,
                        relative_path=str(existing[1]),
                    )

                payload_path = self.cache_dir / relative_path
                self._atomic_write_bytes(payload_path, payload_bytes)
                await connection.execute(
                    """
                    INSERT INTO cache_index (
                        cache_key, content_hash, provider, model_version,
                        prompt_version, recognition_task, payload_sha256,
                        relative_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        cache_key,
                        key.content_hash,
                        key.provider,
                        key.model_version,
                        key.prompt_version,
                        key.recognition_task,
                        payload_sha256,
                        relative_path_text,
                    ),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

        return CacheEntry(
            cache_key=cache_key,
            payload_sha256=payload_sha256,
            relative_path=relative_path_text,
        )

    async def get(self, key: CacheKey) -> dict[str, Any] | None:
        cache_key = key.digest()
        async with aiosqlite.connect(self.db_path) as connection:
            cursor = await connection.execute(
                """
                SELECT payload_sha256, relative_path
                FROM cache_index WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None

        expected_sha = str(row[0])
        payload_path = self._safe_indexed_path(str(row[1]))
        if not payload_path.is_file():
            raise CacheCorruptionError("cache payload file is missing")
        payload_bytes = payload_path.read_bytes()
        actual_sha = hashlib.sha256(payload_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise CacheCorruptionError("cache payload hash does not match index")
        try:
            payload = json.loads(payload_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CacheCorruptionError("cache payload is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise CacheCorruptionError("cache payload must be a JSON object")
        self._reject_sensitive_payload(payload)
        return payload

    @staticmethod
    def _canonical_payload(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _reject_sensitive_payload(cls, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = "".join(character for character in str(key).casefold() if character.isalnum())
                if normalized in _SENSITIVE_KEYS:
                    raise SensitiveCachePayloadError(
                        f"cache payload contains prohibited sensitive field: {key}"
                    )
                cls._reject_sensitive_payload(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                cls._reject_sensitive_payload(item)

    def _safe_indexed_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise CacheCorruptionError("cache index contains an unsafe payload path")
        return self.cache_dir / candidate

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = path.read_bytes()
            if existing != payload:
                raise CacheCorruptionError("content-addressed cache file has conflicting bytes")
            return
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
