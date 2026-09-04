from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CacheConflictError(RuntimeError):
    pass


class SensitiveCachePayloadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CacheKey:
    content_hash: str
    provider: str
    model_version: str
    prompt_version: str
    recognition_task: str

    def digest(self) -> str:
        raise NotImplementedError


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
        raise NotImplementedError

    async def put(self, key: CacheKey, payload: dict[str, Any]) -> CacheEntry:
        raise NotImplementedError

    async def get(self, key: CacheKey) -> dict[str, Any] | None:
        raise NotImplementedError
