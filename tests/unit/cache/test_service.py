from __future__ import annotations

from dataclasses import replace

import pytest

from question_builder.cache.service import (
    CacheConflictError,
    CacheKey,
    CacheService,
    SensitiveCachePayloadError,
)


def _key() -> CacheKey:
    return CacheKey(
        content_hash="a" * 64,
        provider="provider-a",
        model_version="model-v1",
        prompt_version="prompt-v1",
        recognition_task="text_ocr",
    )


@pytest.mark.asyncio
async def test_identical_versioned_cache_key_hits_and_round_trips_payload(tmp_path) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    key = _key()
    payload = {"content": "识别结果", "normalized_score": 0.99}

    stored = await service.put(key, payload)
    loaded = await service.get(key)

    assert stored.cache_key == key.digest()
    assert stored.payload_sha256
    assert loaded == payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content_hash", "b" * 64),
        ("provider", "provider-b"),
        ("model_version", "model-v2"),
        ("prompt_version", "prompt-v2"),
        ("recognition_task", "formula_ocr"),
    ],
)
async def test_any_cache_identity_version_change_is_a_miss(
    tmp_path,
    field: str,
    value: str,
) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    key = _key()
    await service.put(key, {"content": "cached"})

    changed = replace(key, **{field: value})

    assert await service.get(changed) is None


@pytest.mark.asyncio
async def test_cache_payload_is_immutable_for_an_existing_key(tmp_path) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    key = _key()

    first = await service.put(key, {"content": "stable"})
    second = await service.put(key, {"content": "stable"})
    assert second == first

    with pytest.raises(CacheConflictError):
        await service.put(key, {"content": "different"})

    assert await service.get(key) == {"content": "stable"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"headers": {"Authorization": "Bearer TOP-SECRET"}, "content": "x"},
        {"metadata": {"api_key": "TOP-SECRET"}, "content": "x"},
        {"metadata": {"access_token": "TOP-SECRET"}, "content": "x"},
    ],
)
async def test_cache_refuses_api_headers_and_secret_material(tmp_path, payload) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    key = _key()

    with pytest.raises(SensitiveCachePayloadError):
        await service.put(key, payload)

    assert await service.get(key) is None
    persisted_files = [path for path in (tmp_path / "cache").rglob("*") if path.is_file()]
    assert all(b"TOP-SECRET" not in path.read_bytes() for path in persisted_files)
