from __future__ import annotations

import sqlite3

import pytest

from question_builder.cache.service import CacheKey, CacheService, SensitiveCachePayloadError
from question_builder.recognition.calibration import (
    CalibrationProfile,
    CalibrationRegistry,
    normalize_score,
)
from question_builder.recognition.contracts import ProviderOutput, RecognitionTask


def _key() -> CacheKey:
    return CacheKey(
        content_hash="a" * 64,
        provider="provider-a",
        model_version="model-v1",
        prompt_version="prompt-v1",
        recognition_task="text_ocr",
    )


def _raw_output() -> ProviderOutput:
    return ProviderOutput(
        request_id="req-raw-1",
        latency_ms=12.5,
        raw_score=0.96,
        raw_score_reference="provider_confidence",
        content="识别结果",
    )


@pytest.mark.asyncio
async def test_cached_provider_raw_output_is_renormalized_by_current_calibration(tmp_path) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    key = _key()

    await service.put(key, _raw_output())
    cached = await service.get(key)

    assert cached == _raw_output()

    v1 = CalibrationRegistry(
        (
            CalibrationProfile(
                provider="provider-a",
                model="model-v1",
                task=RecognitionTask.TEXT_OCR,
                version="v1",
                offset=0.03,
            ),
        )
    )
    v2 = CalibrationRegistry(
        (
            CalibrationProfile(
                provider="provider-a",
                model="model-v1",
                task=RecognitionTask.TEXT_OCR,
                version="v2",
                offset=-0.02,
            ),
        )
    )

    normalized_v1 = normalize_score(
        "provider-a",
        "model-v1",
        RecognitionTask.TEXT_OCR,
        cached.raw_score,
        registry=v1,
    )
    normalized_v2 = normalize_score(
        "provider-a",
        "model-v1",
        RecognitionTask.TEXT_OCR,
        cached.raw_score,
        registry=v2,
    )

    assert normalized_v1.score == pytest.approx(0.99)
    assert normalized_v1.calibration_id == "provider-a/model-v1/text_ocr@v1"
    assert normalized_v2.score == pytest.approx(0.94)
    assert normalized_v2.calibration_id == "provider-a/model-v1/text_ocr@v2"


@pytest.mark.asyncio
async def test_cache_rejects_post_calibration_score_as_persistent_provider_payload(
    tmp_path,
) -> None:
    service = CacheService(
        db_path=tmp_path / "state.sqlite3",
        cache_dir=tmp_path / "cache",
    )
    await service.initialize()
    payload = {
        "request_id": "req-raw-1",
        "latency_ms": 12.5,
        "raw_score": 0.96,
        "raw_score_reference": "provider_confidence",
        "content": "识别结果",
        "normalized_score": 0.99,
        "calibration_id": "provider-a/model-v1/text_ocr@v1",
    }

    with pytest.raises(ValueError):
        await service.put(_key(), payload)

    assert await service.get(_key()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_key",
    ["X-API-Key", "Proxy-Authorization", "X-Access-Token"],
)
async def test_cache_secret_structure_is_fail_closed_and_leaves_no_artifact(
    tmp_path,
    secret_key: str,
) -> None:
    db_path = tmp_path / "state.sqlite3"
    cache_dir = tmp_path / "cache"
    service = CacheService(db_path=db_path, cache_dir=cache_dir)
    await service.initialize()

    with pytest.raises(SensitiveCachePayloadError):
        await service.put(
            _key(),
            {
                "request_id": "req-secret",
                "latency_ms": 1.0,
                "raw_score": 0.5,
                "raw_score_reference": "provider_confidence",
                "content": "x",
                "metadata": {secret_key: "TOP-SECRET"},
            },
        )

    with sqlite3.connect(db_path) as connection:
        cache_rows = connection.execute("SELECT COUNT(*) FROM cache_index").fetchone()
    assert cache_rows == (0,)

    persisted_files = [path for path in cache_dir.rglob("*") if path.is_file()]
    assert persisted_files == []
