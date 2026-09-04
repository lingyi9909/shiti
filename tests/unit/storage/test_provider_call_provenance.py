from __future__ import annotations

import pytest

from question_builder.storage.db import ProviderCallRecord, RunStore


@pytest.mark.asyncio
async def test_provider_call_round_trip_preserves_raw_and_calibration_provenance(tmp_path) -> None:
    store = RunStore(tmp_path / "state.sqlite3")
    await store.initialize()
    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )

    await store.record_provider_call(
        ProviderCallRecord(
            run_id=run.run_id,
            stage="recognition",
            provider="provider-a",
            model="model-v1",
            task="text_ocr",
            request_id="req-1",
            prompt_version="ocr-v1",
            content_hash="c" * 64,
            latency_ms=12.5,
            raw_score=0.96,
            raw_score_reference="provider_confidence",
            normalized_score=0.94,
            calibration_id="provider-a/model-v1/text_ocr@v2",
            cache_hit=True,
            fallback_reason=None,
            token_usage=321,
        )
    )

    calls = await store.list_provider_calls(run.run_id)

    assert len(calls) == 1
    assert calls[0].raw_score == pytest.approx(0.96)
    assert calls[0].raw_score_reference == "provider_confidence"
    assert calls[0].normalized_score == pytest.approx(0.94)
    assert calls[0].calibration_id == "provider-a/model-v1/text_ocr@v2"
    assert calls[0].model == "model-v1"
    assert calls[0].token_usage == 321
