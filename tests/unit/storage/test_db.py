from __future__ import annotations

import sqlite3

import pytest

from question_builder.storage.db import (
    InvalidStageTransitionError,
    ProviderCallRecord,
    RunStore,
    StageState,
    compute_run_fingerprint,
)


@pytest.mark.asyncio
async def test_run_store_persists_required_schema_and_versioned_run_metadata(tmp_path) -> None:
    db_path = tmp_path / "state.sqlite3"
    store = RunStore(db_path)
    await store.initialize()

    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )

    assert run.run_id == compute_run_fingerprint(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    assert run.input_hash == "a" * 64
    assert run.config_hash == "b" * 64
    assert run.pipeline_version == "0.1.0"

    stage = await store.ensure_stage(run.run_id, "parse", subject_id="doc-1")
    assert stage.state is StageState.PENDING

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
            normalized_score=0.99,
            calibration_id="provider-a/model-v1/text_ocr@v1",
            cache_hit=False,
            fallback_reason=None,
            token_usage=321,
        )
    )
    calls = await store.list_provider_calls(run.run_id)
    assert len(calls) == 1
    assert calls[0].provider == "provider-a"
    assert calls[0].model == "model-v1"
    assert calls[0].prompt_version == "ocr-v1"
    assert calls[0].content_hash == "c" * 64
    assert calls[0].raw_score == pytest.approx(0.96)
    assert calls[0].raw_score_reference == "provider_confidence"
    assert calls[0].calibration_id == "provider-a/model-v1/text_ocr@v1"
    assert calls[0].token_usage == 321

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"runs", "stage_states", "provider_calls", "cache_index"} <= tables


@pytest.mark.asyncio
async def test_stage_state_transitions_are_validated_and_terminal_states_are_immutable(
    tmp_path,
) -> None:
    store = RunStore(tmp_path / "state.sqlite3")
    await store.initialize()
    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    await store.ensure_stage(run.run_id, "parse", subject_id="doc-1")

    running = await store.transition_stage(
        run.run_id,
        "parse",
        StageState.RUNNING,
        subject_id="doc-1",
    )
    completed = await store.transition_stage(
        run.run_id,
        "parse",
        StageState.COMPLETED,
        subject_id="doc-1",
    )

    assert running.state is StageState.RUNNING
    assert completed.state is StageState.COMPLETED

    with pytest.raises(InvalidStageTransitionError):
        await store.transition_stage(
            run.run_id,
            "parse",
            StageState.RUNNING,
            subject_id="doc-1",
        )


@pytest.mark.asyncio
async def test_failed_stage_can_be_retried_but_rejected_stage_is_terminal(tmp_path) -> None:
    store = RunStore(tmp_path / "state.sqlite3")
    await store.initialize()
    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )

    await store.ensure_stage(run.run_id, "recognition", subject_id="img-1")
    await store.transition_stage(
        run.run_id,
        "recognition",
        StageState.RUNNING,
        subject_id="img-1",
    )
    await store.transition_stage(
        run.run_id,
        "recognition",
        StageState.FAILED,
        subject_id="img-1",
        error="timeout",
    )
    retried = await store.transition_stage(
        run.run_id,
        "recognition",
        StageState.RUNNING,
        subject_id="img-1",
    )
    assert retried.state is StageState.RUNNING

    await store.transition_stage(
        run.run_id,
        "recognition",
        StageState.REJECTED,
        subject_id="img-1",
        error="low confidence",
    )
    with pytest.raises(InvalidStageTransitionError):
        await store.transition_stage(
            run.run_id,
            "recognition",
            StageState.RUNNING,
            subject_id="img-1",
        )


@pytest.mark.asyncio
async def test_resume_starts_at_first_incomplete_stage_and_skips_completed_paid_work(
    tmp_path,
) -> None:
    store = RunStore(tmp_path / "state.sqlite3")
    await store.initialize()
    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    stages = ("parse", "recognition", "split", "answer")
    for stage in stages:
        await store.ensure_stage(run.run_id, stage)

    for stage in ("parse", "recognition"):
        await store.transition_stage(run.run_id, stage, StageState.RUNNING)
        await store.transition_stage(run.run_id, stage, StageState.COMPLETED)

    first = await store.first_incomplete_stage(run.run_id, stages)
    assert first == "split"

    calls = {stage: 0 for stage in stages}
    assert first is not None
    for stage in stages[stages.index(first) :]:
        calls[stage] += 1

    assert calls == {"parse": 0, "recognition": 0, "split": 1, "answer": 1}


@pytest.mark.asyncio
async def test_rejected_stage_terminates_resume_without_running_downstream(tmp_path) -> None:
    store = RunStore(tmp_path / "state.sqlite3")
    await store.initialize()
    run = await store.ensure_run(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    stages = ("parse", "recognition", "split", "answer")
    for stage in stages:
        await store.ensure_stage(run.run_id, stage)

    await store.transition_stage(run.run_id, "parse", StageState.RUNNING)
    await store.transition_stage(run.run_id, "parse", StageState.COMPLETED)
    await store.transition_stage(run.run_id, "recognition", StageState.RUNNING)
    await store.transition_stage(run.run_id, "recognition", StageState.REJECTED)

    assert await store.first_incomplete_stage(run.run_id, stages) is None


def test_run_fingerprint_is_stable_and_changes_with_any_versioned_input() -> None:
    baseline = compute_run_fingerprint(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )

    assert baseline == compute_run_fingerprint(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    assert baseline != compute_run_fingerprint(
        input_hash="c" * 64,
        config_hash="b" * 64,
        pipeline_version="0.1.0",
    )
    assert baseline != compute_run_fingerprint(
        input_hash="a" * 64,
        config_hash="d" * 64,
        pipeline_version="0.1.0",
    )
    assert baseline != compute_run_fingerprint(
        input_hash="a" * 64,
        config_hash="b" * 64,
        pipeline_version="0.2.0",
    )
