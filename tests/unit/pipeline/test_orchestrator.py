from __future__ import annotations

from collections import Counter
from pathlib import Path

import aiosqlite
import pytest

from question_builder.pipeline.orchestrator import (
    PIPELINE_STAGES,
    PipelineContext,
    PipelineStageError,
    QuestionBuilderPipeline,
    RunStatus,
)


def _write_config(path: Path) -> None:
    path.write_text("{}\n", encoding="utf-8")


async def _stage_state(context: PipelineContext, stage: str) -> str | None:
    async with aiosqlite.connect(context.run_store.db_path) as connection:
        cursor = await connection.execute(
            """
            SELECT state FROM stage_states
            WHERE run_id = ? AND stage = ? AND subject_id = '__run__'
            """,
            (context.run_id, stage),
        )
        row = await cursor.fetchone()
    return None if row is None else str(row[0])


@pytest.mark.asyncio
async def test_run_executes_exact_stage_order_and_checkpoints_before_downstream(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.docx").write_bytes(b"synthetic-docx")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    observed: list[str] = []
    handlers = {}
    for index, stage in enumerate(PIPELINE_STAGES):

        async def handler(
            context: PipelineContext,
            *,
            stage: str = stage,
            index: int = index,
        ) -> None:
            assert await _stage_state(context, stage) == "RUNNING"
            if index:
                assert await _stage_state(context, PIPELINE_STAGES[index - 1]) == "COMPLETED"
            observed.append(stage)

        handlers[stage] = handler

    pipeline = QuestionBuilderPipeline(
        state_root=tmp_path / "state",
        stage_handlers=handlers,
    )
    summary = await pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
    )

    assert observed == list(PIPELINE_STAGES)
    assert summary.status is RunStatus.COMPLETED
    assert summary.completed_stages == PIPELINE_STAGES
    assert summary.next_stage is None
    for stage in PIPELINE_STAGES:
        assert await _stage_state(pipeline.context_for(summary.run_id), stage) == "COMPLETED"


@pytest.mark.asyncio
async def test_resume_starts_at_failed_stage_without_repeating_completed_work(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.docx").write_bytes(b"synthetic-docx")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    calls: Counter[str] = Counter()

    async def successful(context: PipelineContext) -> None:
        calls[context.current_stage] += 1

    async def fail_recognition(context: PipelineContext) -> None:
        calls[context.current_stage] += 1
        raise RuntimeError("recognition interrupted")

    first_handlers = {stage: successful for stage in PIPELINE_STAGES}
    first_handlers["recognition"] = fail_recognition
    first = QuestionBuilderPipeline(
        state_root=tmp_path / "state",
        stage_handlers=first_handlers,
    )

    with pytest.raises(PipelineStageError) as exc_info:
        await first.run(
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
        )

    run_id = exc_info.value.run_id
    assert exc_info.value.stage == "recognition"
    assert calls["ingest"] == 1
    assert calls["parse"] == 1
    assert calls["recognition"] == 1
    assert calls["understanding"] == 0

    second = QuestionBuilderPipeline(
        state_root=tmp_path / "state",
        stage_handlers={stage: successful for stage in PIPELINE_STAGES},
    )
    summary = await second.resume(run_id)

    assert summary.status is RunStatus.COMPLETED
    assert calls["ingest"] == 1
    assert calls["parse"] == 1
    assert calls["recognition"] == 2
    assert calls["understanding"] == 1
    assert calls["export_report"] == 1


@pytest.mark.asyncio
async def test_run_fails_closed_when_a_required_stage_handler_is_missing(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.docx").write_bytes(b"synthetic-docx")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    pipeline = QuestionBuilderPipeline(
        state_root=tmp_path / "state",
        stage_handlers={},
    )

    with pytest.raises(PipelineStageError, match="stage handler is not configured") as exc_info:
        await pipeline.run(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            config_path=config_path,
        )

    assert exc_info.value.stage == "ingest"
