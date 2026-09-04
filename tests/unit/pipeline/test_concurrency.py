from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from question_builder.config.models import ConcurrencyConfig
from question_builder.pipeline.orchestrator import PipelineConcurrency, WorkloadKind


@pytest.mark.asyncio
async def test_each_workload_uses_its_own_configured_concurrency_limit() -> None:
    config = ConcurrencyConfig(
        document_parse=2,
        text_ocr=3,
        formula_ocr=2,
        table=2,
        vision=2,
        llm=2,
    )
    limiter = PipelineConcurrency(config)
    active: dict[WorkloadKind, int] = defaultdict(int)
    peak: dict[WorkloadKind, int] = defaultdict(int)
    lock = asyncio.Lock()

    async def work(kind: WorkloadKind, item: int) -> int:
        async with lock:
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])
        await asyncio.sleep(0.01)
        async with lock:
            active[kind] -= 1
        return item * 2

    expected_limits = {
        WorkloadKind.DOCUMENT_PARSE: 2,
        WorkloadKind.TEXT_OCR: 3,
        WorkloadKind.FORMULA_OCR: 2,
        WorkloadKind.TABLE: 2,
        WorkloadKind.VISION: 2,
        WorkloadKind.LLM: 2,
    }
    for kind, expected_limit in expected_limits.items():
        results = await limiter.map_bounded(
            kind,
            range(12),
            lambda item, kind=kind: work(kind, item),
        )
        assert results == [item * 2 for item in range(12)]
        assert peak[kind] == expected_limit


@pytest.mark.asyncio
async def test_map_bounded_creates_only_limit_sized_worker_pool() -> None:
    limiter = PipelineConcurrency(
        ConcurrencyConfig(
            document_parse=2,
            text_ocr=2,
            formula_ocr=2,
            table=2,
            vision=2,
            llm=2,
        )
    )
    created_workers = 0

    async def worker(item: int) -> int:
        nonlocal created_workers
        task = asyncio.current_task()
        assert task is not None
        marker = getattr(task, "_qbuilder_counted", False)
        if not marker:
            setattr(task, "_qbuilder_counted", True)
            created_workers += 1
        await asyncio.sleep(0)
        return item

    values = await limiter.map_bounded(WorkloadKind.LLM, range(1000), worker)

    assert values == list(range(1000))
    assert created_workers <= 2
