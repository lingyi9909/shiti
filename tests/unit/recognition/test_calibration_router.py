from __future__ import annotations

import asyncio

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.recognition.calibration import (
    CalibrationProfile,
    CalibrationRegistry,
    MissingCalibrationError,
    normalize_score,
)
from question_builder.recognition.contracts import (
    ImageClass,
    ProviderOutput,
    RecognitionRequest,
    RecognitionTask,
)
from question_builder.recognition.providers.fake import FakeRecognitionProvider
from question_builder.recognition.router import (
    ProviderAuthenticationError,
    ProviderAuthorizationError,
    ProviderConnectionResetError,
    ProviderRateLimitError,
    ProviderSchemaError,
    ProviderServerError,
    ProviderTimeoutError,
    RecognitionDecision,
    RecognitionRejected,
    RecognitionRouter,
    RetryPolicy,
    call_with_retry,
)


def _registry() -> CalibrationRegistry:
    return CalibrationRegistry(
        profiles=(
            CalibrationProfile(
                provider="primary",
                model="model-v1",
                task=RecognitionTask.TEXT_OCR,
                version="v1",
                scale=1.0,
                offset=0.0,
            ),
            CalibrationProfile(
                provider="fallback",
                model="model-v2",
                task=RecognitionTask.TEXT_OCR,
                version="v3",
                scale=1.0,
                offset=0.0,
            ),
        )
    )


def _provider(
    name: str,
    model: str,
    *,
    score: float,
    content: str,
) -> FakeRecognitionProvider:
    return FakeRecognitionProvider(
        provider=name,
        model=model,
        outputs={
            RecognitionTask.TEXT_OCR: ProviderOutput(
                request_id=f"{name}-request",
                latency_ms=2.0,
                raw_score=score,
                raw_score_reference="fake-score",
                content=content,
            )
        },
    )


def test_missing_calibration_never_blindly_trusts_provider_score() -> None:
    registry = CalibrationRegistry()

    with pytest.raises(MissingCalibrationError):
        normalize_score(
            "unknown-provider",
            "unknown-model",
            RecognitionTask.TEXT_OCR,
            0.999,
            registry=registry,
        )


def test_normalize_score_clamps_and_records_versioned_identity() -> None:
    registry = CalibrationRegistry(
        profiles=(
            CalibrationProfile(
                provider="provider-a",
                model="model-a",
                task=RecognitionTask.TEXT_OCR,
                version="benchmark-2026-08",
                scale=1.2,
                offset=0.05,
            ),
        )
    )

    normalized = normalize_score(
        "provider-a",
        "model-a",
        RecognitionTask.TEXT_OCR,
        0.9,
        registry=registry,
    )

    assert normalized.score == 1.0
    assert normalized.calibration_id == (
        "provider-a/model-a/text_ocr@benchmark-2026-08"
    )


@pytest.mark.asyncio
async def test_critical_primary_at_or_above_098_accepts_without_fallback() -> None:
    fallback = _provider("fallback", "model-v2", score=0.99, content="secondary")
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=_provider("primary", "model-v1", score=0.98, content="primary"),
        fallback=fallback,
    )

    assert result.decision is RecognitionDecision.ACCEPT
    assert result.result is not None
    assert result.result.content == "primary"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_critical_score_between_floor_and_accept_uses_fallback() -> None:
    fallback = _provider("fallback", "model-v2", score=0.99, content="same")
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=_provider("primary", "model-v1", score=0.95, content="same"),
        fallback=fallback,
    )

    assert result.decision is RecognitionDecision.ACCEPT
    assert result.result is not None
    assert result.result.provider == "fallback"
    assert result.primary_result is not None
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_critical_score_below_floor_rejects_without_fallback() -> None:
    fallback = _provider("fallback", "model-v2", score=0.99, content="same")
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=_provider("primary", "model-v1", score=0.899, content="same"),
        fallback=fallback,
    )

    assert result.decision is RecognitionDecision.REJECT
    assert result.reason == "below_recognition_floor"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_conflicting_high_score_fallback_results_reject_instead_of_picking_top1() -> None:
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=_provider("primary", "model-v1", score=0.95, content="answer A"),
        fallback=_provider("fallback", "model-v2", score=0.995, content="answer B"),
    )

    assert result.decision is RecognitionDecision.REJECT
    assert result.reason == "conflicting_recognition_results"
    assert result.result is None


@pytest.mark.asyncio
async def test_noncritical_content_accepts_at_095() -> None:
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=False,
        ),
        primary=_provider("primary", "model-v1", score=0.95, content="text"),
    )

    assert result.decision is RecognitionDecision.ACCEPT
    assert result.result is not None


@pytest.mark.parametrize(
    ("image_class", "expected_task"),
    [
        (ImageClass.TEXT_IMAGE, RecognitionTask.TEXT_OCR),
        (ImageClass.FORMULA_IMAGE, RecognitionTask.FORMULA_OCR),
        (ImageClass.TABLE_IMAGE, RecognitionTask.TABLE_RECOGNITION),
        (ImageClass.QUESTION_SCREENSHOT, RecognitionTask.VISION),
        (ImageClass.DIAGRAM, RecognitionTask.VISION),
        (ImageClass.GEOMETRY, RecognitionTask.VISION),
        (ImageClass.CHART, RecognitionTask.VISION),
        (ImageClass.MAP, RecognitionTask.VISION),
        (ImageClass.CHEMISTRY, RecognitionTask.VISION),
    ],
)
def test_image_class_routes_to_expected_recognition_task(
    image_class: ImageClass,
    expected_task: RecognitionTask,
) -> None:
    assert RecognitionRouter.task_for_image_class(image_class) is expected_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProviderTimeoutError("timeout"),
        ProviderRateLimitError("429"),
        ProviderServerError("503"),
        ProviderConnectionResetError("reset"),
    ],
)
async def test_retryable_provider_errors_retry_until_success(error: Exception) -> None:
    attempts = 0
    sleeps: list[float] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise error
        return "ok"

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        await asyncio.sleep(0)

    result = await call_with_retry(
        operation,
        RetryPolicy(max_attempts=3, base_delay_seconds=0.1, jitter_seconds=0.0),
        sleep=fake_sleep,
    )

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProviderAuthenticationError("401"),
        ProviderAuthorizationError("403"),
        ProviderSchemaError("bad schema"),
    ],
)
async def test_nonretryable_provider_errors_fail_fast(error: Exception) -> None:
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(type(error)):
        await call_with_retry(
            operation,
            RetryPolicy(max_attempts=5, base_delay_seconds=0.0, jitter_seconds=0.0),
        )

    assert attempts == 1


@pytest.mark.asyncio
async def test_router_rejects_when_fallback_required_but_unavailable() -> None:
    router = RecognitionRouter(QualityThresholds(), _registry())

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=_provider("primary", "model-v1", score=0.95, content="text"),
    )

    assert result.decision is RecognitionDecision.REJECT
    assert result.reason == "fallback_required"


@pytest.mark.asyncio
async def test_provider_contract_failure_is_not_converted_to_acceptance() -> None:
    router = RecognitionRouter(QualityThresholds(), _registry())
    provider = _provider("primary", "model-v1", score=0.99, content="text")
    provider.error = ProviderSchemaError("invalid provider payload")

    with pytest.raises(RecognitionRejected, match="provider_contract_error"):
        await router.recognize(
            RecognitionRequest(
                task=RecognitionTask.TEXT_OCR,
                image_class=ImageClass.TEXT_IMAGE,
                input_ref="image/a.png",
                critical=True,
            ),
            primary=provider,
        )
