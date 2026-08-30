from __future__ import annotations

import pytest

from question_builder.config.models import QualityThresholds
from question_builder.recognition.calibration import CalibrationProfile, CalibrationRegistry
from question_builder.recognition.contracts import (
    ImageClass,
    ProviderOutput,
    RecognitionRequest,
    RecognitionTask,
)
from question_builder.recognition.providers.fake import FakeRecognitionProvider
from question_builder.recognition.router import (
    RecognitionDecision,
    RecognitionRejected,
    RecognitionRouter,
)


def _output(score: float, content: str) -> ProviderOutput:
    return ProviderOutput(
        request_id="req",
        latency_ms=1.0,
        raw_score=score,
        raw_score_reference="fake-score",
        content=content,
    )


def _provider(
    provider: str,
    model: str,
    task: RecognitionTask,
    *,
    score: float = 0.99,
    content: str = "same",
) -> FakeRecognitionProvider:
    return FakeRecognitionProvider(
        provider=provider,
        model=model,
        outputs={task: _output(score, content)},
    )


def _registry(*profiles: CalibrationProfile) -> CalibrationRegistry:
    return CalibrationRegistry(profiles=profiles)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_class", "task"),
    [
        (ImageClass.FORMULA_IMAGE, RecognitionTask.TEXT_OCR),
        (ImageClass.TABLE_IMAGE, RecognitionTask.TEXT_OCR),
        (ImageClass.TEXT_IMAGE, RecognitionTask.VISION),
    ],
)
async def test_router_rejects_image_class_task_mismatch(
    image_class: ImageClass,
    task: RecognitionTask,
) -> None:
    provider = _provider("provider-a", "model-v1", task)
    router = RecognitionRouter(
        QualityThresholds(),
        _registry(
            CalibrationProfile(
                provider="provider-a",
                model="model-v1",
                task=task,
                version="v1",
            )
        ),
    )

    with pytest.raises(RecognitionRejected, match="image_class_task_mismatch"):
        await router.recognize(
            RecognitionRequest(
                task=task,
                image_class=image_class,
                input_ref="image/a.png",
                critical=True,
            ),
            primary=provider,
        )

    assert provider.calls == 0


@pytest.mark.asyncio
async def test_question_screenshot_requires_vision_then_multimodal_llm() -> None:
    vision = _provider("vision-a", "vision-v1", RecognitionTask.VISION, content="ocr text")
    llm = _provider("llm-a", "llm-v1", RecognitionTask.LLM, content="verified content")
    router = RecognitionRouter(
        QualityThresholds(),
        _registry(
            CalibrationProfile(
                provider="vision-a",
                model="vision-v1",
                task=RecognitionTask.VISION,
                version="v1",
            ),
            CalibrationProfile(
                provider="llm-a",
                model="llm-v1",
                task=RecognitionTask.LLM,
                version="v1",
            ),
        ),
    )

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.VISION,
            image_class=ImageClass.QUESTION_SCREENSHOT,
            input_ref="image/screenshot.png",
            critical=True,
        ),
        primary=vision,
        multimodal_llm=llm,
    )

    assert result.decision is RecognitionDecision.ACCEPT
    assert result.reason == "question_screenshot_multimodal_verified"
    assert result.primary_result is not None
    assert result.primary_result.task is RecognitionTask.VISION
    assert result.result is not None
    assert result.result.task is RecognitionTask.LLM
    assert vision.calls == 1
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_same_provider_and_model_cannot_be_independent_fallback() -> None:
    primary = _provider(
        "provider-a",
        "model-v1",
        RecognitionTask.TEXT_OCR,
        score=0.95,
        content="same",
    )
    fallback = _provider(
        "provider-a",
        "model-v1",
        RecognitionTask.TEXT_OCR,
        score=0.99,
        content="same",
    )
    router = RecognitionRouter(
        QualityThresholds(),
        _registry(
            CalibrationProfile(
                provider="provider-a",
                model="model-v1",
                task=RecognitionTask.TEXT_OCR,
                version="v1",
            )
        ),
    )

    result = await router.recognize(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=primary,
        fallback=fallback,
    )

    assert result.decision is RecognitionDecision.REJECT
    assert result.reason == "fallback_not_independent"
    assert fallback.calls == 0
