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
from question_builder.recognition.router import RecognitionDecision, RecognitionRouter


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
    content: str = "content",
) -> FakeRecognitionProvider:
    return FakeRecognitionProvider(
        provider=provider,
        model=model,
        outputs={task: _output(score, content)},
    )


def _router() -> RecognitionRouter:
    return RecognitionRouter(
        QualityThresholds(),
        CalibrationRegistry(
            profiles=(
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
            )
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("image_class", [ImageClass.MIXED, ImageClass.UNKNOWN])
async def test_mixed_unknown_reject_without_multimodal_provider(
    image_class: ImageClass,
) -> None:
    vision = _provider("vision-a", "vision-v1", RecognitionTask.VISION)

    result = await _router().recognize(
        RecognitionRequest(
            task=RecognitionTask.VISION,
            image_class=image_class,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=vision,
    )

    assert result.decision is RecognitionDecision.REJECT
    assert result.reason == "multimodal_llm_required"
    assert vision.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("image_class", [ImageClass.MIXED, ImageClass.UNKNOWN])
async def test_mixed_unknown_require_verified_multimodal_path(
    image_class: ImageClass,
) -> None:
    vision = _provider("vision-a", "vision-v1", RecognitionTask.VISION)
    llm = _provider("llm-a", "llm-v1", RecognitionTask.LLM)

    result = await _router().recognize(
        RecognitionRequest(
            task=RecognitionTask.VISION,
            image_class=image_class,
            input_ref="image/a.png",
            critical=True,
        ),
        primary=vision,
        multimodal_llm=llm,
    )

    assert result.decision is RecognitionDecision.ACCEPT
    assert result.reason == "multimodal_fallback_verified"
    assert result.primary_result is not None
    assert result.primary_result.task is RecognitionTask.VISION
    assert result.result is not None
    assert result.result.task is RecognitionTask.LLM
    assert vision.calls == 1
    assert llm.calls == 1
