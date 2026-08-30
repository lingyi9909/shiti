from __future__ import annotations

import inspect

import pytest

from question_builder.recognition.contracts import (
    FormulaOCRProvider,
    ImageClass,
    LLMProvider,
    ProviderOutput,
    RecognitionRequest,
    RecognitionResult,
    RecognitionTask,
    TableRecognitionProvider,
    TextOCRProvider,
    VisionProvider,
)
from question_builder.recognition.providers.fake import FakeRecognitionProvider


def test_recognition_result_requires_complete_normalized_provenance() -> None:
    result = RecognitionResult(
        task=RecognitionTask.TEXT_OCR,
        image_class=ImageClass.TEXT_IMAGE,
        provider="fake-primary",
        model="fake-v1",
        request_id="req-001",
        latency_ms=12.5,
        raw_score=0.97,
        raw_score_reference="provider_confidence",
        normalized_score=0.96,
        calibration_id="fake-primary/fake-v1/text_ocr@v1",
        content="recognized text",
    )

    assert result.provider == "fake-primary"
    assert result.model == "fake-v1"
    assert result.request_id == "req-001"
    assert result.latency_ms == 12.5
    assert result.raw_score == 0.97
    assert result.raw_score_reference == "provider_confidence"
    assert result.normalized_score == 0.96
    assert result.calibration_id.endswith("@v1")
    assert result.content == "recognized text"


@pytest.mark.parametrize(
    "missing_field",
    [
        "provider",
        "model",
        "request_id",
        "latency_ms",
        "raw_score_reference",
        "normalized_score",
        "calibration_id",
        "content",
    ],
)
def test_recognition_result_rejects_missing_required_contract_fields(
    missing_field: str,
) -> None:
    payload = {
        "task": RecognitionTask.TEXT_OCR,
        "image_class": ImageClass.TEXT_IMAGE,
        "provider": "fake-primary",
        "model": "fake-v1",
        "request_id": "req-001",
        "latency_ms": 12.5,
        "raw_score": 0.97,
        "raw_score_reference": "provider_confidence",
        "normalized_score": 0.96,
        "calibration_id": "fake-primary/fake-v1/text_ocr@v1",
        "content": "recognized text",
    }
    payload.pop(missing_field)

    with pytest.raises((TypeError, ValueError)):
        RecognitionResult(**payload)


def test_all_provider_contracts_are_runtime_checkable_async_protocols() -> None:
    provider = FakeRecognitionProvider(
        provider="fake-primary",
        model="fake-v1",
        outputs={
            RecognitionTask.TEXT_OCR: ProviderOutput(
                request_id="text-1",
                latency_ms=1.0,
                raw_score=0.99,
                raw_score_reference="fake-score",
                content="text",
            ),
            RecognitionTask.FORMULA_OCR: ProviderOutput(
                request_id="formula-1",
                latency_ms=1.0,
                raw_score=0.99,
                raw_score_reference="fake-score",
                content="x^2",
            ),
            RecognitionTask.TABLE_RECOGNITION: ProviderOutput(
                request_id="table-1",
                latency_ms=1.0,
                raw_score=0.99,
                raw_score_reference="fake-score",
                content="|a|",
            ),
            RecognitionTask.VISION: ProviderOutput(
                request_id="vision-1",
                latency_ms=1.0,
                raw_score=0.99,
                raw_score_reference="fake-score",
                content="diagram",
            ),
            RecognitionTask.LLM: ProviderOutput(
                request_id="llm-1",
                latency_ms=1.0,
                raw_score=0.99,
                raw_score_reference="fake-score",
                content="classification",
            ),
        },
    )

    assert isinstance(provider, TextOCRProvider)
    assert isinstance(provider, FormulaOCRProvider)
    assert isinstance(provider, TableRecognitionProvider)
    assert isinstance(provider, VisionProvider)
    assert isinstance(provider, LLMProvider)
    assert inspect.iscoroutinefunction(provider.recognize_text)
    assert inspect.iscoroutinefunction(provider.recognize_formula)
    assert inspect.iscoroutinefunction(provider.recognize_table)
    assert inspect.iscoroutinefunction(provider.recognize_vision)
    assert inspect.iscoroutinefunction(provider.complete)


@pytest.mark.asyncio
async def test_fake_provider_returns_contract_output_not_vendor_dto() -> None:
    output = ProviderOutput(
        request_id="req-1",
        latency_ms=3.0,
        raw_score=0.93,
        raw_score_reference="fake-score",
        content="hello",
    )
    provider = FakeRecognitionProvider(
        provider="fake-primary",
        model="fake-v1",
        outputs={RecognitionTask.TEXT_OCR: output},
    )

    response = await provider.recognize_text(
        RecognitionRequest(
            task=RecognitionTask.TEXT_OCR,
            image_class=ImageClass.TEXT_IMAGE,
            input_ref="image/img.png",
        )
    )

    assert response == output
    assert type(response) is ProviderOutput
