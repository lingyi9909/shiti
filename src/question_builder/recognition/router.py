from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from question_builder.config.models import QualityThresholds
from question_builder.recognition.calibration import (
    CalibrationRegistry,
    MissingCalibrationError,
    normalize_score,
)
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

Provider = (
    TextOCRProvider
    | FormulaOCRProvider
    | TableRecognitionProvider
    | VisionProvider
    | LLMProvider
)


class ProviderError(RuntimeError):
    pass


class ProviderTimeoutError(ProviderError):
    pass


class ProviderRateLimitError(ProviderError):
    pass


class ProviderServerError(ProviderError):
    pass


class ProviderConnectionResetError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderAuthorizationError(ProviderError):
    pass


class ProviderSchemaError(ProviderError):
    pass


class ProviderHTTPError(ProviderError):
    def __init__(self, status_code: int, message: str = "provider HTTP error") -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code


class RecognitionRejected(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    jitter_seconds: float = 0.10

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0 or self.jitter_seconds < 0:
            raise ValueError("retry delays must be non-negative")


def is_retryable_error(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderServerError,
            ProviderConnectionResetError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionResetError,
        ),
    ):
        return True
    if isinstance(error, ProviderHTTPError):
        return error.status_code == 429 or 500 <= error.status_code <= 599
    return False


async def call_with_retry[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_fn: Callable[[], float] = random.random,
) -> T:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except BaseException as exc:
            if not is_retryable_error(exc) or attempt >= policy.max_attempts:
                raise
            backoff = policy.base_delay_seconds * (2 ** (attempt - 1))
            jitter = policy.jitter_seconds * random_fn()
            await sleep(backoff + jitter)
    raise AssertionError("retry loop exhausted unexpectedly")


class RecognitionDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class RecognitionRoutingResult:
    decision: RecognitionDecision
    reason: str
    result: RecognitionResult | None = None
    primary_result: RecognitionResult | None = None
    fallback_result: RecognitionResult | None = None


class RecognitionRouter:
    def __init__(
        self,
        thresholds: QualityThresholds,
        calibrations: CalibrationRegistry,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._thresholds = thresholds
        self._calibrations = calibrations
        self._retry_policy = retry_policy or RetryPolicy()

    @staticmethod
    def task_for_image_class(image_class: ImageClass) -> RecognitionTask:
        if image_class is ImageClass.TEXT_IMAGE:
            return RecognitionTask.TEXT_OCR
        if image_class is ImageClass.FORMULA_IMAGE:
            return RecognitionTask.FORMULA_OCR
        if image_class is ImageClass.TABLE_IMAGE:
            return RecognitionTask.TABLE_RECOGNITION
        return RecognitionTask.VISION

    async def recognize(
        self,
        request: RecognitionRequest,
        *,
        primary: Provider,
        fallback: Provider | None = None,
        multimodal_llm: LLMProvider | None = None,
    ) -> RecognitionRoutingResult:
        expected_task = self.task_for_image_class(request.image_class)
        if request.task is not expected_task:
            raise RecognitionRejected(
                "image_class_task_mismatch: "
                f"{request.image_class.value} requires {expected_task.value}, "
                f"got {request.task.value}"
            )

        try:
            primary_result = await self._call_provider(primary, request)
        except ProviderSchemaError as exc:
            raise RecognitionRejected(f"provider_contract_error: {exc}") from exc
        except MissingCalibrationError:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="missing_calibration",
            )

        accept_threshold = (
            self._thresholds.critical_recognition_accept
            if request.critical
            else self._thresholds.noncritical_recognition_accept
        )
        floor = self._thresholds.recognition_fallback_floor

        if primary_result.normalized_score < floor:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="below_recognition_floor",
                primary_result=primary_result,
            )

        if primary_result.normalized_score >= accept_threshold:
            if request.image_class is ImageClass.QUESTION_SCREENSHOT:
                return await self._verify_question_screenshot(
                    request,
                    primary_result,
                    multimodal_llm,
                )
            if request.image_class in (ImageClass.MIXED, ImageClass.UNKNOWN):
                return await self._verify_multimodal_fallback(
                    request,
                    primary_result,
                    multimodal_llm,
                )
            return RecognitionRoutingResult(
                decision=RecognitionDecision.ACCEPT,
                reason="primary_accept",
                result=primary_result,
                primary_result=primary_result,
            )

        if fallback is None:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="fallback_required",
                primary_result=primary_result,
            )

        if (primary.provider, primary.model) == (fallback.provider, fallback.model):
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="fallback_not_independent",
                primary_result=primary_result,
            )

        try:
            fallback_result = await self._call_provider(fallback, request)
        except ProviderSchemaError as exc:
            raise RecognitionRejected(f"provider_contract_error: {exc}") from exc
        except MissingCalibrationError:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="missing_calibration",
                primary_result=primary_result,
            )

        if fallback_result.normalized_score < floor:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="fallback_below_recognition_floor",
                primary_result=primary_result,
                fallback_result=fallback_result,
            )

        if fallback_result.normalized_score < accept_threshold:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="recognition_not_verified",
                primary_result=primary_result,
                fallback_result=fallback_result,
            )

        if self._canonical_content(primary_result.content) != self._canonical_content(
            fallback_result.content
        ):
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="conflicting_recognition_results",
                primary_result=primary_result,
                fallback_result=fallback_result,
            )

        if request.image_class is ImageClass.QUESTION_SCREENSHOT:
            return await self._verify_question_screenshot(
                request,
                fallback_result,
                multimodal_llm,
                primary_result=primary_result,
            )
        if request.image_class in (ImageClass.MIXED, ImageClass.UNKNOWN):
            return await self._verify_multimodal_fallback(
                request,
                fallback_result,
                multimodal_llm,
                primary_result=primary_result,
            )

        return RecognitionRoutingResult(
            decision=RecognitionDecision.ACCEPT,
            reason="fallback_verified",
            result=fallback_result,
            primary_result=primary_result,
            fallback_result=fallback_result,
        )

    async def _verify_question_screenshot(
        self,
        request: RecognitionRequest,
        vision_result: RecognitionResult,
        multimodal_llm: LLMProvider | None,
        *,
        primary_result: RecognitionResult | None = None,
    ) -> RecognitionRoutingResult:
        if multimodal_llm is None:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="multimodal_llm_required",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        llm_request = RecognitionRequest(
            task=RecognitionTask.LLM,
            image_class=request.image_class,
            input_ref=request.input_ref,
            critical=request.critical,
        )
        try:
            llm_result = await self._call_provider(multimodal_llm, llm_request)
        except ProviderSchemaError as exc:
            raise RecognitionRejected(f"provider_contract_error: {exc}") from exc
        except MissingCalibrationError:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="missing_calibration",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        accept_threshold = (
            self._thresholds.critical_recognition_accept
            if request.critical
            else self._thresholds.noncritical_recognition_accept
        )
        if llm_result.normalized_score < accept_threshold:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="multimodal_llm_not_verified",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        return RecognitionRoutingResult(
            decision=RecognitionDecision.ACCEPT,
            reason="question_screenshot_multimodal_verified",
            result=llm_result,
            primary_result=primary_result or vision_result,
            fallback_result=vision_result if primary_result is not None else None,
        )

    async def _verify_multimodal_fallback(
        self,
        request: RecognitionRequest,
        vision_result: RecognitionResult,
        multimodal_llm: LLMProvider | None,
        *,
        primary_result: RecognitionResult | None = None,
    ) -> RecognitionRoutingResult:
        if multimodal_llm is None:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="multimodal_llm_required",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        llm_request = RecognitionRequest(
            task=RecognitionTask.LLM,
            image_class=request.image_class,
            input_ref=request.input_ref,
            critical=request.critical,
        )
        try:
            llm_result = await self._call_provider(multimodal_llm, llm_request)
        except ProviderSchemaError as exc:
            raise RecognitionRejected(f"provider_contract_error: {exc}") from exc
        except MissingCalibrationError:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="missing_calibration",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        accept_threshold = (
            self._thresholds.critical_recognition_accept
            if request.critical
            else self._thresholds.noncritical_recognition_accept
        )
        if llm_result.normalized_score < accept_threshold:
            return RecognitionRoutingResult(
                decision=RecognitionDecision.REJECT,
                reason="multimodal_llm_not_verified",
                primary_result=primary_result or vision_result,
                fallback_result=vision_result if primary_result is not None else None,
            )

        return RecognitionRoutingResult(
            decision=RecognitionDecision.ACCEPT,
            reason="multimodal_fallback_verified",
            result=llm_result,
            primary_result=primary_result or vision_result,
            fallback_result=vision_result if primary_result is not None else None,
        )

    async def _call_provider(
        self,
        provider: Provider,
        request: RecognitionRequest,
    ) -> RecognitionResult:
        output = await call_with_retry(
            lambda: self._invoke(provider, request),
            self._retry_policy,
        )
        if not isinstance(output, ProviderOutput):
            raise ProviderSchemaError("provider returned non-contract output")
        normalized = normalize_score(
            provider.provider,
            provider.model,
            request.task,
            output.raw_score,
            registry=self._calibrations,
        )
        return RecognitionResult(
            task=request.task,
            image_class=request.image_class,
            provider=provider.provider,
            model=provider.model,
            request_id=output.request_id,
            latency_ms=output.latency_ms,
            raw_score=output.raw_score,
            raw_score_reference=output.raw_score_reference,
            normalized_score=normalized.score,
            calibration_id=normalized.calibration_id,
            content=output.content,
        )

    @staticmethod
    async def _invoke(provider: Provider, request: RecognitionRequest) -> ProviderOutput:
        if request.task is RecognitionTask.TEXT_OCR:
            method = getattr(provider, "recognize_text", None)
        elif request.task is RecognitionTask.FORMULA_OCR:
            method = getattr(provider, "recognize_formula", None)
        elif request.task is RecognitionTask.TABLE_RECOGNITION:
            method = getattr(provider, "recognize_table", None)
        elif request.task is RecognitionTask.VISION:
            method = getattr(provider, "recognize_vision", None)
        else:
            method = getattr(provider, "complete", None)
        if method is None or not callable(method):
            raise ProviderSchemaError(
                f"provider does not implement task {request.task.value}"
            )
        return cast(ProviderOutput, await method(request))

    @staticmethod
    def _canonical_content(content: str) -> str:
        return " ".join(content.split())
