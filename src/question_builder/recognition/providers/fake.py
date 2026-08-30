from __future__ import annotations

from question_builder.recognition.contracts import (
    ProviderOutput,
    RecognitionRequest,
    RecognitionTask,
)


class FakeRecognitionProvider:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        outputs: dict[RecognitionTask, ProviderOutput],
        error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.outputs = dict(outputs)
        self.error = error
        self.calls = 0

    async def _call(self, task: RecognitionTask) -> ProviderOutput:
        self.calls += 1
        if self.error is not None:
            raise self.error
        try:
            return self.outputs[task]
        except KeyError as exc:
            raise KeyError(f"fake output missing for task {task.value}") from exc

    async def recognize_text(self, request: RecognitionRequest) -> ProviderOutput:
        return await self._call(RecognitionTask.TEXT_OCR)

    async def recognize_formula(self, request: RecognitionRequest) -> ProviderOutput:
        return await self._call(RecognitionTask.FORMULA_OCR)

    async def recognize_table(self, request: RecognitionRequest) -> ProviderOutput:
        return await self._call(RecognitionTask.TABLE_RECOGNITION)

    async def recognize_vision(self, request: RecognitionRequest) -> ProviderOutput:
        return await self._call(RecognitionTask.VISION)

    async def complete(self, request: RecognitionRequest) -> ProviderOutput:
        return await self._call(RecognitionTask.LLM)
