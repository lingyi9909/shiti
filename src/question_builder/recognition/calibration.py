from __future__ import annotations

from dataclasses import dataclass

from question_builder.recognition.contracts import RecognitionTask


class MissingCalibrationError(LookupError):
    """Raised when a provider/model/task has no approved calibration profile."""


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    provider: str
    model: str
    task: RecognitionTask
    version: str
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.provider or not self.model or not self.version:
            raise ValueError("calibration provider, model, and version are required")

    @property
    def calibration_id(self) -> str:
        return f"{self.provider}/{self.model}/{self.task.value}@{self.version}"

    def normalize(self, raw_score: float) -> float:
        if not 0.0 <= raw_score <= 1.0:
            raise ValueError("raw_score must be between 0 and 1")
        return min(1.0, max(0.0, raw_score * self.scale + self.offset))


@dataclass(frozen=True, slots=True)
class NormalizedScore:
    score: float
    calibration_id: str


class CalibrationRegistry:
    def __init__(self, profiles: tuple[CalibrationProfile, ...] = ()) -> None:
        self._profiles = {
            (profile.provider, profile.model, profile.task): profile
            for profile in profiles
        }
        if len(self._profiles) != len(profiles):
            raise ValueError("duplicate calibration profile identity")

    def resolve(
        self,
        provider: str,
        model: str,
        task: RecognitionTask,
    ) -> CalibrationProfile:
        try:
            return self._profiles[(provider, model, task)]
        except KeyError as exc:
            raise MissingCalibrationError(
                f"missing calibration for {provider}/{model}/{task.value}"
            ) from exc


def normalize_score(
    provider: str,
    model: str,
    task: RecognitionTask,
    raw_score: float,
    *,
    registry: CalibrationRegistry,
) -> NormalizedScore:
    profile = registry.resolve(provider, model, task)
    return NormalizedScore(
        score=profile.normalize(raw_score),
        calibration_id=profile.calibration_id,
    )
