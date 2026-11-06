"""Everything one document run can call, and the time and capacity limits those calls share."""
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import Settings
from .judge_capacity import JudgeCapacity
from .laya_client import TypeDecision
from .signature_model import SignatureDecision
from .vision.analyze import Evidence
from .vision_judge import SignatureOpinion

MIN_REVIEW_SECONDS = 1.0  # less time than this cannot produce an answer from the judge


class Classifier(Protocol):
    def classify(self, ocr_text: str, extracted: dict[str, Any]) -> TypeDecision: ...


class SignatureModel(Protocol):
    def decide(self, state: dict[str, Any]) -> SignatureDecision: ...


class SignatureShadow(Protocol):
    def observe(self, data: bytes, rotation: int) -> dict[str, Any]: ...


class Judge(Protocol):
    def transcribe(self, data: bytes) -> str: ...

    def read_access_key(self, data: bytes, timeout: float) -> str: ...

    def assess_signature(self, data: bytes, timeout: float) -> SignatureOpinion: ...


class Auditor(Protocol):
    def audit(self, answer: dict[str, Any], ocr_text: str, timeout: float) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Engines:
    settings: Settings
    analyze: Callable[[bytes], Evidence]
    laya: Classifier
    signature_model: SignatureModel | None = None
    shadow: SignatureShadow | None = None
    hermes: Classifier | None = None
    judge: Judge | None = None
    judge_capacity: JudgeCapacity | None = None
    auditor: Auditor | None = None

    def seconds_left(self, started: float) -> float:
        return self.settings.review_deadline - (time.monotonic() - started)

    def enter_judge(self, meta: dict[str, Any]) -> bool:
        if self.judge_capacity is None:
            return True
        refusal = self.judge_capacity.enter()
        meta["capacidade_juiz"] = refusal or "ok"
        return refusal is None

    def leave_judge(self) -> None:
        if self.judge_capacity is not None:
            self.judge_capacity.leave()
