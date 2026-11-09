"""Labels by consensus of sources independent of the model being trained; one source alone never labels."""
from collections.abc import Mapping
from dataclasses import dataclass

LABEL_SIGNED, LABEL_UNSIGNED, LABEL_STAMP_ONLY = "assinado", "nao_assinado", "carimbo_somente"
SOURCE_VLLM, SOURCE_GEMINI, SOURCE_BUBBLE = "vllm", "gemini", "bubble"
AGREED, DISAGREED, INSUFFICIENT = "concordam", "discordam", "insuficiente"


@dataclass(frozen=True)
class Sample:
    item_id: str
    votes: Mapping[str, bool | None]
    rotation: int = 0


@dataclass(frozen=True)
class Decision:
    status: str
    label: str | None
    reason: str


@dataclass(frozen=True)
class Partition:
    training: tuple[tuple[Sample, str], ...]
    review: tuple[Sample, ...]
    skipped: int


def decide(votes: Mapping[str, bool | None]) -> Decision:
    """Bubble only confirms a signature: its "unsigned" is not evidence, and its "signed" can veto an unsigned label."""
    vllm, gemini = votes.get(SOURCE_VLLM), votes.get(SOURCE_GEMINI)
    if vllm is None or gemini is None:
        return Decision(INSUFFICIENT, None, "faltou um voto independente")
    if vllm != gemini:
        return Decision(DISAGREED, None, "vLLM e Gemini discordam")
    if not vllm and votes.get(SOURCE_BUBBLE) is True:
        return Decision(DISAGREED, None, "o Bubble indica assinatura")
    return Decision(AGREED, LABEL_SIGNED if vllm else LABEL_UNSIGNED, "as fontes concordam")


def partition(samples: list[Sample]) -> Partition:
    training: list[tuple[Sample, str]] = []
    review: list[Sample] = []
    skipped = 0
    for sample in samples:
        decision = decide(sample.votes)
        if decision.status == AGREED and decision.label:
            training.append((sample, decision.label))
        elif decision.status == DISAGREED:
            review.append(sample)
        else:
            skipped += 1
    return Partition(tuple(training), tuple(review), skipped)
