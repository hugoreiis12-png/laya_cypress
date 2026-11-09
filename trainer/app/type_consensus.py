"""Labels the document type by the agreement of the rules engine and Gemini; Laya never labels itself."""
from collections.abc import Mapping
from dataclasses import dataclass

DOC_TYPES = ("canhoto", "comprovante", "pedido", "imagem_invalida")
SOURCE_RULES, SOURCE_GEMINI = "regras", "gemini"
AGREED, DISAGREED, INSUFFICIENT = "concordam", "discordam", "insuficiente"


@dataclass(frozen=True)
class TypeSample:
    item_id: str
    votes: Mapping[str, str | None]
    rotation: int = 0


@dataclass(frozen=True)
class TypeDecision:
    status: str
    label: str | None


def decide(votes: Mapping[str, str | None]) -> TypeDecision:
    rules, gemini = votes.get(SOURCE_RULES), votes.get(SOURCE_GEMINI)
    if rules not in DOC_TYPES or gemini not in DOC_TYPES:
        return TypeDecision(INSUFFICIENT, None)
    return TypeDecision(AGREED, rules) if rules == gemini else TypeDecision(DISAGREED, None)


def training_set(samples: list[TypeSample]) -> tuple[tuple[TypeSample, str], ...]:
    decided = [(sample, decide(sample.votes)) for sample in samples]
    return tuple((sample, decision.label) for sample, decision in decided if decision.label)
