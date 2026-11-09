"""Removes personal data from OCR text before it leaves the host; ids become keyed pseudonyms."""
import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Protocol

PSEUDONYM_CHARS = 16
PATTERNS = (
    ("CHAVE", re.compile(r"(?<!\d)(?:\d[ .-]?){44}(?!\d)")),
    ("CNPJ", re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("CPF", re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")),
    ("TELEFONE", re.compile(r"(?<!\d)\(?\d{2}\)?\s?9?\d{4}-?\d{4}(?!\d)")),
    ("CEP", re.compile(r"(?<!\d)\d{5}-?\d{3}(?!\d)")),
)


class NameRedactor(Protocol):
    def redact(self, text: str) -> str: ...


@dataclass(frozen=True)
class Anonymized:
    text: str
    names_removed: bool


def anonymize(text: str, names: NameRedactor | None = None) -> Anonymized:
    """Patterns cover documents and contacts; person names need a local model, and the result says whether one ran."""
    for label, pattern in PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    if names is not None:
        text = names.redact(text)
    return Anonymized(text, names is not None)


def find_residual(text: str) -> list[str]:
    """Labels of the personal data patterns still present, used as a last check before packaging."""
    return [label for label, pattern in PATTERNS if pattern.search(text)]


def pseudonym(secret: str, value: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:PSEUDONYM_CHARS]
