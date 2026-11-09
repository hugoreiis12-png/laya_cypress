"""Asks the judge for the access key of a photo the barcode and OCR could not read; only a validated key counts."""
from typing import Protocol

from .cnpj_key import KEY_CNPJ, access_key_is_valid, invoice_number, pick_issuer
from .judge_capacity import JudgeCapacity
from .vision_judge import JudgeError

FOUND = "chave"
NOT_READ = "sem_chave"
INVALID = "invalida"
OUTSIDE_WHITELIST = "fora_da_whitelist"
TEXT_CONFLICT = "conflito_com_o_texto"
NO_CAPACITY = "sem_capacidade"
FAILED = "falha"


class KeyReader(Protocol):
    def read_access_key(self, data: bytes, timeout: float) -> str: ...


def _agrees_with_text(key: str, text: str, whitelist: tuple[str, ...]) -> bool:
    """The check digit passes by chance for about one wrong read in ten; what the OCR read of the same note must match."""
    printed_number = invoice_number(text)
    if printed_number and printed_number != invoice_number("", key):
        return False
    printed_issuer = pick_issuer(text, whitelist)
    return not printed_issuer or printed_issuer == pick_issuer("", whitelist, key)


def _validated(read: str, whitelist: tuple[str, ...], text: str) -> tuple[str, str]:
    if not read:
        return NOT_READ, ""
    if not access_key_is_valid(read):
        return INVALID, ""
    if read[KEY_CNPJ] not in whitelist:
        return OUTSIDE_WHITELIST, ""
    return (FOUND, read) if _agrees_with_text(read, text, whitelist) else (TEXT_CONFLICT, "")


def read(judge: KeyReader, capacity: JudgeCapacity | None, data: bytes, timeout: float,
         whitelist: tuple[str, ...], text: str = "") -> tuple[str, str]:
    """Returns what happened and the key, which is empty unless it passed every check."""
    if capacity is not None and capacity.enter() is not None:
        return NO_CAPACITY, ""
    try:
        return _validated(judge.read_access_key(data, timeout), whitelist, text)
    except JudgeError:
        return FAILED, ""
    finally:
        if capacity is not None:
            capacity.leave()
