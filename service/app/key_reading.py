"""Asks the judge for the access key of a photo the barcode and OCR could not read; only a validated key counts."""
from typing import Protocol

from .cnpj_key import KEY_CNPJ, access_key_is_valid, find_cnpjs, invoice_number, invoice_numbers
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
    """A wrong read passes the check digit about once in ten; it conflicts only if no printed copy matches the key."""
    printed_numbers = invoice_numbers(text)
    if printed_numbers and invoice_number("", key) not in printed_numbers:
        return False
    printed_issuers = [cnpj for cnpj in find_cnpjs(text) if cnpj in whitelist]
    return not printed_issuers or key[KEY_CNPJ] in printed_issuers


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
