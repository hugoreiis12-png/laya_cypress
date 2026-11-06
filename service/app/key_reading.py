"""Asks the judge for the access key of a photo the barcode and OCR could not read; only a validated key counts."""
from typing import Protocol

from .cnpj_key import KEY_CNPJ, access_key_is_valid
from .judge_capacity import JudgeCapacity
from .vision_judge import JudgeError

FOUND = "chave"
NOT_READ = "sem_chave"
INVALID = "invalida"
OUTSIDE_WHITELIST = "fora_da_whitelist"
NO_CAPACITY = "sem_capacidade"
FAILED = "falha"


class KeyReader(Protocol):
    def read_access_key(self, data: bytes, timeout: float) -> str: ...


def _validated(read: str, whitelist: tuple[str, ...]) -> tuple[str, str]:
    if not read:
        return NOT_READ, ""
    if not access_key_is_valid(read):
        return INVALID, ""
    return (FOUND, read) if read[KEY_CNPJ] in whitelist else (OUTSIDE_WHITELIST, "")


def read(judge: KeyReader, capacity: JudgeCapacity | None, data: bytes, timeout: float,
         whitelist: tuple[str, ...]) -> tuple[str, str]:
    """Returns what happened and the key, which is empty unless it passed every check."""
    if capacity is not None and capacity.enter() is not None:
        return NO_CAPACITY, ""
    try:
        return _validated(judge.read_access_key(data, timeout), whitelist)
    except JudgeError:
        return FAILED, ""
    finally:
        if capacity is not None:
            capacity.leave()
