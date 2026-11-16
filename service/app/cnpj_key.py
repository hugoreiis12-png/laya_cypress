"""Deterministic extraction of CNPJ, NF-e access key and invoice number."""
import itertools
import re

CNPJ_LENGTH = 14
KEY_LENGTH = 44
KEY_CHECK_POSITION = 43
KEY_CNPJ = slice(6, 20)
KEY_NUMBER = slice(25, 34)
CNPJ_FIRST_WEIGHTS = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
CNPJ_SECOND_WEIGHTS = (6,) + CNPJ_FIRST_WEIGHTS
MODULUS = 11
MIN_REMAINDER = 2
KEY_WEIGHT_RANGE = range(2, 10)

_KEY_PATTERN = re.compile(r"(?<!\d)(?:\d{4}[ .]?){10}\d{4}(?!\d)")
# OCR splits CNPJs with spaces and commas ("46.126.673/000 193"); the check digit and whitelist filter candidates
_CNPJ_PATTERN = re.compile(r"(?<!\d)(?=(\d(?:[ .,/-]{0,2}\d){13}))")
_GROUPED_NUMBER = r"(\d{1,3}(?:[.,]\d{3})+|\d{1,9})"  # OCR often reads the thousands dot as a comma
_NUMBER_PATTERN = re.compile(rf"(?:N[º°o]|NF-?e?|N[úu]mero)\.?\s*:?\s*{_GROUPED_NUMBER}(?!\d)", re.IGNORECASE)
# "N. 117.986 SÉRIE 1": some DANFEs print a plain "N." and the series right after it anchors the number
_SERIES_NUMBER_PATTERN = re.compile(rf"(?<![A-Za-z])N\.\s*{_GROUPED_NUMBER}\s+S[ée]rie(?![a-z])", re.IGNORECASE)


def digits(text: object) -> str:
    return re.sub(r"\D", "", str(text or ""))


def _check_digit(total: int) -> int:
    remainder = total % MODULUS
    return 0 if remainder < MIN_REMAINDER else MODULUS - remainder


def cnpj_is_valid(value: str) -> bool:
    numbers = [int(char) for char in digits(value)]
    if len(numbers) != CNPJ_LENGTH or len(set(numbers)) == 1:
        return False
    first = _check_digit(sum(n * w for n, w in zip(numbers, CNPJ_FIRST_WEIGHTS)))
    second = _check_digit(sum(n * w for n, w in zip(numbers, CNPJ_SECOND_WEIGHTS)))
    return numbers[12:] == [first, second]


def access_key_is_valid(value: str) -> bool:
    numbers = [int(char) for char in digits(value)]
    if len(numbers) != KEY_LENGTH:
        return False
    weights = itertools.cycle(KEY_WEIGHT_RANGE)
    total = sum(n * next(weights) for n in reversed(numbers[:KEY_CHECK_POSITION]))
    return numbers[KEY_CHECK_POSITION] == _check_digit(total)


def format_cnpj(value: str) -> str:
    d = digits(value)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def strip_access_keys(text: str) -> str:
    return _KEY_PATTERN.sub(" ", text)


def find_access_keys(text: str) -> list[str]:
    found = (digits(match.group()) for match in _KEY_PATTERN.finditer(text))
    return list(dict.fromkeys(key for key in found if access_key_is_valid(key)))


def find_cnpjs(text: str) -> list[str]:
    found = (digits(match.group(1)) for match in _CNPJ_PATTERN.finditer(strip_access_keys(text)))
    return list(dict.fromkeys(cnpj for cnpj in found if cnpj_is_valid(cnpj)))


def pick_issuer(text: str, whitelist: tuple[str, ...], key: str = "") -> str:
    """Issuer CNPJ in contract format; only whitelisted values are ever returned."""
    if key and key[KEY_CNPJ] in whitelist:
        return format_cnpj(key[KEY_CNPJ])
    return next((format_cnpj(cnpj) for cnpj in find_cnpjs(text) if cnpj in whitelist), "")


def invoice_number(text: str, key: str = "") -> str:
    if key and access_key_is_valid(key):
        return str(int(key[KEY_NUMBER]))
    numbers = invoice_numbers(text)
    return numbers[0] if numbers else ""


def invoice_numbers(text: str) -> list[str]:
    """Every labeled invoice number in the text, in order; a DANFE prints its own at least twice."""
    clean = strip_access_keys(text)
    matches = [*_NUMBER_PATTERN.finditer(clean), *_SERIES_NUMBER_PATTERN.finditer(clean)]
    found = (digits(match.group(1)).lstrip("0") for match in sorted(matches, key=lambda match: match.start()))
    return list(dict.fromkeys(number for number in found if number))
