"""Value normalizers shared by scoring."""
import re
import unicodedata
from typing import Any

INVOICE_NUMBER_WIDTH = 6  # mirrors padStart(6, '0') in n8n ajustar_resposta


def digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def invoice_number(value: Any) -> str:
    only_digits = digits(value)
    return only_digits.zfill(INVOICE_NUMBER_WIDTH) if only_digits else ""


def text(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", plain).strip().casefold()


def quantity(value: Any) -> str:
    try:
        return f"{float(str(value).replace(',', '.')):g}"
    except (TypeError, ValueError):
        return ""
