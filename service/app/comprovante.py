"""Heuristic extraction of payment-receipt fields (PIX/TED/DOC)."""
import re
from datetime import date

BANKS = (
    "Banco do Brasil", "Bradesco", "Itaú", "Itau", "Santander", "Caixa", "Nubank", "Inter",
    "C6 Bank", "Sicoob", "Sicredi", "BTG", "PagBank", "Mercado Pago", "Banrisul", "Safra",
    "PicPay", "Neon", "Original",
)
FIELDS = ("instituicao", "valor", "data", "hora", "favorecido", "pagador", "identificacao")
MAX_NAME_LENGTH = 80

_MONEY = re.compile(r"R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})")
_DATE = re.compile(r"(?<!\d)(\d{2})/(\d{2})/(\d{4})(?!\d)")
_TIME = re.compile(r"(?<!\d)([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?!\d)")
_PAYEE = r"favorecido|benefici[aá]rio|destinat[aá]rio|recebedor|destino"
_PAYER = r"pagador|remetente|origem|quem pagou"
_ID = r"id da transa[cç][aã]o|c[oó]digo de autentica[cç][aã]o|autentica[cç][aã]o|identifica[cç][aã]o|id"


def _bank(text: str) -> str | None:
    lowered = text.casefold()
    return next((bank for bank in BANKS if bank.casefold() in lowered), None)


def _amount(text: str) -> str | None:
    match = _MONEY.search(text)
    return match.group(1).replace(".", "").replace(",", ".") if match else None


def _iso_date(text: str) -> str | None:
    for day, month, year in _DATE.findall(text):
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            continue
    return None


def _clock(text: str) -> str | None:
    match = _TIME.search(text)
    return f"{match.group(1)}:{match.group(2)}:{match.group(3) or '00'}" if match else None


def _labeled_name(text: str, labels: str) -> str | None:
    match = re.search(rf"(?:{labels})\s*:?\s*\n?\s*([^\n]{{2,{MAX_NAME_LENGTH}}})", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _labeled_code(text: str) -> str | None:
    match = re.search(rf"\b(?:{_ID})\b\s*:?\s*([A-Za-z0-9]{{6,}})", text, re.IGNORECASE)
    return match.group(1) if match else None


def extract(text: str) -> dict[str, str | None]:
    return {
        "instituicao": _bank(text), "valor": _amount(text), "data": _iso_date(text),
        "hora": _clock(text), "favorecido": _labeled_name(text, _PAYEE),
        "pagador": _labeled_name(text, _PAYER), "identificacao": _labeled_code(text),
    }
