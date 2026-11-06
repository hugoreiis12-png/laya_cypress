"""Heuristic extraction of order items (product + quantity) from free text."""
import re
from typing import Any

_QTY = r"\d+(?:[.,]\d+)?"
_UNIT = r"(?:un|und|kg|cx|pc|pct|dz|maço|mc)?\.?"
_IGNORED = re.compile(r"\b(?:bom dia|boa tarde|boa noite|obrigad[oa]|encaminhada|ok)\b|\d{1,2}:\d{2}", re.IGNORECASE)
_CODE_AFTER = re.compile(rf"^(?P<name>[^\d=:]+?)\s+(?P<qty>{_QTY})\s+(?P<code>[A-Z]{{2,}}-?\d+)$")
_NAME_FIRST = re.compile(
    rf"^(?:(?P<code>[A-Z0-9]{{2,}})\s*-\s*)?(?P<name>[^\d=:]+?)\s*[=:x\-]?\s*(?P<qty>{_QTY})\s*{_UNIT}$",
    re.IGNORECASE,
)
_QTY_FIRST = re.compile(rf"^(?P<qty>{_QTY})\s*(?:x\s*)?{_UNIT}\s+(?P<name>[^\d=:]+)$", re.IGNORECASE)


def _quantity(raw: str) -> int | float:
    value = float(raw.replace(",", "."))
    return int(value) if value.is_integer() else value


def _parse_line(line: str) -> dict[str, Any] | None:
    for pattern in (_CODE_AFTER, _NAME_FIRST, _QTY_FIRST):
        match = pattern.match(line)
        if match:
            groups = match.groupdict()
            return {"codigo": groups.get("code"), "produto": groups["name"].strip(" -="), "qnt": _quantity(groups["qty"])}
    return None


def extract(text: str) -> list[dict[str, Any]]:
    lines = (line.strip() for line in text.splitlines())
    parsed = (_parse_line(line) for line in lines if line and not _IGNORED.search(line))
    return [item for item in parsed if item and item["produto"]]
