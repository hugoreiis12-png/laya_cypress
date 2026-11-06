"""Build gold/hints.jsonl: labeling hints per image, read from the Bubble filename and the Power BI nota.

Hints are NOT labels. A human confirms each field against the image before it enters gold/manifest.jsonl.
Contains client data: gold/hints.jsonl is git-ignored.
"""
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from service.app.cnpj_key import access_key_is_valid, format_cnpj  # noqa: E402

LOG = Path("gold/origem/download_powerbi.jsonl")
MANIFEST_SOURCE = Path("assets/canhotos_manifest.jsonl")
OUT = Path("gold/hints.jsonl")
KEY_PATTERN = re.compile(r"(?<!\d)(\d{44})(?!\d)")
CNPJ_SLICE = slice(6, 20)
INVOICE_SLICE = slice(25, 34)
NOTA_TAIL_DIGITS = 6
SHA_PREFIX_LENGTH = 16  # same id length evalkit uses for gold/images/<id>.<ext>


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key_from_filename(url: str) -> str:
    match = KEY_PATTERN.search(url.rsplit("/", 1)[-1])
    return match.group(1) if match and access_key_is_valid(match.group(1)) else ""


def hint_for(row: dict[str, Any]) -> dict[str, Any]:
    key = key_from_filename(row["url"])
    nota = str(row.get("nota") or "")
    invoice = key[INVOICE_SLICE].lstrip("0") if key else ""
    extension = Path(row["arquivo"]).suffix
    return {
        "image": f"images/{row['sha256'][:SHA_PREFIX_LENGTH]}{extension}",
        "sha256": row["sha256"],
        "bubble_id": row["id"],
        "nota_powerbi": nota,
        "chave_do_nome": key or None,
        "cnpj_do_nome": format_cnpj(key[CNPJ_SLICE]) if key else None,
        "numero_do_nome": key[INVOICE_SLICE] if key else None,
        "nota_bate_com_chave": bool(key) and nota.lstrip("0") == invoice[-NOTA_TAIL_DIGITS:].lstrip("0"),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [row for row in load_jsonl(LOG) if row["status"] == "ok"]
    hints = [hint_for(row) for row in rows]
    OUT.write_text("\n".join(json.dumps(h, ensure_ascii=False) for h in hints) + "\n", encoding="utf-8")
    with_key = [h for h in hints if h["chave_do_nome"]]
    agree = sum(1 for h in with_key if h["nota_bate_com_chave"])
    print(f"{len(hints)} dicas -> {OUT}; com chave no nome: {len(with_key)}; nota bate com chave: {agree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
