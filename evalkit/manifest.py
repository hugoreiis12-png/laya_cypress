"""Gold manifest: one JSON record per image, appended as WhatsApp documents arrive."""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

DOC_TYPES = ("canhoto", "comprovante", "pedido", "imagem_invalida")
STATUS_LABELED = "labeled"
STATUS_UNLABELED = "unlabeled"
STATUS_DISCARDED = "descartado"  # out of the gold set; the sha256 blocks re-import
CNPJ_DIGITS = 14
ACCESS_KEY_DIGITS = 44
TEST_SPLIT_MODULO = 5  # 1 in 5 images is held out for the test split
ID_LENGTH = 16


def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records)
    path.write_text(body + "\n", encoding="utf-8")


def split_of(record: dict[str, Any]) -> str:
    return "test" if int(record["sha256"][:8], 16) % TEST_SPLIT_MODULO == 0 else "dev"


def labeled(records: list[dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    keep = [rec for rec in records if rec.get("status") == STATUS_LABELED]
    return [rec for rec in keep if split in (None, "all") or split_of(rec) == split]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_image(src: Path, root: Path, records: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    sha = sha256_of(src)
    if any(rec["sha256"] == sha for rec in records):
        return None
    record_id = sha[:ID_LENGTH]
    target = root / "images" / f"{record_id}{src.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    stub = {"id": record_id, "image": f"images/{target.name}", "sha256": sha, "source": source,
            "status": STATUS_UNLABELED, "tipo_documento": None, "ocr_text": None, "notes": ""}
    records.append(stub)
    return stub


def _digits_errors(value: Any, expected: int, label: str) -> list[str]:
    digits = "".join(char for char in str(value or "") if char.isdigit())
    return [f"{label} deve ter {expected} dígitos"] if value and len(digits) != expected else []


def _labeled_errors(rec: dict[str, Any]) -> list[str]:
    errors = [] if rec.get("tipo_documento") in DOC_TYPES else ["tipo_documento inválido"]
    if rec.get("tipo_documento") == "canhoto":
        signature = rec.get("assinatura") or {}
        if not isinstance(signature.get("possui_assinatura"), bool):
            errors.append("canhoto exige assinatura.possui_assinatura booleano")
        errors += _digits_errors((rec.get("emitente") or {}).get("cnpj"), CNPJ_DIGITS, "emitente.cnpj")
        errors += _digits_errors((rec.get("qrcode") or {}).get("chave"), ACCESS_KEY_DIGITS, "qrcode.chave")
    if rec.get("tipo_documento") == "pedido" and not isinstance(rec.get("itens"), list):
        errors.append("pedido exige itens[]")
    return errors


def validate(rec: dict[str, Any], root: Path) -> list[str]:
    if rec.get("status") == STATUS_DISCARDED:
        return []
    image = root / rec.get("image", "")
    if not image.exists():
        return ["imagem ausente em disco"]
    if sha256_of(image) != rec.get("sha256"):
        return ["sha256 não confere com a imagem"]
    return _labeled_errors(rec) if rec.get("status") == STATUS_LABELED else []
