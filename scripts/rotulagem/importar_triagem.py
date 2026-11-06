"""Importa gold/triagem/triagem.csv para gold/manifest.jsonl.

Valida tudo antes de gravar. Qualquer linha inválida interrompe a importação sem alterar o manifest.
Sem --aplicar, só valida e mostra o resumo.

Uso:
    python scripts/rotulagem/importar_triagem.py            # valida
    python scripts/rotulagem/importar_triagem.py --aplicar  # grava, com backup do manifest anterior
Registros importados ganham rotulo_parcial=true: só tipo e assinatura são confiáveis, os campos
de emitente, nota e chave não foram rotulados e não são pontuados.
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalkit.manifest import DOC_TYPES, STATUS_LABELED  # noqa: E402

GOLD = Path("gold")
MANIFEST = GOLD / "manifest.jsonl"
CSV_PATH = GOLD / "triagem" / "triagem.csv"
BACKUPS = GOLD / "backups"
CANDIDATE_NOTE = "candidato:canhoto_nao_assinado"
BOOLEANS = {"true": True, "false": False}
STRAY_QUOTES = "'\" "


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_note(raw: str) -> str:
    return raw.strip().strip(STRAY_QUOTES).strip()


def check_row(row: dict[str, str], candidates: set[str], seen: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    rid = (row.get("id") or "").strip()
    if rid not in candidates:
        return None, f"id {rid!r} não é candidato a não assinado"
    if rid in seen:
        return None, f"id {rid!r} duplicado no CSV"
    tipo = (row.get("tipo_documento") or "").strip()
    if tipo not in DOC_TYPES:
        return None, f"{rid}: tipo_documento {tipo!r} inválido (use {', '.join(DOC_TYPES)})"
    sig_raw = (row.get("possui_assinatura") or "").strip().lower()
    if sig_raw not in BOOLEANS:
        return None, f"{rid}: possui_assinatura {sig_raw!r} deve ser true ou false"
    return {"id": rid, "tipo_documento": tipo, "possui_assinatura": BOOLEANS[sig_raw],
            "nota": clean_note(row.get("nota") or "")}, None


def validate(rows: list[dict[str, str]], manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    candidates = {r["id"] for r in manifest if (r.get("notes") or "").strip() == CANDIDATE_NOTE}
    accepted: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, start=2):
        item, error = check_row(row, candidates, seen)
        if error:
            errors.append(f"linha {number}: {error}")
            continue
        seen.add(item["id"])
        accepted.append(item)
    pending = sorted(candidates - seen)
    return accepted, errors, pending


def apply_labels(manifest: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in accepted}
    for record in manifest:
        item = by_id.get(record["id"])
        if not item:
            continue
        record.update(status=STATUS_LABELED, tipo_documento=item["tipo_documento"],
                      assinatura={"possui_assinatura": item["possui_assinatura"], "descricao": ""},
                      rotulo_parcial=True, triagem={"nota": item["nota"], "fonte": "triagem_manual"})
    return manifest


def write_with_backup(manifest: list[dict[str, Any]]) -> Path:
    BACKUPS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = BACKUPS / f"manifest-{stamp}.jsonl"
    shutil.copyfile(MANIFEST, backup)
    temp = MANIFEST.with_suffix(".jsonl.tmp")
    temp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8")
    temp.replace(MANIFEST)
    return backup


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--aplicar", action="store_true", help="grava no manifest; sem isso só valida")
    args = parser.parse_args(argv)
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = read_jsonl(MANIFEST)
    accepted, errors, pending = validate(rows, manifest)
    if errors:
        print("Importação recusada, nada foi gravado. Corrija:")
        print("\n".join(errors))
        return 1
    signed = sum(1 for a in accepted if a["possui_assinatura"])
    print(f"válidas: {len(accepted)} (assinados {signed}, não assinados {len(accepted) - signed}); pendentes: {len(pending)}")
    if not args.aplicar:
        print("validação OK. Rode com --aplicar para gravar.")
        return 0
    backup = write_with_backup(apply_labels(manifest, accepted))
    print(f"gravado. backup anterior: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
