"""Rank unlabeled photos of one source as unsigned-canhoto candidates and mark them for the rotulador.

Uses only the cached OCR/ink (gold/ocr/<sha256>.json). Canhoto-like text first, then little ink in the
recipient field. Marks the top --limite with notes "candidato:canhoto_nao_assinado"; the label is the user's.
Dry-run by default; --aplicar writes the manifest with a backup. Prints counts only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rotulagem.importar_triagem import MANIFEST, read_jsonl, write_with_backup  # noqa: E402

GOLD = ROOT / "gold"
NOTE = "candidato:canhoto_nao_assinado"
CANHOTO_TERMS = ("RECEBEMOS", "CANHOTO", "RECEBIMENTO", "RECEBEDOR", "IDENTIFICA", "ASSINATURA", "NF-E", "NOTA FISCAL")
MIN_TERMS = 2
SIGNED_INK_P10 = 10_000  # ink_pixels p10 of signed canhotos with the field found (dev benchmark)


def canhoto_terms(text: str) -> int:
    upper = text.upper()
    return sum(term in upper for term in CANHOTO_TERMS)


def score(cached: dict) -> tuple[int, int, int] | None:
    terms = canhoto_terms(cached.get("text", ""))
    if terms < MIN_TERMS and not cached.get("access_key"):
        return None
    ink = cached["ink"]
    if ink["region_found"]:
        return (0 if ink["ink_pixels"] < SIGNED_INK_P10 else 2, ink["ink_pixels"], -terms)
    return (1, 0, -terms)  # field not found: unknown ink, after the low-ink ones


def ranked(records: list[dict], fonte: str) -> tuple[list[tuple], Counter]:
    stats: Counter = Counter()
    out = []
    for record in records:
        if record.get("source") != fonte or record.get("status") != "unlabeled":
            continue
        path = GOLD / "ocr" / f"{record['sha256']}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if not cached or not cached.get("ok"):
            stats["sem_ocr"] += 1
            continue
        key = score(cached)
        if key is None:
            stats["nao_parece_canhoto"] += 1
            continue
        stats[("pouca_tinta", "campo_nao_achado", "tinta_alta")[key[0]]] += 1
        out.append((key, record))
    return sorted(out, key=lambda item: item[0]), stats


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte", default="bubble_foto_canhoto")
    parser.add_argument("--limite", type=int, default=300)
    parser.add_argument("--aplicar", action="store_true")
    args = parser.parse_args()
    manifest = read_jsonl(MANIFEST)
    candidates, stats = ranked(manifest, args.fonte)
    chosen = [record for key, record in candidates if key[0] < 2][: args.limite]
    print(json.dumps({"parecem_canhoto": len(candidates), **stats, "marcados": len(chosen)}, ensure_ascii=False))
    if not args.aplicar:
        print("simulação. Rode com --aplicar para marcar no manifest.")
        return 0
    for record in chosen:
        record["notes"] = NOTE
    print(f"gravado. backup anterior: {write_with_backup(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
