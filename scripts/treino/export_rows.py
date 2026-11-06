"""Turn labeled canhotos + cached OCR/ink (gold/ocr/<sha256>.json) into hf-encoder typed rows.

Writes into --saida: rows.jsonl (dev split), teste.jsonl (evalkit test split, declared test) and
resumo.json. The state is the service state (state_for) plus the ink evidence of the recipient field.
Needs WHITELIST_CNPJ in the environment. Prints counts only.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE.parent  # in the service container PYTHONPATH=/srv
sys.path.insert(0, str(ROOT))

from service.app.config import _whitelist  # noqa: E402
from service.app.laya_client import state_for  # noqa: E402
from service.app.pipeline import extract_fields, laya_signals  # noqa: E402
from service.app.vision.analyze import Evidence  # noqa: E402
from service.app.vision.types import InkFeatures  # noqa: E402

QUESTION = "possui_assinatura"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_test(record: dict[str, Any]) -> bool:
    # same rule as evalkit.manifest.split_of: the evalkit test split stays the declared test
    return int(record["sha256"][:8], 16) % 5 == 0


def typed_row(record: dict[str, Any], cached: dict[str, Any], whitelist: tuple[str, ...]) -> dict[str, Any]:
    ink = InkFeatures(**cached["ink"])
    evidence = Evidence(cached["text"], cached["access_key"], ink, cached["rotation"])
    fields = extract_fields(evidence, whitelist)
    state = state_for(evidence.text, laya_signals(fields), ink)
    return {
        "id": record["id"],
        "group_id": fields["chave"] or record["sha256"],  # two photos of the same nota never cross splits
        "state": state,
        "answers": {QUESTION: bool(record["assinatura"]["possui_assinatura"])},
    }


def skip_reason(record: dict[str, Any], cached: dict[str, Any] | None) -> str | None:
    if record.get("status") != "labeled" or record.get("tipo_documento") != "canhoto":
        return "fora_do_escopo"
    if not isinstance((record.get("assinatura") or {}).get("possui_assinatura"), bool):
        return "sem_rotulo_assinatura"
    if cached is None:
        return "sem_ocr"
    if not cached.get("ok"):
        return "ocr_erro"
    if not cached["text"].strip():
        return "texto_vazio"  # the service never asks Laya about an empty OCR text
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=ROOT / "gold")
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    gold, out = args.gold, args.saida
    whitelist = _whitelist(os.environ.get("WHITELIST_CNPJ", ""))
    if not whitelist:
        print(json.dumps({"erro": "WHITELIST_CNPJ ausente no ambiente"}))
        return 1
    outputs: dict[str, list[dict[str, Any]]] = {"rows": [], "teste": []}
    skipped: Counter[str] = Counter()
    labels: dict[str, Counter[str]] = {"rows": Counter(), "teste": Counter()}
    for record in read_jsonl(gold / "manifest.jsonl"):
        cache_path = gold / "ocr" / f"{record['sha256']}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        reason = skip_reason(record, cached)
        if reason:
            skipped[reason] += 1
            continue
        target = "teste" if is_test(record) else "rows"
        row = typed_row(record, cached, whitelist)
        outputs[target].append(row)
        labels[target][str(row["answers"][QUESTION]).lower()] += 1
    for name, rows in outputs.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    summary = {"linhas": {k: len(v) for k, v in outputs.items()}, "rotulos": {k: dict(v) for k, v in labels.items()},
               "descartados": dict(skipped)}
    (out / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
