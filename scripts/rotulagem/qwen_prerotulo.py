"""Pre-label gold photos with the Qwen vision model on 192.168.0.140 (vLLM), following the SPEC contracts.

Suggestions only: never writes labels into the manifest. Two modes:
  --avaliar      compare with labeled records and print the agreement matrix (counts only)
  --sugerir      write suggestions for unlabeled candidates to gold/sugestoes_qwen.jsonl

Usage:
    python scripts/rotulagem/qwen_prerotulo.py --avaliar [--limite N]
    python scripts/rotulagem/qwen_prerotulo.py --sugerir [--nota candidato:canhoto_nao_assinado]
"""
import argparse
import base64
import io
import json
import random
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "gold"
SPEC = ROOT / "SPEC"
SUGGESTIONS = GOLD / "sugestoes_qwen.jsonl"
URL = "http://192.168.0.140:8001/v1/chat/completions"
MODEL = "RedHatAI/Qwen3.8-27B-INT4"
MAX_SIDE = 1280
JPEG_QUALITY = 85
WORKERS = 4
TIMEOUT_SECONDS = 120
DOC_TYPES = ("canhoto", "comprovante", "pedido", "imagem_invalida")
RULES = """Voce rotula fotos de documentos recebidas pelo WhatsApp de uma distribuidora de hortifruti.
Classifique o tipo de documento pelos contratos abaixo e, se for canhoto, diga se ha assinatura.
Regras de assinatura: conta assinatura manuscrita ou rubrica do recebedor em qualquer lugar do canhoto,
mesmo fora do campo de assinatura. Carimbo, data, nome digitado ou impresso sozinhos NAO contam.
Foto ilegivel, cortada a ponto de nao identificar o documento, ou sem documento e imagem_invalida.
Responda so o JSON pedido."""
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["tipo_documento", "possui_assinatura", "confianca"],
          "properties": {"tipo_documento": {"enum": list(DOC_TYPES)},
                         "possui_assinatura": {"type": ["boolean", "null"]},
                         "confianca": {"type": "integer", "minimum": 0, "maximum": 100}}}


def spec_text() -> str:
    parts = [f"{p.stem}: {p.read_text(encoding='utf-8').strip()}" for p in sorted(SPEC.glob("*_contract.json"))
             if p.read_text(encoding="utf-8").strip()]
    return RULES + "\n\nContratos (SPEC):\n" + "\n".join(parts)


def encoded(image_path: Path) -> str:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode()


def ask(system: str, image_path: Path) -> dict[str, Any]:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "rotulo", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [{"type": "image_url", "image_url": {
                             "url": "data:image/jpeg;base64," + encoded(image_path)}}]}]}
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(json.load(response)["choices"][0]["message"]["content"])


def safe_ask(system: str, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        return record, ask(system, GOLD / record["image"])
    except (OSError, ValueError, KeyError) as exc:
        print(f"falha {record['id']}: {type(exc).__name__}", file=sys.stderr)
        return record, None


def run_all(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    system = spec_text()
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda r: safe_ask(system, r), records))


def gold_signature(record: dict[str, Any]) -> bool | None:
    return (record.get("assinatura") or {}).get("possui_assinatura")


def evaluate(records: list[dict[str, Any]], limit: int, seed: int) -> None:
    labeled = [r for r in records if r.get("status") == "labeled" and r.get("tipo_documento") == "canhoto"]
    unsigned = [r for r in labeled if gold_signature(r) is False]
    signed = [r for r in labeled if gold_signature(r) is True]
    sample = unsigned + random.Random(seed).sample(signed, min(len(signed), max(0, limit - len(unsigned))))
    matrix: Counter = Counter()
    for record, answer in run_all(sample):
        got = "falha" if answer is None else ("nao_canhoto:" + answer["tipo_documento"] if answer["tipo_documento"] != "canhoto"
                                               else ("assinado" if answer["possui_assinatura"] else "nao_assinado"))
        matrix[("assinado" if gold_signature(record) else "nao_assinado", got)] += 1
    print(json.dumps({"amostra": len(sample), "gold_x_qwen": {f"{g} -> {q}": n for (g, q), n in sorted(matrix.items())}},
                     ensure_ascii=False, indent=1))


def suggest(records: list[dict[str, Any]], note: str) -> None:
    done = {json.loads(line)["id"] for line in SUGGESTIONS.read_text(encoding="utf-8").splitlines()} if SUGGESTIONS.exists() else set()
    todo = [r for r in records if r.get("status") == "unlabeled" and (r.get("notes") or "").strip() == note and r["id"] not in done]
    stats: Counter = Counter()
    with SUGGESTIONS.open("a", encoding="utf-8") as out:
        for record, answer in run_all(todo):
            if answer is None:
                stats["falha"] += 1
                continue
            out.write(json.dumps({"id": record["id"], **answer}, ensure_ascii=False) + "\n")
            stats[answer["tipo_documento"] + ("" if answer["tipo_documento"] != "canhoto" else
                                              (":assinado" if answer["possui_assinatura"] else ":nao_assinado"))] += 1
    print(json.dumps({"processados": len(todo), **stats, "arquivo": str(SUGGESTIONS)}, ensure_ascii=False))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--avaliar", action="store_true")
    mode.add_argument("--sugerir", action="store_true")
    parser.add_argument("--limite", type=int, default=120, help="tamanho da amostra de avaliacao")
    parser.add_argument("--semente", type=int, default=7)
    parser.add_argument("--nota", default="candidato:canhoto_nao_assinado")
    args = parser.parse_args()
    lines = (GOLD / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if args.avaliar:
        evaluate(records, args.limite, args.semente)
    else:
        suggest(records, args.nota)
    return 0


if __name__ == "__main__":
    sys.exit(main())
