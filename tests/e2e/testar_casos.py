"""Sends a folder of photos to the service and prints one summary row per photo; never prints field values.

Usage: API_KEY=... python3 tests/e2e/testar_casos.py --pasta /tmp/casos [--url http://192.168.0.160:8020] [--esperado casos.json]
`casos.json` maps a file name to what is acceptable, e.g. {"nota.jpg": {"tipo": "canhoto", "inconclusivo": false}}.
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://192.168.0.160:8020"
ENDPOINT = "/v1/extrair-documento"
TIMEOUT_SECONDS = 60
SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_ID_CHARS = 64
COLUMNS = ("arquivo", "tipo", "assinatura", "inconclusivo", "motivos", "rotacao", "ocr", "juiz", "estado", "ms", "campos", "ok")


def call(url: str, api_key: str, photo: Path) -> dict:
    payload = {"imagem_base64": base64.b64encode(photo.read_bytes()).decode("ascii"),
               "contexto": {"id": f"teste-{photo.stem}"[:MAX_ID_CHARS], "origem": "testar_casos"}}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(url + ENDPOINT, json.dumps(payload).encode("utf-8"), headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def _filled(output: dict) -> str:
    """Which of cnpj, numero and chave came filled; the values themselves stay out of the report."""
    values = (("cnpj", (output.get("emitente") or {}).get("cnpj")), ("numero", (output.get("nota_fiscal") or {}).get("numero")),
              ("chave", (output.get("qrcode") or {}).get("chave")))
    return "+".join(name for name, value in values if value) or "-"


def summarize(name: str, result: dict) -> dict:
    output, meta = json.loads(result["output"]), result["_meta"]
    signature = output.get("assinatura") or {}
    return {"arquivo": name, "tipo": output.get("tipo_documento"),
            "assinatura": f"{signature.get('possui_assinatura')}/{signature.get('confianca')}" if signature else "-",
            "inconclusivo": bool(meta.get("inconclusivo")), "motivos": ",".join(meta.get("motivos_inconclusivo", [])) or "-",
            "rotacao": meta.get("rotacao"), "ocr": (meta.get("leitura") or {}).get("caracteres_ocr"),
            "juiz": (meta.get("revisao_assinatura") or {}).get("resultado") or meta.get("capacidade_juiz") or "-",
            "estado": meta.get("estado_assinatura"), "ms": meta.get("latencia_ms"), "campos": _filled(output)}


def verdict(row: dict, expected: dict | None) -> str:
    if not expected:
        return "?"
    wrong = [key for key, value in expected.items() if row.get(key) != value]
    return "OK" if not wrong else "FALHOU:" + ",".join(wrong)


def photos_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in SUFFIXES)


def run_one(url: str, api_key: str, photo: Path, expected: dict) -> dict:
    try:
        row = summarize(photo.name, call(url, api_key, photo))
    except urllib.error.HTTPError as exc:
        return {"arquivo": photo.name, "tipo": f"HTTP {exc.code}", "ok": "FALHOU:servico"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"arquivo": photo.name, "tipo": f"erro: {type(exc).__name__}", "ok": "FALHOU:servico"}
    row["ok"] = verdict(row, expected.get(photo.name))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pasta", required=True, type=Path)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--esperado", type=Path)
    args = parser.parse_args()
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        sys.exit("defina API_KEY no ambiente (não passe a chave na linha de comando)")
    expected = json.loads(args.esperado.read_text(encoding="utf-8")) if args.esperado else {}
    print(" | ".join(COLUMNS))
    rows = [run_one(args.url, api_key, photo, expected) for photo in photos_in(args.pasta)]
    for row in rows:
        print(" | ".join(str(row.get(column, "")) for column in COLUMNS))
    return 1 if any(str(row.get("ok", "")).startswith("FALHOU") for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
