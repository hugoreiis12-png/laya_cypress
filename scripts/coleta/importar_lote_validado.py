"""Import photos the user validated outside the labeler (result files from a Bubble pipeline) into the gold set.

Keeps only Bubble type foto_canhoto with the given signature verdict; other types (foto_mc, documentos) never enter.
A plain JSON list (rows with url, no type) is taken whole: the user already validated each row with that verdict.
Downloads from *.cdn.bubble.io into gold/inbox_validado/, adds them with evalkit and labels them with
triagem.fonte = "validacao_usuario". Refuses to flip a label the user already gave in the labeler.
Dry-run by default; --aplicar writes (manifest backup first). Prints counts only.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rotulagem.importar_triagem import MANIFEST, read_jsonl, write_with_backup  # noqa: E402

INBOX = ROOT / "gold" / "inbox_validado"
SOURCE = "bubble_foto_canhoto"
KEPT_TYPE = "foto_canhoto"
FONTE = "validacao_usuario"
ALLOWED_HOST_SUFFIX = "cdn.bubble.io"
MAX_BYTES = 15_000_000
TIMEOUT_SECONDS = 30
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def selected(paths: list[Path], verdict: bool) -> dict[str, dict]:
    rows = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):  # plain list: every row already carries the user's verdict, keyed by url
            rows.update((row["url"], row) for row in data if row.get("url"))
            continue
        for row in data["resultados"]:
            if row.get("type") == KEPT_TYPE and row.get("assinado") is verdict:
                rows[row["id"]] = row
    return rows


def is_allowed(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host == ALLOWED_HOST_SUFFIX or host.endswith("." + ALLOWED_HOST_SUFFIX)


def fetch(row: dict) -> str | None:
    url = "https:" + row["url"] if row["url"].startswith("//") else row["url"]
    if not is_allowed(url):
        return None
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            content_type, body = response.headers.get_content_type(), response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, OSError):
        return None
    extension = EXTENSIONS.get(content_type)
    if extension is None or len(body) > MAX_BYTES:
        return None
    digest = hashlib.sha256(body).hexdigest()
    (INBOX / f"{digest[:16]}{extension}").write_bytes(body)
    return digest


def label(manifest: list[dict], digests: set[str], verdict: bool) -> Counter:
    stats: Counter = Counter()
    for record in manifest:
        if record["sha256"] not in digests:
            continue
        current = (record.get("assinatura") or {}).get("possui_assinatura")
        if record.get("status") == "labeled" and current is not verdict:
            stats["conflito_mantido"] += 1  # the labeler verdict wins; never flipped silently
            continue
        record.update(status="labeled", tipo_documento="canhoto", assinatura={"possui_assinatura": verdict},
                      rotulo_parcial=True, triagem={"fonte": FONTE})
        stats["rotulado"] += 1
    return stats


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("arquivos", nargs="+", type=Path)
    parser.add_argument("--assinado", choices=("true", "false"), required=True)
    parser.add_argument("--aplicar", action="store_true")
    args = parser.parse_args()
    verdict = args.assinado == "true"
    rows = selected(args.arquivos, verdict)
    print(json.dumps({"selecionados": len(rows), "veredito_assinado": verdict}))
    if not args.aplicar:
        print("simulação. Rode com --aplicar para baixar e rotular.")
        return 0
    INBOX.mkdir(parents=True, exist_ok=True)
    digests = {d for d in (fetch(row) for row in rows.values()) if d}
    subprocess.run([sys.executable, "-m", "evalkit", "add", str(INBOX), "--source", SOURCE], check=True, cwd=ROOT)
    manifest = read_jsonl(MANIFEST)
    stats = label(manifest, digests, verdict)
    print(json.dumps({"baixados": len(digests), "falha_download": len(rows) - len(digests), **stats}))
    print(f"backup: {write_with_backup(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
