"""Sample foto_canhoto (WhatsApp photos in Bubble, read through the bubble MCP) and download them.

Stratified by month: BLOCKS_PER_MONTH random blocks of BLOCK consecutive records per month.
Downloads only from *.cdn.bubble.io into gold/inbox_foto_canhoto/ and logs sha256 per file.
Idempotent: files already downloaded are skipped. Prints counts only.
Next step: python -m evalkit add gold/inbox_foto_canhoto --source bubble_foto_canhoto
"""
import argparse
import hashlib
import json
import random
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bubble_mcp import Client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT / "gold" / "inbox_foto_canhoto"
LOG = INBOX / "_download_log.jsonl"
TYPE = "foto_canhoto"
MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11")
BLOCK = 20
ALLOWED_HOST_SUFFIX = "cdn.bubble.io"
MAX_BYTES = 15_000_000
TIMEOUT_SECONDS = 30
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def payload(result: dict) -> dict:
    text = result.get("texto", "")
    return json.loads(text[text.index("\n") + 1:])


def month_query(client: Client, start: str, end: str, cursor: int, limit: int) -> dict:
    constraints = [{"key": "Created Date", "constraint_type": "greater than", "value": f"{start}-01T00:00:00Z"},
                   {"key": "Created Date", "constraint_type": "less than", "value": f"{end}-01T00:00:00Z"}]
    return payload(client.tool("bubble_query", {"type": TYPE, "action": "list", "cursor": cursor, "limit": limit,
                                                "constraints": constraints, "fields": ["imagem"],
                                                "exclude_remaining": False}))


def sample_month(client: Client, start: str, end: str, target: int, rng: random.Random) -> list[dict]:
    total = month_query(client, start, end, 0, 1).get("remaining", 0) + 1
    if total <= 1:
        return []
    blocks = min(max(1, target // BLOCK), max(1, total // BLOCK))
    starts = sorted(rng.sample(range(0, max(1, total - BLOCK)), blocks))
    picked = []
    for cursor in starts:
        picked += month_query(client, start, end, cursor, BLOCK)["records"]
    return picked


def normalized(url: str) -> str:
    return "https:" + url if url.startswith("//") else url


def is_allowed(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host == ALLOWED_HOST_SUFFIX or host.endswith("." + ALLOWED_HOST_SUFFIX)


def fetch(record: dict) -> dict:
    url = normalized(record.get("imagem") or "")
    row = {"bubble_id": record["_id"]}
    if not is_allowed(url):
        return {**row, "status": "erro", "motivo": "host fora do CDN do Bubble"}
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            content_type, body = response.headers.get_content_type(), response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, OSError) as exc:
        return {**row, "status": "erro", "motivo": type(exc).__name__}
    extension = EXTENSIONS.get(content_type)
    if extension is None or len(body) > MAX_BYTES:
        return {**row, "status": "ignorado", "motivo": content_type}
    digest = hashlib.sha256(body).hexdigest()
    (INBOX / f"{digest[:16]}{extension}").write_bytes(body)
    return {**row, "status": "ok", "sha256": digest, "bytes": len(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()
    INBOX.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["bubble_id"] for line in LOG.read_text(encoding="utf-8").splitlines()} if LOG.exists() else set()
    client, rng = Client(), random.Random(args.semente)
    per_month = args.total // (len(MONTHS) - 1)
    counts: dict[str, int] = {}
    with LOG.open("a", encoding="utf-8") as log:
        for start, end in zip(MONTHS, MONTHS[1:]):
            for record in sample_month(client, start, end, per_month, rng):
                if record["_id"] in done:
                    continue
                row = fetch(record)
                done.add(record["_id"])
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(json.dumps({"mes": start, **counts}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
