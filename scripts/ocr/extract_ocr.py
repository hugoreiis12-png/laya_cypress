"""Run the production vision stage over every gold image and cache the evidence as gold/ocr/<sha256>.json.

Must run where Tesseract and libzbar exist, i.e. inside the laya-cypress image on 192.168.0.160:
  python /srv/scripts/ocr/extract_ocr.py --gold /srv/gold
Idempotent: cached records are skipped. Failures are cached too, so they are visible in the build report.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from service.app.vision.analyze import analyze
from service.app.vision.image_io import ImageError

OCR_DIR = "ocr"
MANIFEST = "manifest.jsonl"


def load_manifest(gold: Path) -> list[dict[str, Any]]:
    lines = (gold / MANIFEST).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def cache_one(gold: Path, record: dict[str, Any]) -> str:
    target = gold / OCR_DIR / f"{record['sha256']}.json"
    if target.exists():
        return "cache"
    try:
        evidence = analyze((gold / record["image"]).read_bytes())
        payload: dict[str, Any] = {"ok": True, **dataclasses.asdict(evidence)}
        status = "ok"
    except ImageError as exc:
        payload, status = {"ok": False, "erro": str(exc)}, "erro"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=Path("gold"))
    args = parser.parse_args(argv)
    (args.gold / OCR_DIR).mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for record in load_manifest(args.gold):
        status = cache_one(args.gold, record)
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
