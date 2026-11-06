"""Pack gold records for the OCR + ink extraction on 192.168.0.160.

Default: labeled canhotos. With --fonte: every record of that source not yet in gold/ocr.
Writes <saida> (.tar.gz) with gold/manifest.jsonl (selected records), their images, scripts/ocr/extract_ocr.py
and export_rows.py. Prints counts only.
"""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "gold"
JOB = "ocr-job"


def selected(fonte: str | None) -> list[dict]:
    lines = (GOLD / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if fonte:
        return [r for r in records if r.get("source") == fonte and not (GOLD / "ocr" / f"{r['sha256']}.json").exists()]
    return [r for r in records if r.get("status") == "labeled" and r.get("tipo_documento") == "canhoto"]


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fonte")
    parser.add_argument("--saida", type=Path, required=True)
    args = parser.parse_args()
    records = selected(args.fonte)
    missing = [r for r in records if not (GOLD / r["image"]).is_file()]
    if missing or not records:
        print(json.dumps({"erro": "imagens ausentes ou nada a enviar", "ausentes": len(missing), "registros": len(records)}))
        return 1
    manifest = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    with tarfile.open(args.saida, "w:gz") as tar:
        add_bytes(tar, f"{JOB}/gold/manifest.jsonl", manifest.encode("utf-8"))
        tar.add(ROOT / "scripts" / "ocr" / "extract_ocr.py", f"{JOB}/extract_ocr.py")
        tar.add(ROOT / "scripts" / "treino" / "export_rows.py", f"{JOB}/export_rows.py")
        for record in records:
            tar.add(GOLD / record["image"], f"{JOB}/gold/{record['image']}")
    print(json.dumps({"registros": len(records), "mb": round(args.saida.stat().st_size / 1024**2, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
