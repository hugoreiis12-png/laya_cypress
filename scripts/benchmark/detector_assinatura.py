"""Spike: can an open-source signature detector separate signed from unsigned canhotos better than the ink rule?

Population = the Laya rounds' rows (labeled canhotos with non-empty cached OCR); split = evalkit (sha256[:8] % 5).
  pontuar --modelo yolos|rfdetr|tinta   writes scores-<modelo>.jsonl (sha, label, split, score; no content)
  pontuar --modelo rfdetr --quarentena  scores the canhotos awaiting review into scores-<modelo>-quarentena.jsonl
  relatorio                             threshold on dev (zero unsigned released), test opened once; labels re-read
Prints counts only.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from service.app.vision.types import InkFeatures  # noqa: E402

GOLD = ROOT / "gold"
OUT = ROOT / "_itsfine_runs" / "spike-assinatura-visual"
YOLOS_ID = "mdefrance/yolos-tiny-signature-detection"
RFDETR_ID = "Mo-Awadalla/legaldocuman-rfdetr"
RFDETR_WEIGHTS = "checkpoint_best_total.pth"
MIN_INK_PIXELS, MIN_COMPONENTS, MIN_RELATIVE_INK = 400, 1, 3.0  # production defaults (service/app/config.py)
MIN_COVERAGE = 0.7
QUARANTINE = "pendente_revisao"


def canhotos(status: str) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for line in (GOLD / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line) if line.strip() else {}
        if record.get("status") != status or record.get("tipo_documento") != "canhoto":
            continue
        cache = GOLD / "ocr" / f"{record['sha256']}.json"
        yield record, json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}


def as_row(record: dict[str, Any], cached: dict[str, Any], signed: bool | None) -> dict[str, Any]:
    split = "teste" if int(record["sha256"][:8], 16) % 5 == 0 else "dev"
    return {"sha": record["sha256"], "image": GOLD / record["image"], "signed": signed, "split": split, "cached": cached}


def population() -> list[dict[str, Any]]:
    rows = []
    for record, cached in canhotos("labeled"):
        signed = (record.get("assinatura") or {}).get("possui_assinatura")
        if isinstance(signed, bool) and cached.get("ok") and str(cached.get("text", "")).strip():
            rows.append(as_row(record, cached, signed))
    return rows


def quarantine() -> list[dict[str, Any]]:
    return [as_row(record, cached, None) for record, cached in canhotos(QUARANTINE)]  # rotation 0 without OCR cache


def upright(row: dict[str, Any]):
    import cv2
    from PIL import Image

    from service.app.vision.image_io import decode, rotate

    image = rotate(decode(row["image"].read_bytes()), int(row["cached"].get("rotation", 0)))
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def ink_scorer() -> Callable[[dict[str, Any]], float]:
    def score(row: dict[str, Any]) -> float:
        ink = InkFeatures(**row["cached"]["ink"])
        if not ink.region_found:
            return 0.0
        margin = min(ink.ink_pixels / MIN_INK_PIXELS, ink.component_count / MIN_COMPONENTS,
                     ink.density_relative / MIN_RELATIVE_INK)
        return margin / (margin + 1)  # same monotone score as service/app/signature.py
    return score


def yolos_scorer() -> Callable[[dict[str, Any]], float]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(YOLOS_ID)
    model = AutoModelForObjectDetection.from_pretrained(YOLOS_ID).eval()
    wanted = {i for i, name in model.config.id2label.items() if "sign" in name.lower()} or set(model.config.id2label)

    def score(row: dict[str, Any]) -> float:
        image = upright(row)
        with torch.no_grad():
            outputs = model(**processor(images=image, return_tensors="pt"))
        found = processor.post_process_object_detection(outputs, threshold=0.0,
                                                         target_sizes=[image.size[::-1]])[0]
        return max((float(s) for s, label in zip(found["scores"], found["labels"]) if int(label) in wanted),
                   default=0.0)
    return score


def rfdetr_scorer() -> Callable[[dict[str, Any]], float]:
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=hf_hub_download(RFDETR_ID, RFDETR_WEIGHTS))  # rfdetr wants a local file

    def score(row: dict[str, Any]) -> float:
        detections = model.predict(upright(row), threshold=0.01)
        return float(max(detections.confidence, default=0.0))
    return score


SCORERS = {"tinta": ink_scorer, "yolos": yolos_scorer, "rfdetr": rfdetr_scorer}


def score_all(name: str, quarantined: bool) -> dict[str, int]:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / (f"scores-{name}-quarentena.jsonl" if quarantined else f"scores-{name}.jsonl")
    done = {json.loads(line)["sha"] for line in target.read_text(encoding="utf-8").splitlines()} if target.exists() else set()
    scorer, stats = SCORERS[name](), {"pontuados": 0, "cache": 0, "erro": 0}
    with target.open("a", encoding="utf-8") as handle:
        for row in quarantine() if quarantined else population():
            if row["sha"] in done:
                stats["cache"] += 1
                continue
            try:
                value = scorer(row)
            except Exception as exc:  # noqa: BLE001 - one unreadable image must not stop the spike
                stats["erro"] += 1
                print(json.dumps({"erro": type(exc).__name__}), file=sys.stderr)
                continue
            handle.write(json.dumps({"sha": row["sha"], "signed": row["signed"], "split": row["split"],
                                     "score": value}) + "\n")
            stats["pontuados"] += 1
    return stats


def auc(rows: list[dict[str, Any]]) -> float | None:
    positives = [r["score"] for r in rows if r["signed"]]
    negatives = [r["score"] for r in rows if not r["signed"]]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


def operate(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    released = [r for r in rows if r["score"] >= threshold]
    signed = sum(r["signed"] for r in rows)
    return {"n": len(rows), "assinados": signed, "nao_assinados": len(rows) - signed,
            "fn_liberados": sum(not r["signed"] for r in released),
            "assinados_liberados": sum(r["signed"] for r in released),
            "cobertura_automatica": round(len(released) / len(rows), 4) if rows else None,
            "revisar": len(rows) - len(released)}


def report_one(name: str, labels: dict[str, bool]) -> dict[str, Any] | None:
    path = OUT / f"scores-{name}.jsonl"
    if not path.exists():
        return None
    scored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [{**r, "signed": labels[r["sha"]]} for r in scored if r["sha"] in labels]  # current labels, not cached ones
    dev, test = [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "teste"]
    unsigned = [r["score"] for r in dev if not r["signed"]]
    threshold = max(unsigned) + 1e-9 if unsigned else 0.0  # strictest: no dev unsigned is released
    result_test = operate(test, threshold)
    return {"modelo": name, "limiar": round(threshold, 6), "auc_dev": auc(dev), "auc_teste": auc(test),
            "dev": operate(dev, threshold), "teste": result_test,
            "passa": result_test["fn_liberados"] == 0 and (result_test["cobertura_automatica"] or 0) >= MIN_COVERAGE}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="comando", required=True)
    score_parser = commands.add_parser("pontuar")
    score_parser.add_argument("--modelo", choices=sorted(SCORERS), required=True)
    score_parser.add_argument("--quarentena", action="store_true")
    commands.add_parser("relatorio")
    args = parser.parse_args()
    if args.comando == "pontuar":
        print(json.dumps({"modelo": args.modelo, **score_all(args.modelo, args.quarentena)}))
        return 0
    labels = {row["sha"]: row["signed"] for row in population()}
    summary = [r for r in (report_one(name, labels) for name in SCORERS) if r]
    (OUT / "resumo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())