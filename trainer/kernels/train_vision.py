"""Kaggle kernel: trains a YOLOv8 classifier on the receiver strips and writes metricas.json and pesos.bin.

Runs on Kaggle, never inside the trainer container. Not yet exercised on the real service.
"""
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

INPUT = next(Path("/kaggle/input").iterdir())
WORK = Path("/kaggle/working")
DATA = WORK / "dados"
EPOCHS = 30
IMAGE_SIZE = 320
VALIDATION_SHARE = 0.1
SEED = 7
UNSIGNED_LABELS = ("nao_assinado", "carimbo_somente")
SIGNED = "assinado"
SAFETY_KEYS = {"nao_assinado": "nao_assinado_liberado", "carimbo_somente": "carimbo_liberado"}


def split_validation() -> None:
    """The frozen test stays untouched; validation comes out of the training images only."""
    shutil.copytree(INPUT / "train", DATA / "train")
    shutil.copytree(INPUT / "test", DATA / "test")
    random.seed(SEED)
    for folder in (DATA / "train").iterdir():
        files = sorted(folder.glob("*.jpg"))
        for path in random.sample(files, max(1, int(len(files) * VALIDATION_SHARE))):
            target = DATA / "val" / folder.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target / path.name)


def evaluate(model: YOLO) -> dict:
    wrong: dict[str, int] = {}
    total: dict[str, int] = {}
    safety = {key: 0 for key in SAFETY_KEYS.values()}
    confidences = []
    for folder in (DATA / "test").iterdir():
        for path in folder.glob("*.jpg"):
            result = model.predict(str(path), imgsz=IMAGE_SIZE, verbose=False)[0]
            predicted, confidence = result.names[result.probs.top1], float(result.probs.top1conf)
            confidences.append(confidence)
            total[folder.name] = total.get(folder.name, 0) + 1
            wrong[folder.name] = wrong.get(folder.name, 0) + (predicted != folder.name)
            if folder.name in SAFETY_KEYS and predicted == SIGNED:
                safety[SAFETY_KEYS[folder.name]] += 1
    count = sum(total.values())
    return {"teste": INPUT.name, "erro": sum(wrong.values()) / count if count else 1.0,
            "por_classe": {name: wrong.get(name, 0) / size for name, size in total.items()},
            "seguranca": safety, "confianca": sum(confidences) / len(confidences) if confidences else 0.0}


def main() -> None:
    split_validation()
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(DATA), epochs=EPOCHS, imgsz=IMAGE_SIZE, project=str(WORK / "runs"), name="treino")
    best = YOLO(str(model.trainer.best))
    (WORK / "metricas.json").write_text(json.dumps(evaluate(best)), encoding="utf-8")
    shutil.copyfile(model.trainer.best, WORK / "pesos.bin")


if __name__ == "__main__":
    main()
