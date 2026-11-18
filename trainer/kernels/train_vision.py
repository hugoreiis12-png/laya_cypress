"""Kaggle kernel: trains a YOLOv8 classifier on the receiver strips and writes metricas.json and pesos.onnx.

The frozen test is measured on the exported ONNX with the service's own input transform, so the numbers are
those of the model the service runs. Runs on Kaggle, never inside the trainer container.
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = ("ultralytics==8.3.40", "onnx==1.18.0", "onnxscript", "onnxruntime==1.20.1")  # Python 3.13: onnx 1.17 has no wheel; the torch there exports through onnxscript
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402

INPUT = min(Path("/kaggle/input").rglob("manifest.json"), key=lambda path: len(path.parts)).parent  # Kaggle nests mounts under datasets/<owner>/
WORK = Path("/kaggle/working")
DATA = WORK / "dados"
EPOCHS = 30
CLASSIFIER_SIZE = 320
PIXEL_SCALE = 255.0
VALIDATION_SHARE = 0.1
SEED = 7
SIGNED = "assinado"
SAFETY_KEYS = {"nao_assinado": "nao_assinado_liberado", "carimbo_somente": "carimbo_liberado"}


def classifier_input(image: np.ndarray) -> np.ndarray:
    """Copy of service/app/vision/strip.py:classifier_input; a test keeps the two identical."""
    height, width = image.shape[:2]
    scale = CLASSIFIER_SIZE / min(height, width)
    size = (max(CLASSIFIER_SIZE, round(width * scale)), max(CLASSIFIER_SIZE, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    top, left = (resized.shape[0] - CLASSIFIER_SIZE) // 2, (resized.shape[1] - CLASSIFIER_SIZE) // 2
    square = resized[top:top + CLASSIFIER_SIZE, left:left + CLASSIFIER_SIZE]
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / PIXEL_SCALE
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


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


def probabilities(session: onnxruntime.InferenceSession, path: Path) -> np.ndarray:
    feed = {session.get_inputs()[0].name: classifier_input(cv2.imread(str(path), cv2.IMREAD_COLOR))}
    return np.asarray(session.run(None, feed)[0][0], dtype=np.float64)


def evaluate(session: onnxruntime.InferenceSession, classes: list[str]) -> dict:
    wrong: dict[str, int] = {}
    total: dict[str, int] = {}
    safety = {key: 0 for key in SAFETY_KEYS.values()}
    confidences = []
    for folder in (DATA / "test").iterdir():
        for path in folder.glob("*.jpg"):
            probs = probabilities(session, path)
            predicted = classes[int(probs.argmax())]
            confidences.append(float(probs.max()))
            total[folder.name] = total.get(folder.name, 0) + 1
            wrong[folder.name] = wrong.get(folder.name, 0) + (predicted != folder.name)
            if folder.name in SAFETY_KEYS and predicted == SIGNED:
                safety[SAFETY_KEYS[folder.name]] += 1
    count = sum(total.values())
    return {"teste": json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))["teste"],
            "erro": sum(wrong.values()) / count if count else 1.0,
            "por_classe": {name: wrong.get(name, 0) / size for name, size in total.items()},
            "seguranca": safety, "confianca": sum(confidences) / len(confidences) if confidences else 0.0,
            "amostras_teste": count, "classes": classes}


def main() -> None:
    settings.update({"raytune": False})  # Kaggle's ray dropped the private API ultralytics 8.3.40 calls
    split_validation()
    model = YOLO("yolov8n-cls.pt")
    model.train(data=str(DATA), epochs=EPOCHS, imgsz=CLASSIFIER_SIZE, project=str(WORK / "runs"), name="treino")
    best = YOLO(str(model.trainer.best))
    classes = [best.names[index] for index in range(len(best.names))]
    exported = Path(best.export(format="onnx", imgsz=CLASSIFIER_SIZE))
    session = onnxruntime.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
    (WORK / "metricas.json").write_text(json.dumps(evaluate(session, classes)), encoding="utf-8")
    shutil.copyfile(exported, WORK / "pesos.onnx")


if __name__ == "__main__":
    main()
