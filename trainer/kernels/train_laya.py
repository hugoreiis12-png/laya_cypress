"""Kaggle kernel: fine-tunes the Laya encoder to classify the document type from anonymized OCR text.

Runs on Kaggle, never inside the trainer container. Writes metricas.json and pesos.bin (the fine-tuned weights);
turning them into the serving package of the hf-encoder is a separate step. Not yet exercised on the real service.
"""
import json
import random
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_REPO = "convaiinnovations/laya-multilingual"
BASE_REVISION = "b4a904d1a2a54c822b829e24291d4b8f280fe43e"
INPUT = next(Path("/kaggle/input").iterdir())
WORK = Path("/kaggle/working")
MAX_LENGTH, EPOCHS, BATCH_SIZE, LEARNING_RATE = 512, 4, 16, 3e-5
VALIDATION_SHARE, SEED = 0.1, 7
RELEASED_TYPE = "canhoto"  # the type that lets a document be released: taking another type for it is the unsafe error


def load_rows(name: str) -> list[dict]:
    lines = (INPUT / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def split_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The frozen test stays untouched; validation comes out of the training rows only."""
    shuffled = random.Random(SEED).sample(rows, len(rows))
    cut = max(1, int(len(rows) * VALIDATION_SHARE))
    return shuffled[cut:], shuffled[:cut]


def predict(model, tokenizer, rows: list[dict], labels: list[str]) -> list[tuple[str, float]]:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            texts = [row["texto"] for row in rows[start:start + BATCH_SIZE]]
            batch = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            probabilities = torch.softmax(model(**batch.to(model.device)).logits, dim=-1)
            confidence, index = probabilities.max(dim=-1)
            results += [(labels[i], float(c)) for i, c in zip(index.tolist(), confidence.tolist())]
    return results


def error_rate(model, tokenizer, rows: list[dict], labels: list[str]) -> float:
    if not rows:
        return 1.0
    wrong = sum(guess != row["rotulo"] for (guess, _), row in zip(predict(model, tokenizer, rows, labels), rows))
    return wrong / len(rows)


def train_epoch(model, tokenizer, rows: list[dict], labels: list[str], optimizer) -> None:
    model.train()
    order = random.Random(SEED).sample(rows, len(rows))
    for start in range(0, len(order), BATCH_SIZE):
        chunk = order[start:start + BATCH_SIZE]
        batch = tokenizer([row["texto"] for row in chunk], truncation=True, max_length=MAX_LENGTH,
                          padding=True, return_tensors="pt").to(model.device)
        targets = torch.tensor([labels.index(row["rotulo"]) for row in chunk], device=model.device)
        loss = model(**batch, labels=targets).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, tokenizer, rows: list[dict], labels: list[str]) -> dict:
    guesses = predict(model, tokenizer, rows, labels)
    total: dict[str, int] = {}
    wrong: dict[str, int] = {}
    safety = {f"{label}_como_{RELEASED_TYPE}": 0 for label in labels if label != RELEASED_TYPE}
    for (guess, _), row in zip(guesses, rows):
        label = row["rotulo"]
        total[label] = total.get(label, 0) + 1
        wrong[label] = wrong.get(label, 0) + (guess != label)
        if label != RELEASED_TYPE and guess == RELEASED_TYPE:
            safety[f"{label}_como_{RELEASED_TYPE}"] += 1
    count = len(rows)
    return {"teste": json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))["teste"],
            "erro": sum(wrong.values()) / count if count else 1.0,
            "por_classe": {name: wrong.get(name, 0) / size for name, size in total.items()},
            "seguranca": safety, "confianca": sum(c for _, c in guesses) / count if count else 0.0}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    train, validation = split_validation(load_rows("train.jsonl"))
    labels = sorted({row["rotulo"] for row in train})
    base = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION))
    tokenizer = AutoTokenizer.from_pretrained(base / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(base / "encoder", num_labels=len(labels))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_error = 2.0
    for _ in range(EPOCHS):
        train_epoch(model, tokenizer, train, labels, optimizer)
        error = error_rate(model, tokenizer, validation, labels)
        if error < best_error:
            best_error = error
            model.save_pretrained(WORK / "melhor")
    best = AutoModelForSequenceClassification.from_pretrained(WORK / "melhor").to(model.device)
    (WORK / "metricas.json").write_text(json.dumps(evaluate(best, tokenizer, load_rows("test.jsonl"), labels)), encoding="utf-8")
    shutil.copyfile(WORK / "melhor" / "model.safetensors", WORK / "pesos.bin")


if __name__ == "__main__":
    main()
