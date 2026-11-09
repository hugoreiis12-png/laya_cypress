"""Builds the versioned training packages: only anonymized strips and text, split by the frozen test rule."""
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .anonymizer import find_residual
from .jsonio import write_json
from .split import is_test

PSEUDONYM = re.compile(r"^[0-9a-f]{16}$")
TRAIN, TEST = "train", "test"


class PackageError(RuntimeError):
    """The package would leak or is incomplete; nothing is written."""


@dataclass(frozen=True)
class StripItem:
    item_id: str
    label: str
    jpeg: bytes


@dataclass(frozen=True)
class TextItem:
    item_id: str
    label: str
    text: str


def _check_ids(ids: Sequence[str]) -> None:
    raw = [item_id for item_id in ids if not PSEUDONYM.match(item_id)]
    if raw:
        raise PackageError("id sem pseudônimo no pacote: use anonymizer.pseudonym antes de empacotar")


def _digest(files: dict[str, str]) -> str:
    joined = "\n".join(f"{name}:{files[name]}" for name in sorted(files))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _manifest(name: str, kind: str, percent: int, counts: dict[str, Any], files: dict[str, str], **extra: Any) -> dict[str, Any]:
    return {"pacote": name, "tipo": kind, "teste": f"teste-p{percent}", "contagens": counts,
            "sha256": _digest(files), "anonimizado": True, **extra}


def write_vision_package(directory: Path, name: str, items: Sequence[StripItem], percent: int) -> dict[str, Any]:
    _check_ids([item.item_id for item in items])
    counts: dict[str, int] = {}
    files: dict[str, str] = {}
    for item in items:
        split = TEST if is_test(item.item_id, percent) else TRAIN
        target = directory / split / item.label / f"{item.item_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.jpeg)
        files[f"{split}/{item.label}/{item.item_id}"] = hashlib.sha256(item.jpeg).hexdigest()
        counts[f"{split}/{item.label}"] = counts.get(f"{split}/{item.label}", 0) + 1
    manifest = _manifest(name, "visao", percent, counts, files)
    write_json(directory / "manifest.json", manifest)
    return manifest


def _check_texts(items: Sequence[TextItem], names_removed: bool) -> None:
    if not names_removed:
        raise PackageError("pacote do Laya exige a remoção de nomes de pessoas")
    leaks = sorted({label for item in items for label in find_residual(item.text)})
    if leaks:
        raise PackageError(f"texto com dado pessoal remanescente: {', '.join(leaks)}")


def write_text_package(directory: Path, name: str, items: Sequence[TextItem], percent: int,
                       names_removed: bool) -> dict[str, Any]:
    _check_ids([item.item_id for item in items])
    _check_texts(items, names_removed)
    lines: dict[str, list[str]] = {TRAIN: [], TEST: []}
    for item in items:
        row = {"id": item.item_id, "texto": item.text, "rotulo": item.label}
        lines[TEST if is_test(item.item_id, percent) else TRAIN].append(json.dumps(row, ensure_ascii=False))
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in lines.items():
        content = "\n".join(rows)
        (directory / f"{split}.jsonl").write_text(content, encoding="utf-8")
        files[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = _manifest(name, "laya", percent, {split: len(rows) for split, rows in lines.items()}, files,
                         nomes_removidos=names_removed)
    write_json(directory / "manifest.json", manifest)
    return manifest
