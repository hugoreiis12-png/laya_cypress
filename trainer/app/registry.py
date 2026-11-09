"""Version registry: weights stored by hash outside Git, a shadow history per kind and rollback."""
import hashlib
import shutil
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .metrics import Metrics

CANDIDATE, BASELINE, SHADOW, REVERTED = "candidata", "referencia", "sombra", "revertida"
HASH_CHUNK = 1 << 20


class RegistryError(RuntimeError):
    """The version is unknown or its weights are missing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Registry:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._file = directory / "versoes.json"

    def _load(self) -> dict[str, Any]:
        return read_json(self._file, {"versions": {}, "shadow": {}, "baseline": {}})

    def next_version(self, kind: str) -> str:
        count = sum(1 for record in self._load()["versions"].values() if record["kind"] == kind)
        return f"{kind}-{count + 1:04d}"

    def register(self, kind: str, version: str, weights: Path, metrics: Metrics) -> None:
        digest = _sha256(weights)
        stored = self._directory / "pesos" / f"{digest}.bin"
        stored.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights, stored)
        data = self._load()
        data["versions"][version] = {"kind": kind, "sha256": digest, "status": CANDIDATE, "metrics": metrics.to_dict()}
        write_json(self._file, data)

    def _record(self, data: dict[str, Any], version: str) -> dict[str, Any]:
        if version not in data["versions"]:
            raise RegistryError(f"versão desconhecida: {version}")
        return data["versions"][version]

    def mark_baseline(self, kind: str, version: str) -> None:
        data = self._load()
        self._record(data, version)["status"] = BASELINE
        data["baseline"][kind] = version
        write_json(self._file, data)

    def promote_to_shadow(self, kind: str, version: str) -> None:
        data = self._load()
        self._record(data, version)["status"] = SHADOW
        data["shadow"].setdefault(kind, []).append(version)
        write_json(self._file, data)

    def current(self, kind: str) -> str | None:
        data = self._load()
        history = data["shadow"].get(kind) or []
        return history[-1] if history else data["baseline"].get(kind)

    def metrics_of(self, version: str | None) -> Metrics | None:
        if version is None:
            return None
        return Metrics.from_dict(self._record(self._load(), version)["metrics"])

    def rollback(self, kind: str) -> str | None:
        data = self._load()
        history = data["shadow"].get(kind) or []
        if not history:
            return None
        self._record(data, history.pop())["status"] = REVERTED
        write_json(self._file, data)
        return history[-1] if history else data["baseline"].get(kind)
