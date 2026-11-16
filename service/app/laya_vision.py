"""Laya's own signature classifier: the version the trainer put in use, read from its registry and run in ONNX."""
import hashlib
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .vision.strip import STRIP_JPEG_QUALITY, classifier_input, upright_top_half

log = logging.getLogger(__name__)
KIND = "visao"
SIGNED = "assinado"
REGISTRY_FILE, WEIGHTS_DIR = "versoes.json", "pesos"
ONNX_SUFFIX = ".onnx"
HASH_CHUNK = 1 << 20
CONFIDENCE_DIGITS = 4
PROBABILITY_TOLERANCE = 1e-3


class VisionModelError(RuntimeError):
    """The registry, the weights or the model run is unusable."""


class Session(Protocol):
    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[Any]: ...


@dataclass(frozen=True)
class LoadedVersion:
    version: str
    classes: tuple[str, ...]
    ready: bool
    session: Session
    input_name: str


def _onnx_session(path: Path) -> tuple[Session, str]:
    import onnxruntime  # only a service with a registry pays for the import

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1  # OCR already uses the CPUs of this container
    session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_record(registry: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The same rule as the trainer's Registry.current: the last shadow version, else the baseline."""
    history = (registry.get("shadow") or {}).get(KIND) or []
    version = history[-1] if history else (registry.get("baseline") or {}).get(KIND)
    record = (registry.get("versions") or {}).get(version) if version else None
    return (version, record) if record else None


def _strip_input(data: bytes, rotation: int) -> np.ndarray:
    """The strip exactly as the trainer cuts and stores it, JPEG round trip included, then the model input."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise VisionModelError("imagem ilegível")
    done, encoded = cv2.imencode(".jpg", upright_top_half(image, rotation), [cv2.IMWRITE_JPEG_QUALITY, STRIP_JPEG_QUALITY])
    if not done:
        raise VisionModelError("falha ao codificar a faixa")
    return classifier_input(cv2.imdecode(encoded, cv2.IMREAD_COLOR))


def _probabilities(raw: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if abs(values.sum() - 1.0) <= PROBABILITY_TOLERANCE:
        return values
    exp = np.exp(values - values.max())
    return exp / exp.sum()


class LayaVision:
    def __init__(self, directory: str, open_session: Callable[[Path], tuple[Session, str]] = _onnx_session) -> None:
        self._directory = Path(directory)
        self._open = open_session
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._loaded: LoadedVersion | None = None

    def _load(self) -> LoadedVersion | None:
        registry = json.loads((self._directory / REGISTRY_FILE).read_text(encoding="utf-8"))
        found = current_record(registry)
        if found is None or not str(found[1].get("file", "")).endswith(ONNX_SUFFIX) or not found[1].get("classes"):
            return None
        version, record = found
        weights = self._directory / WEIGHTS_DIR / record["file"]
        if _sha256(weights) != record["sha256"]:
            raise VisionModelError(f"pesos da versão {version} não conferem com o registro")
        session, input_name = self._open(weights)
        return LoadedVersion(version, tuple(record["classes"]), record.get("pronta") is True, session, input_name)

    def _current(self) -> LoadedVersion | None:
        """Reloads only when the registry file changes, so a promotion or a rollback reaches the service by itself."""
        status = (self._directory / REGISTRY_FILE).stat()
        stamp = (status.st_mtime_ns, status.st_size)
        with self._lock:
            if stamp == self._stamp:
                return self._loaded
            self._stamp, self._loaded = stamp, None  # a broken version is not retried until the registry changes
            try:
                self._loaded = self._load()
            except Exception as exc:  # noqa: BLE001 - onnxruntime raises its own types; any of them disables the model
                log.warning("Laya visão: versão do registro não carregada: %s", exc)
            log.info("Laya visão: versão %s em uso", self._loaded.version if self._loaded else "nenhuma")
            return self._loaded

    def observe(self, data: bytes, rotation: int) -> dict[str, Any]:
        """Never raises: the classifier failing only means the judge decides, as before."""
        try:
            loaded = self._current()
            if loaded is None:
                return {"erro": "sem_versao"}
            probs = _probabilities(loaded.session.run(None, {loaded.input_name: _strip_input(data, rotation)})[0])
        except FileNotFoundError:
            return {"erro": "sem_registro"}  # the trainer has not registered any version yet: expected, not logged
        except Exception as exc:  # noqa: BLE001 - same boundary as the load: the judge decides instead
            log.warning("Laya visão indisponível: %s", exc)
            return {"erro": type(exc).__name__}
        if len(probs) != len(loaded.classes):
            return {"erro": "classes_divergentes", "versao": loaded.version}
        best = int(probs.argmax())
        return {"versao": loaded.version, "rotulo": loaded.classes[best],
                "confianca": round(float(probs[best]), CONFIDENCE_DIGITS), "pronta": loaded.ready}
