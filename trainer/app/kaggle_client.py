"""Kaggle gateway: private dataset, one GPU kernel, output download and dataset deletion.

Written against the official `kaggle` package but not yet exercised against the real service.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Protocol

RUNNING, COMPLETE, FAILED = "rodando", "completo", "falhou"
QUOTA_HINTS = ("quota", "429", "too many requests", "limit")
STATUS_WORDS = {"complete": COMPLETE, "error": FAILED, "cancel": FAILED, "queued": RUNNING, "running": RUNNING}
LICENSE = "CC0-1.0"
SCRIPT_NAME = "treino.py"


class KaggleError(RuntimeError):
    """The Kaggle call failed."""


class QuotaExceeded(KaggleError):
    """The GPU quota is spent; the scheduler waits for the next window."""


class JobGateway(Protocol):
    def create_dataset(self, folder: Path, slug: str) -> None: ...
    def push_kernel(self, folder: Path, slug: str, script: Path) -> str: ...
    def status(self, kernel: str) -> str: ...
    def download(self, kernel: str, destination: Path) -> None: ...
    def delete_dataset(self, slug: str) -> None: ...


def dataset_metadata(username: str, slug: str) -> dict[str, Any]:
    return {"title": slug, "id": f"{username}/{slug}", "licenses": [{"name": LICENSE}]}


def kernel_metadata(username: str, slug: str) -> dict[str, Any]:
    return {"id": f"{username}/{slug}", "title": slug, "code_file": SCRIPT_NAME, "language": "python",
            "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
            "dataset_sources": [f"{username}/{slug}"], "competition_sources": [], "kernel_sources": []}


def normalize_status(raw: object) -> str:
    text = str(getattr(raw, "status", raw)).lower().rsplit(".", 1)[-1]
    for word, status in STATUS_WORDS.items():
        if word in text:
            return status
    return RUNNING


def translate(exc: Exception) -> KaggleError:
    message = str(exc)
    if any(hint in message.lower() for hint in QUOTA_HINTS):
        return QuotaExceeded(f"cota do Kaggle esgotada: {message}")
    return KaggleError(f"falha no Kaggle: {message}")


class KaggleClient:
    def __init__(self, username: str, api: Any = None) -> None:
        self._username = username
        self._api = api

    def _client(self) -> Any:
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    def create_dataset(self, folder: Path, slug: str) -> None:
        (folder / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(self._username, slug)), encoding="utf-8")
        try:
            self._client().dataset_create_new(folder=str(folder), public=False, quiet=True, dir_mode="zip")
        except Exception as exc:  # the library raises many types; they all become KaggleError
            raise translate(exc) from exc

    def push_kernel(self, folder: Path, slug: str, script: Path) -> str:
        kernel_dir = folder / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script, kernel_dir / SCRIPT_NAME)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(self._username, slug)), encoding="utf-8")
        try:
            self._client().kernels_push(str(kernel_dir))
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc
        return f"{self._username}/{slug}"

    def status(self, kernel: str) -> str:
        try:
            return normalize_status(self._client().kernels_status(kernel))
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc

    def download(self, kernel: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        try:
            self._client().kernels_output(kernel, path=str(destination), quiet=True)
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc

    def delete_dataset(self, slug: str) -> None:
        delete = getattr(self._client(), "dataset_delete", None)
        if delete is None:
            raise KaggleError("esta versão do kaggle não apaga datasets: apague pelo site")
        try:
            delete(f"{self._username}/{slug}")
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc
