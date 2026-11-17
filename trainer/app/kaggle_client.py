"""Kaggle gateway: private dataset, one GPU kernel, output download and dataset deletion.

Written against `kaggle` 2.2.5 (pinned): create and push report errors in the response instead of raising.
Not yet exercised against the real service.
"""
import json
import shutil
import time
from pathlib import Path
from typing import Any, Protocol

RUNNING, COMPLETE, FAILED = "rodando", "completo", "falhou"
QUOTA_HINTS = ("quota", "429", "too many requests", "limit")
STATUS_WORDS = {"complete": COMPLETE, "error": FAILED, "cancel": FAILED, "queued": RUNNING, "running": RUNNING}
LICENSE = "CC0-1.0"
SCRIPT_NAME = "treino.py"
DATASET_READY, PENDING = "ready", "pending"
NOT_VISIBLE_YET = (403, 404)  # a fresh dataset is denied to its own owner until Kaggle propagates it
DATASET_WAIT_SECONDS, DATASET_POLL_SECONDS = 180, 5  # Kaggle answers 409 to a kernel pushed before its dataset is ready
BODY_LIMIT = 300


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


def checked(response: Any) -> Any:
    """kaggle 2.x returns refusals inside the response; they must stop the job like a raised error."""
    error = getattr(response, "error", None)
    if error:
        raise translate(RuntimeError(str(error)))
    if getattr(response, "invalid_dataset_sources", None):
        raise KaggleError(f"o kernel não encontrou o dataset: {response.invalid_dataset_sources}")
    return response


def translate(exc: Exception) -> KaggleError:
    message = _with_body(exc)
    if any(hint in message.lower() for hint in QUOTA_HINTS):
        return QuotaExceeded(f"cota do Kaggle esgotada: {message}")
    return KaggleError(f"falha no Kaggle: {message}")


def _with_body(exc: Exception) -> str:
    """The SDK keeps the refusal reason in the HTTP body, which str(exc) drops."""
    body = str(getattr(getattr(exc, "response", None), "text", "") or "").strip()
    return f"{exc} | {body[:BODY_LIMIT]}" if body else str(exc)


class KaggleClient:
    def __init__(self, username: str, api: Any = None, sleep: Any = time.sleep) -> None:
        self._username = username
        self._api = api
        self._sleep = sleep

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
            response = self._client().dataset_create_new(folder=str(folder), public=False, quiet=True, dir_mode="zip")
        except Exception as exc:  # the library raises many types; they all become KaggleError
            raise translate(exc) from exc
        checked(response)

    def _dataset_status(self, slug: str) -> str:
        try:
            return str(self._client().dataset_status(f"{self._username}/{slug}")).lower()
        except Exception as exc:  # see create_dataset
            if getattr(getattr(exc, "response", None), "status_code", None) in NOT_VISIBLE_YET:
                return PENDING
            raise translate(exc) from exc

    def _wait_dataset_ready(self, slug: str) -> None:
        waited = 0
        while not (status := self._dataset_status(slug)).endswith(DATASET_READY):
            if waited >= DATASET_WAIT_SECONDS:
                raise KaggleError(f"o dataset {slug} não ficou pronto em {DATASET_WAIT_SECONDS}s (status: {status})")
            self._sleep(DATASET_POLL_SECONDS)
            waited += DATASET_POLL_SECONDS

    def push_kernel(self, folder: Path, slug: str, script: Path) -> str:
        self._wait_dataset_ready(slug)
        kernel_dir = folder / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(script, kernel_dir / SCRIPT_NAME)
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(self._username, slug)), encoding="utf-8")
        try:
            response = self._client().kernels_push(str(kernel_dir))
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc
        checked(response)
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
            deleted = delete(self._username, slug, no_confirm=True)
        except Exception as exc:  # see create_dataset
            raise translate(exc) from exc
        if deleted is False:
            raise KaggleError(f"o Kaggle não confirmou a exclusão do dataset {slug}")
