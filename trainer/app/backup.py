"""Failure backup on the trainer volume: the job folder and why it failed, kept until the resumed job succeeds."""
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json

BACKUPS_DIR, FAILURE_FILE, PACKAGE_DIR, MANIFEST_FILE = "backups", "falha.json", "pacote", "manifest.json"
MAX_REASON_CHARS = 500
JOB_KEYS = ("dataset", "kernel", "samples")


def save(state_dir: Path, kind: str, job: Mapping[str, Any], reason: str, now: float) -> dict[str, Any]:
    """Moves the job folder (package, kernel, any output) under backups/ and records the failure beside it."""
    workdir = Path(job["workdir"])
    target = state_dir / BACKUPS_DIR / workdir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if workdir.is_dir():
        shutil.move(str(workdir), str(target))
    target.mkdir(parents=True, exist_ok=True)
    failure = {"tipo": kind, "motivo": reason[:MAX_REASON_CHARS], "em": datetime.fromtimestamp(now).isoformat(),
               "backup": str(target), "job": {key: job[key] for key in JOB_KEYS if key in job}}
    write_json(target / FAILURE_FILE, failure)
    return failure


def reusable_package(folder: str | None) -> tuple[Path, int] | None:
    """The backed-up package and its sample count, when the failed job got as far as building one."""
    if not folder:
        return None
    base = Path(folder)
    samples = read_json(base / FAILURE_FILE, {}).get("job", {}).get("samples")
    package = base / PACKAGE_DIR
    if not (package / MANIFEST_FILE).is_file() or not isinstance(samples, int):
        return None
    return package, samples


def restore(package: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package, target)


def discard(folder: str | None) -> None:
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
