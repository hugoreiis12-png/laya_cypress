"""Leaves each photo in the trainer inbox on this host, because the n8n that sends it runs on another machine."""
import contextlib
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # the same rule the trainer applies before reading a photo
PNG_MAGIC = b"\x89PNG"
PARTIAL_PREFIX = "."  # the trainer skips hidden files, so it never reads a photo still being written


def _suffix(data: bytes) -> str:
    return ".png" if data.startswith(PNG_MAGIC) else ".jpg"


def drop(directory: str, record_id: str | None, data: bytes) -> bool:
    """Never raises: a photo that cannot be left only costs one training sample."""
    if not directory or not record_id or not SAFE_ID.match(record_id):
        return False
    target = Path(directory) / f"{record_id}{_suffix(data)}"
    partial = target.with_name(f"{PARTIAL_PREFIX}{target.name}.part")
    try:
        partial.write_bytes(data)
        os.replace(partial, target)
    except OSError as exc:
        log.warning("foto não entregue ao trainer: %s", exc)
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        return False
    return True
