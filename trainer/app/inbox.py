"""Turns the photos n8n drops in the inbox into anonymized receiver strips; the original never outlives the cut."""
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .anonymizer import pseudonym
from .consensus import Sample
from .cropper import CropError, receiver_strip

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}
SECONDS_PER_DAY = 86400
WAIT_DAYS_FOR_DECISION = 3  # an original whose decision never shows up is not kept
STRIP_TTL_DAYS = 30  # a strip that never reached a consensus is not kept either
STRIP_SUFFIX = ".jpg"


@dataclass(frozen=True)
class IngestReport:
    cut: int = 0
    waiting: int = 0
    dropped: int = 0


def strip_path(strips: Path, secret: str, item_id: str) -> Path:
    return strips / f"{pseudonym(secret, item_id)}{STRIP_SUFFIX}"


def _photos(inbox: Path) -> list[Path]:
    if not inbox.is_dir():
        return []
    return [path for path in sorted(inbox.iterdir()) if path.is_file()]


def _cut(photo: Path, sample: Sample, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(receiver_strip(photo.read_bytes(), sample.rotation))


def ingest(inbox: Path, strips: Path, samples: Mapping[str, Sample], secret: str,
           clock: Callable[[], float] = time.time) -> IngestReport:
    """Cuts every photo whose decision is known, then deletes it; unsafe names, bad images and stale files are dropped."""
    cut = waiting = dropped = 0
    for photo in _photos(inbox):
        sample = samples.get(photo.stem)
        stale = clock() - photo.stat().st_mtime > WAIT_DAYS_FOR_DECISION * SECONDS_PER_DAY
        if photo.suffix.lower() not in PHOTO_SUFFIXES or not SAFE_ID.match(photo.stem):
            photo.unlink()
            dropped += 1
        elif sample is None and not stale:
            waiting += 1
        elif sample is None:
            photo.unlink()
            dropped += 1
        else:
            try:
                _cut(photo, sample, strip_path(strips, secret, photo.stem))
                cut += 1
            except CropError:
                dropped += 1
            photo.unlink()
    return IngestReport(cut, waiting, dropped)


def prune_strips(strips: Path, clock: Callable[[], float] = time.time) -> int:
    if not strips.is_dir():
        return 0
    old = [path for path in strips.glob(f"*{STRIP_SUFFIX}") if clock() - path.stat().st_mtime > STRIP_TTL_DAYS * SECONDS_PER_DAY]
    for path in old:
        path.unlink()
    return len(old)
