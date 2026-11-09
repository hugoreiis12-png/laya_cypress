"""Turns the photos n8n drops in the inbox into anonymized derivatives; the original never outlives the cut."""
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .anonymizer import pseudonym
from .cropper import receiver_strip

SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}
SECONDS_PER_DAY = 86400
WAIT_DAYS_FOR_DECISION = 3  # an original whose decision never shows up is not kept
KEPT_DAYS = 30  # a derivative that never reached a consensus is not kept either
STRIP_SUFFIX = ".jpg"
Cutter = Callable[[bytes, int, str], None]  # (photo bytes, rotation from the decision, raw id) -> writes a derivative


@dataclass(frozen=True)
class IngestReport:
    cut: int = 0
    waiting: int = 0
    dropped: int = 0


def derived_path(directory: Path, secret: str, item_id: str, suffix: str = STRIP_SUFFIX) -> Path:
    return directory / f"{pseudonym(secret, item_id)}{suffix}"


def strip_path(strips: Path, secret: str, item_id: str) -> Path:
    return derived_path(strips, secret, item_id)


def strip_cutter(strips: Path, secret: str) -> Cutter:
    def cut(data: bytes, rotation: int, item_id: str) -> None:
        target = strip_path(strips, secret, item_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(receiver_strip(data, rotation))
    return cut


def _photos(inbox: Path) -> list[Path]:
    return [path for path in sorted(inbox.iterdir()) if path.is_file()] if inbox.is_dir() else []


def _process(photo: Path, rotation: int, cutters: Sequence[Cutter]) -> bool:
    try:
        data = photo.read_bytes()
        for cutter in cutters:
            cutter(data, rotation, photo.stem)
        return True
    except ValueError:  # unreadable image, empty OCR or strip too small: dropped, never guessed
        return False


def ingest(inbox: Path, known: Mapping[str, int], cutters: Sequence[Cutter],
           clock: Callable[[], float] = time.time) -> IngestReport:
    """Runs every cutter on a photo whose decision is known (id -> rotation), then deletes it."""
    cut = waiting = dropped = 0
    for photo in _photos(inbox):
        stale = clock() - photo.stat().st_mtime > WAIT_DAYS_FOR_DECISION * SECONDS_PER_DAY
        unsafe = photo.suffix.lower() not in PHOTO_SUFFIXES or not SAFE_ID.match(photo.stem)
        if not unsafe and photo.stem not in known and not stale:
            waiting += 1
            continue
        done = not unsafe and photo.stem in known and _process(photo, known[photo.stem], cutters)
        photo.unlink()
        cut, dropped = cut + done, dropped + (not done)
    return IngestReport(cut, waiting, dropped)


def prune(directory: Path, clock: Callable[[], float] = time.time) -> int:
    if not directory.is_dir():
        return 0
    old = [path for path in directory.iterdir() if path.is_file() and clock() - path.stat().st_mtime > KEPT_DAYS * SECONDS_PER_DAY]
    for path in old:
        path.unlink()
    return len(old)
