"""The real training data source: strips cut from the inbox, labeled by the consensus of the decision log."""
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from . import collector, consensus, inbox
from .anonymizer import pseudonym
from .config import Settings
from .cycle import DataError
from .packager import StripItem, write_vision_package
from .state import KIND_VISION

log = logging.getLogger(__name__)
INBOX_DIR, STRIPS_DIR = "inbox", "tiras"
CLASSES = (consensus.LABEL_SIGNED, consensus.LABEL_UNSIGNED)
MIN_PER_CLASS = 2  # a package with one class (or one example) cannot be split and measured


class InboxData:
    def __init__(self, settings: Settings, clock: Callable[[], float] | None = None) -> None:
        self._settings = settings
        self._inbox, self._strips = settings.state_dir / INBOX_DIR, settings.state_dir / STRIPS_DIR
        self._clock = clock

    def _ready(self) -> list[tuple[str, str]]:
        """(pseudonym, label) of every agreed photo that already has its strip."""
        if not self._settings.decisions_path.is_file():
            return []
        samples = collector.load_samples(self._settings.decisions_path).samples
        kwargs = {"clock": self._clock} if self._clock else {}
        report = inbox.ingest(self._inbox, self._strips, {s.item_id: s for s in samples}, self._settings.id_secret, **kwargs)
        pruned = inbox.prune_strips(self._strips, **kwargs)
        log.info("caixa de entrada: %d recortadas, %d aguardando, %d descartadas, %d tiras vencidas",
                 report.cut, report.waiting, report.dropped, pruned)
        training = consensus.partition(list(samples)).training
        secret = self._settings.id_secret
        return [(pseudonym(secret, sample.item_id), label) for sample, label in training
                if inbox.strip_path(self._strips, secret, sample.item_id).is_file()]

    def new_samples(self, trained: Mapping[str, int]) -> Mapping[str, int]:
        ready = len(self._ready())
        return {KIND_VISION: max(0, ready - trained.get(KIND_VISION, 0))}

    def prepare(self, kind: str, directory: Path) -> int:
        if kind != KIND_VISION:
            raise DataError(f"a fonte de fotos só atende {KIND_VISION}")
        items = [StripItem(name, label, (self._strips / f"{name}{inbox.STRIP_SUFFIX}").read_bytes())
                 for name, label in self._ready()]
        for label in CLASSES:
            if sum(1 for item in items if item.label == label) < MIN_PER_CLASS:
                raise DataError(f"poucos exemplos de {label} para treinar")
        write_vision_package(directory, directory.parent.name, items, self._settings.test_percent)
        return len(items)
